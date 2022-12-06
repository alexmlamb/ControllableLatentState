import argparse
import distutils
import distutils.util
import os
import random
import time
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torch.autograd import grad, Variable
from torchvision.utils import save_image
from data import ArmDataLoader, filter_invalid
from vit import ViTGenik
from tqdm import tqdm


class Trainer:

    def __init__(self):
        parser = argparse.ArgumentParser(description='Robot Task',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--task', type=str, default='genik', choices=['pred_exo', 'pred_gt', 'genik'])
        parser.add_argument('--batch_size', type=int, default=128)  # 256
        parser.add_argument('--iteration', type=int, default=100)
        parser.add_argument('--max_k', type=int, default=5)
        parser.add_argument('--clip', type=float, default=10.0)
        parser.add_argument("--use_vq", type=lambda x: bool(distutils.util.strtobool(x)), default=False,
                            help="Use VQ Discrete Bottleneck")
        parser.add_argument("--use_gb", type=lambda x: bool(distutils.util.strtobool(x)), default=False,
                            help="Use Gaussian bottleneck")
        parser.add_argument("--kl_penalty", type=float, default=1e-3)
        parser.add_argument("--genik_lossweight", type=float, default=1)
        parser.add_argument("--train_ae", type=lambda x: bool(distutils.util.strtobool(x)), default=False)
        parser.add_argument('--ncodes', type=int, default=2048)
        parser.add_argument('--data_root', type=str, default='../medium_hard/images/')
        parser.add_argument('--csv_file', type=str,
                            default='/home/lambalex/discrete-factors/robotexp/medium_hard/run.csv')
        self.args = parser.parse_args()
        print('args', self.args)

        self.job_dir = "aetrain_%s_maxk_%d_kl_%f" % (str(self.args.train_ae), self.args.max_k, self.args.kl_penalty)
        try:
            os.mkdir('./results/%s' % self.job_dir)
        except:
            pass

        data_loader = ArmDataLoader()

        self.train_dataloader, f2g, f2a, g2ind, a2ind = data_loader.get_data(self.args.batch_size, self.args.max_k,
                                                                 root=self.args.data_root,
                                                                 csv_file=self.args.csv_file)

        self.model = ViTGenik(image_size=256, patch_size=16, num_classes_1=9, num_classes_2=5, num_classes_3=2000,
                              dim=256,
                              depth=6, heads=4, mlp_dim=512, dim_head=256 // 4, args=self.args,
                              use_gb=self.args.use_gb, vq=self.args.use_vq)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        self.ce_loss_1 = nn.CrossEntropyLoss()
        self.ce_loss_2 = nn.CrossEntropyLoss()
        self.model.best_acc = 0.0

    def run(self):
        for e in range(self.args.iteration):
            self.train(e, evaluate=False)
            self.train(e, evaluate=True)
            torch.save(self.model.state_dict(), "model" + str(self.args.use_gb) + "_" + str(self.args.use_vq) + ".pt")

    def train(self, epoch, evaluate):
        if evaluate:
            self.model.eval()
        else:
            self.model.train()

        if epoch >= 0 and self.args.use_gb:  # epoch2
            use_gb = True
            use_vq = self.args.use_vq
        else:
            use_gb = False
            use_vq = False

        total_loss_1 = 0
        total_loss_2 = 0
        total_loss_3 = 0
        total_loss_4 = 0

        acc_1 = 0
        acc_2 = 0
        acc_3 = 0

        total = 0
        num_images_used = 0
        for data in tqdm(self.train_dataloader):
            x, gt, action, ts_ind, _, x_k, k = data
            num_images_used += x.size()[0]

            if torch.cuda.is_available():
                x = x.float().cuda()
                x_k = x_k.float().cuda()
                gt = gt.cuda()
                action = action.cuda()
                ts_ind = ts_ind.cuda()
                k = k.cuda()
            else:
                x = x.float()
                x_k = x_k.float()
                gt = gt
                action = action
                ts_ind = ts_ind
                k = k

            x, gt, action, ts_ind, x_k, k = filter_invalid(x, gt, action, ts_ind, x_k, k)

            x = Variable(x, requires_grad=True)

            if self.args.task == 'genik':
                preds_1, preds_2, preds_3, h_rep, d, codes, loss = self.model(x, x_k, k, action, vq=use_vq,
                                                                              use_gb=use_gb)

            else:
                preds_1, preds_2, preds_3, h_rep, loss = self.model(x)

            ts_ind_coarse = torch.div(ts_ind, 10, rounding_mode='trunc')

            loss_1 = self.ce_loss_1(preds_1, gt)
            loss_2 = self.ce_loss_1(preds_2, action)
            loss_3 = self.ce_loss_1(preds_3, ts_ind_coarse)

            self.optimizer.zero_grad()

            do_saliency = True
            if evaluate and do_saliency and random.uniform(0, 1) < 0.05:
                # ymax = preds_2[:,preds_2.argmax(1)]

                ymax = (torch.abs(h_rep)).sum()

                g = torch.abs(grad(ymax, x, retain_graph=True)[0].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1))
                # print(g.shape)
                # print(x.shape)
                # print('g min max', g.min(), g.max())

                g = g / g.amax(dim=(1, 2, 3), keepdim=True)
                g[:, 1:] *= 0.0
                o = (g * 1.0) + x * 0.6

                save_image(x[0:8], 'results/%s/sal_x.png' % self.job_dir)
                save_image(g[0:8], 'results/%s/sal_g.png' % self.job_dir)
                save_image(o[0:8], 'results/%s/sal_o.png' % self.job_dir)
                save_image(d[0:8], 'results/%s/sal_rec.png' % self.job_dir)

                # torch.save(model.state_dict(), 'model.pt')

            (loss_1 + self.args.genik_lossweight * loss_2 + loss_3 + loss).backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()

            total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()
            total_loss_3 += loss_3.item()
            total_loss_4 += loss.item()

            preds_1_argmax = torch.argmax(preds_1, dim=1)
            preds_2_argmax = torch.argmax(preds_2, dim=1)
            preds_3_argmax = torch.argmax(preds_3, dim=1)

            acc_1 += (gt == preds_1_argmax).sum() / x.shape[0]

            acc_2 += (action == preds_2_argmax).sum() / x.shape[0]

            acc_3 += (ts_ind_coarse == preds_3_argmax).sum() / x.shape[0]

            total += 1

        total_loss_1 = total_loss_1 / total
        total_loss_2 = total_loss_2 / total
        total_loss_3 = total_loss_3 / total
        total_loss_4 = total_loss_4 / total

        acc_1 = acc_1 / total
        acc_2 = acc_2 / total
        acc_3 = acc_3 / total

        if evaluate and (True or acc_1 > self.model.best_acc):
            torch.save(self.model.state_dict(), './results/%s/model_%d.pt' % (self.job_dir, epoch))
            print('saving model!')
            self.model.best_acc = acc_1

        print("Epoch:" + str(epoch) + " ImagesUsed:" + str(num_images_used) + " Loss 1:" + str(
            total_loss_1) + " Loss 2:" + str(total_loss_2) + " Loss 3:" + str(total_loss_3) + " Loss 4:" + str(
            total_loss_4) + " Acc 1:" + str(acc_1.item() * 100) + " Acc 2:" + str(acc_2.item() * 100) + " Acc 3:" + str(
            acc_3.item() * 100))


if __name__ == "__main__":
    print("freeze_support()")
    freeze_support()
    trainer = Trainer()
    trainer.run()
