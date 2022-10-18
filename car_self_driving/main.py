
import torch
from data import get_data_loader
from vit import ViTGenik
import argparse
import torchvision.transforms.functional as F
from tqdm import tqdm
import torch.nn as nn
import distutils
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Robot Task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', type=str, default='genik', choices=[ 'pred_exo', 'pred_gt', 'genik'])

parser.add_argument('--batch_size', type=int, default=128) #256

parser.add_argument('--iteration', type=int, default=400)

parser.add_argument('--max_k', type=int, default=50)
parser.add_argument('--clip', type=float, default=10.0)

parser.add_argument("--use_vq", type=lambda x:bool(distutils.util.strtobool(x)), default=False, help="Use VQ Discrete Bottleneck")

parser.add_argument("--use_gb", type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="Use Gaussian bottleneck")

parser.add_argument("--kl_penalty", type=float, default=1e-4)

parser.add_argument("--genik_lossweight", type=float, default=1)

parser.add_argument("--train_ae", type=lambda x:bool(distutils.util.strtobool(x)), default=False)

parser.add_argument('--ncodes', type=int, default=2048)

args = parser.parse_args()


if __name__ == "__main__":

    loader = get_data_loader()

    model = ViTGenik(image_size = 256, patch_size =  16, dim = 256, depth = 6, heads = 4, mlp_dim = 512, dim_head = 256 // 4, args=args, use_gb=args.use_gb, vq=args.use_vq).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=1e-4)

    ls = nn.L1Loss()

    for iteration in range(args.iteration):

        tloss = 0
        ncount = 0


        for x, xk, k, a1, a2, a3, ind in tqdm(loader):
            x = F.resize(x, (192,320)).cuda()
            xk = F.resize(xk, (192,320)).cuda()


            k = k.cuda()

            ap1, ap2, ap3, h, xrec, codes, extra_loss = model(x, xk, k, use_gb=True)

            l1 = ls(ap1.flatten(), a1.cuda())
            l2 = ls(ap2.flatten(), a2.cuda())
            l3 = ls(ap3.flatten(), a3.cuda())

            loss = l1+l2+l3+extra_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tloss += l1.data+l2.data+l3.data
            ncount += 1
 
            if ncount % 500 == 1:
                print("Loss", iteration, ncount, tloss/ncount)

                save_image(xrec.cpu().data, 'results/xrec.png')
                save_image(x.cpu().data, 'results/x.png')
        
        torch.save(model, 'model.pt')



