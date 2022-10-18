
import h5py
import numpy as np
import torch
import torchvision
import random

def get_dataset(data_c, data_l):

    #data_c = '/data/car_self_driving_comma/dataset/camera/2016-01-30--11-24-51.h5'
    #data_l = '/data/car_self_driving_comma/dataset/log/2016-01-30--11-24-51.h5'

    c = h5py.File(data_c, 'r')
    l = h5py.File(data_l, 'r')

    a1 = np.array(l['speed'])
    a2 = np.array(l['steering_angle']) * (3.14/180.0)
    a3 = np.array(l['car_accel'])
    obs_ind = np.array(l['cam1_ptr'])
    X = np.array(c['X'])

    class CarData(torch.utils.data.Dataset):       

        def __len__(self):
            return a1.shape[0]

        def __getitem__(self, index):

            if index == self.__len__()-1:
                index -= 1

            oi = int(obs_ind[index])

            max_k = 199

            rindex = random.randint(index+1, min(self.__len__()-1, index+max_k))

            k = rindex - index

            koi = int(obs_ind[rindex])

            xt = torch.Tensor(X[oi]).float() / 255.0
            xtk = torch.Tensor(X[koi]).float() / 255.0

            return xt, xtk, k, a1[index], a2[index], a3[index], oi

            #if index == len(self.imgs) - 1:
            #    index = len(self.imgs) - 2

            #rindex = random.randint(index+1, min(len(self.imgs)-1, index+max_k))
            #k = rindex - index
            #assert k >= 1
            #assert k <= max_k
            #other_img = super(ImageFolderWithPaths, self).__getitem__(rindex)[0]
            # the image file path
            #path = self.imgs[index][0]

            #path = path[path.rfind('/')+1:].rstrip('.jpg')

            #if path in f2g:
            #    gt_ind = g2ind[f2g[path]]
            #    a_ind = a2ind[f2a[path]]
            #else:
                #print('not found in csv', path)
            #    gt_ind = -1
            #    a_ind = -1

            # make a new tuple that includes original and the path
            #tuple_with_path = (original_tuple[0:1] + (gt_ind, a_ind, index//(2), path,other_img,k))
            #return tuple_with_path

    return CarData()

def get_data_loader(shuffle=True):

    root = '/data/car_self_driving_comma/dataset/'

    #dlst = ['2016-01-30--11-24-51.h5', '2016-01-31--19-19-25.h5', '2016-02-08--14-56-28.h5', '2016-03-29--10-50-20.h5', '2016-05-12--22-20-00.h5','2016-06-08--11-46-01.h5','2016-01-30--13-46-00.h5','2016-02-02--10-16-58.h5','2016-02-11--21-32-47.h5','2016-04-21--14-48-08.h5','2016-06-02--21-39-29.h5']

    dlst = ['2016-03-29--10-50-20.h5']

    sl = []
    for d in dlst:
        sl.append(get_dataset(root + 'camera/' + d, root + 'log/' + d))

    dataset = torch.utils.data.ConcatDataset(sl)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=shuffle, num_workers=8, pin_memory=True)

    return data_loader


if __name__ == "__main__":
    loader = get_data_loader()

    for x, xk, k, a1, a2, a3 in loader:
        print(x.shape, x.min(), x.max())


