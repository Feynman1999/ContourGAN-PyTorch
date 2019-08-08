import torch
from torchvision import datasets
import matplotlib.pyplot as plt


def filename(item):
    return item[0]


def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    idx = dset.class_to_idx[subfolder]  # 1    dset.class_to_idx is a dict  test->0  and train1->1
    n = 0
    for i in range(dset.__len__()):
        if dset.imgs[n][1] != idx:
            del dset.imgs[n]
            n -= 1
        n += 1
    # 按照文件名排序
    dset.imgs.sort(key=filename)
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def print_network(net, framework_flag=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if(framework_flag):
        print(net)
    print('Total number of parameters: %d' % num_params)


def map_0_1(vector):
    for i in range(vector.size()[0]):
        min_v = torch.min(vector[i])
        range_v = torch.max(vector[i]) - min_v
        if range_v > 0:
            ans = (vector[i] - min_v) / range_v
        else:
            ans = torch.zeros(vector[i].size())
        vector[i] = ans
    return vector


def save_image(img1, img2, path):
    if img2.shape[0] == 1:
        img2 = img2.repeat(3, 1, 1)
    result = torch.cat([map_0_1(img1.cpu()), img2.cpu().round()], 2)
    plt.imsave(path, (result.numpy().transpose(1, 2, 0)))