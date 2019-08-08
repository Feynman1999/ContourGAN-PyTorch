import cv2
import numpy as np
import os

path1 = './data/source/train1/'
path2 = './data/target/edge1/'
path3 = './data/target/edge1_not_closed/'
path4 = './data/target/four_kinds_img/'

class ImageSizeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def crop_and_resize(img):
    assert len(img.shape) == 3
    if img.shape[0] == 321:
        img = img[:, 80:401, :]
    else:
        img = img[80:401, :, :]
    img = cv2.resize(img,(400,400))
    return img


def edge_smooth(img):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, sigma=0)
    # If sigma is non-positive, it is computed from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 .
    gauss = gauss * gauss.transpose(1, 0)

    if img.shape[0] != 400 or img.shape[1] != 400:
        raise ImageSizeError('please makesure imagesize is 400 by 400')

    pad_img = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
    gaussed_img = np.copy(img)
    edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    idx = np.where(edges != 0)

    for i in range(np.sum(edges != 0)):
        x = idx[0][i]
        y = idx[1][i]
        gaussed_img[x, y, 0] = np.sum(np.multiply(pad_img[x:x+kernel_size, y:y+kernel_size, 0], gauss))
        gaussed_img[x, y, 1] = np.sum(np.multiply(pad_img[x:x+kernel_size, y:y+kernel_size, 1], gauss))
        gaussed_img[x, y, 2] = np.sum(np.multiply(pad_img[x:x+kernel_size, y:y+kernel_size, 2], gauss))

    return gaussed_img



n=1

for img in os.listdir(path1):
    # print(os.path.splitext(img)[0])
    filename = os.path.splitext(img)[0]
    img1 = cv2.imread(os.path.join(path1, img))
    img1 = crop_and_resize(img1)

    img2 = cv2.imread(os.path.join(path2, filename+'.png'))
    img2 = crop_and_resize(img2)

    img3 = cv2.imread(os.path.join(path3, filename+'.png'))
    img3 = crop_and_resize(img3)

    # img4 = edge_smooth(img2)

    # result = np.concatenate((img1, img2, img3), 1)
    # cv2.imwrite(os.path.join(path1, filename + '.png'), img1)
    # cv2.imwrite(os.path.join(path2, filename + '.png'), img2)
    cv2.imwrite(os.path.join(path3, filename + '.png'), img3)
    n += 1
