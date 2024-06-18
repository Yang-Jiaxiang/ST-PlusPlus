from dataset.transform import crop, hflip, normalize, resize, blur, cutout

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms

def _findContours(image, threshold):
    src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgh,imgw = src.shape
    
    # src: 輸入的灰度圖像。
    # threshold_value: 閾值。如果像素值大於這個閾值，則設置為 max_value，否則設置為 0。
    # max_value: 當像素值超過閾值時分配的值。通常設置為 255，表示白色。
    # threshold_type: 閾值類型。cv2.THRESH_BINARY 表示使用二值化處理。
    gray = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # gray: 輸入的二值化圖像。在此例中，它是由 cv2.threshold 函數生成的。
    # cv2.RETR_EXTERNAL: 輪廓檢索模式。cv2.RETR_EXTERNAL 只檢測最外層的輪廓。這意味著，如果有多個嵌套的輪廓（如一個輪廓在另一個輪廓內部），只有最外層的輪廓會被檢測到。
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    maxm=0
    #fig = d2l.plt.imshow(src, cmap='gray')
    for i in range(0, len(contours)):

        # cv2.boundingRect: 透過輪廓找到外接矩形
        # 輸出：(x, y)矩形左上角座標、w 矩形寬(x軸方向)、h 矩形高(y軸方向)
        x, y, w, h = cv2.boundingRect(contours[i])
        if w<imgw and h<imgh and w+h>maxm:
            maxm=w+h
            index=i
        # 在原影像上繪製出矩形
        '''fig.axes.add_patch(d2l.plt.Rectangle((x, y), w, h, fill=False,
                           linestyle="-", edgecolor=color,
                           linewidth=2))'''
    x, y, w, h = cv2.boundingRect(contours[index])
    
    return x, y, w, h

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        else:
            if mode == 'val':
                id_path = 'dataset/splits/%s/val.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))

        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        base_size = 400 if self.name == 'pascal' else 2048
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)
