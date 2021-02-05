import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import glob

class ZumDataset(data.Dataset):

    def __init__(self, root, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        imgs = glob.glob(os.path.join(root, "*/*.png"))
        labels = [i.split("/")[-2] for i in imgs]
        classes = list(set(labels))
        
        self.data = [{"imgs": i, "labels": classes.index(j)} for i, j in zip(imgs, labels)]

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        d = self.data[index]
        
        img = d["imgs"]
        label = d["labels"]
        
        data = Image.open(img)
        data = data.convert('L')
        
        data = cv2.imread(img)
        data = cv2.resize(data, (self.input_shape[1:]))
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        data = self.transforms(data)
        
        label = np.int32(label)
        
        return data.float(), label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Dataset(root="/home/kbj/projects/similar_celeb/train_face_img/",
                      phase='train',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (img, label) in enumerate(trainloader):
        print(img.shape)
        print(type(img))
        
        print(label.shape)
        print(type(label))
        break