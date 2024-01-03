import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms as T


class ZooniverseLabelGenerator(Dataset):
    AUGMENTATION = T.Resize(96)

    def __init__(self, imgfolder, label_file, indices=None):
        super().__init__()
        self.imgfolder = imgfolder
        self.labels = np.load(label_file)

        self.imgs = np.asarray(sorted(glob.glob(self.imgfolder + "*.png")))
        self.subject_ids = np.asarray([int(os.path.splitext(os.path.basename(e))[0]) for e in self.imgs])

        assert len(self.labels) == len(
            self.imgs), f"List of imgs ({len(self.imgs)}) and labels ({len(self.labels)})should match!"

        if indices is not None:
            self.indices = indices
            self.ndata = len(indices)
        else:
            self.ndata = len(self.imgs)
            self.indices = np.arange(self.ndata)

        print(f"Found data with {self.ndata} images")

    def __len__(self):
        return self.ndata

    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        img_fname = self.imgs[index]

        img = self.AUGMENTATION(read_image(img_fname, ImageReadMode.RGB) / 255.)
        label = torch.Tensor(self.labels[self.labels[:, 0] == subject_id, 1:].tolist()[0])

        return img, label
