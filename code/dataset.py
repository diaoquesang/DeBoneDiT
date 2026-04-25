from torch.utils.data import Dataset
import pandas as pd
import cv2 as cv
import os


class myDataset(Dataset):
    def __init__(self, filelist, cxr_dir, bs_dir,
                 transform=None):
        self.cxr_dir = cxr_dir
        self.bs_dir = bs_dir

        self.transform = transform
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file = self.filelist.iloc[idx, 0]
        cxr = cv.imread(os.path.join(self.cxr_dir, file))
        bs = cv.imread(os.path.join(self.bs_dir, file))

        if self.transform:
            cxr = self.transform(cxr)
            bs = self.transform(bs)

        return cxr, bs, file


class myDiTDataset(Dataset):
    def __init__(self, filelist, cxr_dir, bs_dir,
                 transform=None):
        self.cxr_dir = cxr_dir
        self.bs_dir = bs_dir

        self.transform = transform
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file = self.filelist.iloc[idx, 0]
        cxr = cv.imread(os.path.join(self.cxr_dir, file))
        bs = cv.imread(os.path.join(self.bs_dir, file))

        if self.transform:
            cxr, bs = self.transform(cxr, bs)

        return cxr, bs, file


class mySingleDataset(Dataset):
    def __init__(self, filelist, cxr_dir,
                 transform=None):
        self.cxr_dir = cxr_dir

        self.transform = transform
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file = self.filelist.iloc[idx, 0]
        cxr = cv.imread(os.path.join(self.cxr_dir, file))

        if self.transform:
            cxr = self.transform(cxr)

        return cxr, file
