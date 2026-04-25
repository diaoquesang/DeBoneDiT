from config import config
from torchvision import transforms
import cv2 as cv
import torchvision.transforms.functional as TF
import random


class JointTransformMethod:
    def __call__(self, img, label):
        img = transforms.ToPILImage()(img).convert('L')
        label = transforms.ToPILImage()(label).convert('L')

        if random.random() > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)

        if random.random() > 0.2:  # 80%的概率做裁切
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            img = TF.resized_crop(img, i, j, h, w, (config.image_size, config.image_size))
            label = TF.resized_crop(label, i, j, h, w, (config.image_size, config.image_size))
        else:
            img = TF.resize(img, (config.image_size, config.image_size))
            label = TF.resize(label, (config.image_size, config.image_size))

        img = TF.to_tensor(img)
        label = TF.to_tensor(label)

        img = (img - 0.5) / 0.5
        label = (label - 0.5) / 0.5

        return img, label


class TestTransformMethod:
    def __call__(self, img):
        img = cv.resize(img, (config.image_size, config.image_size))
        if len(img.shape) == 2:
            img = img[:, :, None]  # H,W,1

        img = transforms.ToTensor()(img)
        img = (img - 0.5) / 0.5
        return img


myDiTTransform = {
    'trainTransform': JointTransformMethod(),
    'testTransform': TestTransformMethod()
}


class myTransformMethod():
    def __call__(self, img):
        img = cv.resize(img, (config.image_size, config.image_size))
        if img.shape[-1] == 3:  # HWC
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img


myTransform = {
    'trainTransform': transforms.Compose([
        myTransformMethod(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'testTransform': transforms.Compose([
        myTransformMethod(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),

}
