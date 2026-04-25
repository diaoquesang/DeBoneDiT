import time
from generative.networks.nets import VQVAE
import matplotlib.pyplot as plt
import torch
from monai.config import print_config
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from tqdm import tqdm
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from datetime import date
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from depth_loss import depth_loss

print_config()

set_determinism(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 1024
vae_batch_size = 4
n_example_images = 4
vae_epoch_number = 200
val_interval = 10

train_file_list = "SZCH-X-Rays_trainset.txt"
test_file_list = "SZCH-X-Rays_valset.txt"

cxr_path = "SZCH-X-Rays-741/CXR"
bs_path = "SZCH-X-Rays-741/BS"

myVQGANModel = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 512),
    num_res_channels=512,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1),),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=1024,
    embedding_dim=4,
)



class myTransformMethod():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数

        img = cv.resize(img, (image_size, image_size))  # 改变图像大小
        if img.shape[-1] == 3:  # HWC
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像


myTransform = {
    'Transform1': transforms.Compose([
        myTransformMethod(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}


class mySingleDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, img_dir, transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.img_dir = img_dir  # 读取图像路径
        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        img_path = self.img_dir  # 读取图片文件夹路径

        file = self.filelist.iloc[idx, 0]  # 读取文件名
        image = cv.imread(os.path.join(img_path, file))  # 用openCV的imread函数读取图像

        if self.transform:
            image = self.transform(image)  # 图像预处理
        return image, file  # 返回图像和名称


myTrainSet = mySingleDataset(train_file_list, cxr_path, myTransform['Transform1']) + mySingleDataset(
    train_file_list, bs_path, myTransform['Transform1'])
myTestSet = mySingleDataset(test_file_list, cxr_path, myTransform['Transform1']) + mySingleDataset(test_file_list,
                                                                                                   bs_path,
                                                                                                   myTransform[
                                                                                                       'Transform1'])

myTrainLoader = DataLoader(myTrainSet, batch_size=vae_batch_size, shuffle=True)
myTestLoader = DataLoader(myTestSet, batch_size=vae_batch_size, shuffle=False)

print("Number of batches in train set:", len(myTrainLoader))  # 输出训练集batch数量
print("Train set size:", len(myTrainSet))  # 输出训练集大小
print("Number of batches in test set:", len(myTestLoader))  # 输出测试集batch数量
print("Test set size:", len(myTestSet))  # 输出测试集大小

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model = myVQGANModel.to(device)

discriminator = PatchDiscriminator(spatial_dims=2, in_channels=1, num_layers_d=3, num_channels=64).to(device)

perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg").to(device)

optimizer_g = torch.optim.Adam(params=model.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=5e-4)

optimizer_scheduler_g = MultiStepLR(optimizer_g, milestones=[200 * len(myTrainLoader)], gamma=0.5)
optimizer_scheduler_d = MultiStepLR(optimizer_d, milestones=[200 * len(myTrainLoader)], gamma=0.5)

adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01
perceptual_weight = 0.001
# msssim_weight = 1
depth_weight = 1

epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []

total_start = time.time()
for epoch in range(vae_epoch_number):
    model.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(myTrainLoader), total=len(myTrainLoader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch[0].to(device=device, non_blocking=True)

        optimizer_g.zero_grad(set_to_none=True)

        # Generator part
        reconstruction, quantization_loss = model(images=images)
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]

        recons_loss = F.mse_loss(reconstruction.float(), images.float())
        p_loss = perceptual_loss(reconstruction.float(), images.float())
        d_loss = depth_loss(reconstruction.float(), images.float())
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        # msssim = pytorch_msssim.MSSSIM(window_size=11, size_average=True, channel=1, normalize='relu')
        # msssim_loss = 1 - msssim(reconstruction.float(), images.float())
        loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + depth_weight * d_loss

        loss_g.backward()
        optimizer_g.step()
        optimizer_scheduler_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()
        optimizer_scheduler_d.step()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(myTestLoader, start=1):
                images = batch[0].to(device=device, non_blocking=True)

                reconstruction, quantization_loss = model(images=images)

                # get the first sample from the first validation batch for visualization
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:n_example_images, 0])

                recons_loss = F.mse_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)

        torch.save(model, str(date.today()) + "-SZCH-X-Rays-VQGAN"+str(depth_weight)+".pth")

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

plt.style.use("seaborn-v0_8")
plt.title("Learning Curves", fontsize=20)
plt.plot(np.linspace(1, vae_epoch_number, vae_epoch_number), epoch_recon_loss_list, color="C0",
         linewidth=2.0,
         label="Train")
plt.plot(
    np.linspace(val_interval, vae_epoch_number, int(vae_epoch_number / val_interval)),
    val_recon_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("Learning-S"+str(depth_weight)+".png")

plt.title("Adversarial Training Curves", fontsize=20)
plt.plot(np.linspace(1, vae_epoch_number, vae_epoch_number), epoch_gen_loss_list, color="C0",
         linewidth=2.0,
         label="Generator")
plt.plot(np.linspace(1, vae_epoch_number, vae_epoch_number), epoch_disc_loss_list, color="C1",
         linewidth=2.0,
         label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig("Adversarial-S"+str(depth_weight)+".png")

fig, ax = plt.subplots(nrows=1, ncols=2)
images = (images[0, 0] * 0.5 + 0.5) * 255
ax[0].imshow(images.detach().cpu(), vmin=0, vmax=255, cmap="gray")
ax[0].axis("off")
ax[0].title.set_text("Inputted Image")
reconstructions = (reconstruction[0, 0] * 0.5 + 0.5) * 255
ax[1].imshow(reconstructions.detach().cpu(), vmin=0, vmax=255, cmap="gray")
ax[1].axis("off")
ax[1].title.set_text("Reconstruction")
plt.savefig("reconstruction images-S"+str(depth_weight)+".png")
