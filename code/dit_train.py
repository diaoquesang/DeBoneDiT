from matplotlib import pyplot as plt
from config import config
from dataset import myDataset, myDiTDataset
from transform import myTransform, myDiTTransform
from torch.utils.data import DataLoader
from model import mySiTModel
from BBDMScheduler import BBDMScheduler
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from datetime import date
import torch.nn.functional as F

import torch
import time
from monai.utils import set_determinism
from torch_ema import ExponentialMovingAverage

set_determinism(42)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

    train_file_list = "SZCH-X-Rays_trainset.txt"
    test_file_list = "SZCH-X-Rays_valset.txt"

    cxr_path = "SZCH-X-Rays-741/CXR"
    bs_path = "SZCH-X-Rays-741/BS"

    myTrainSet = myDiTDataset(train_file_list, cxr_path, bs_path,
                              myDiTTransform['trainTransform'])
    myTestSet = myDataset(test_file_list, cxr_path, bs_path,
                          myTransform['testTransform'])

    myTrainLoader = DataLoader(myTrainSet, batch_size=config.batch_size, shuffle=True)
    myTestLoader = DataLoader(myTestSet, batch_size=config.batch_size, shuffle=False)

    print("Number of batches in train set:", len(myTrainLoader))
    print("Train set size:", len(myTrainSet))
    print("Number of batches in test set:", len(myTestLoader))
    print("Test set size:", len(myTestSet))

    model = mySiTModel.to(device).train()
    noise_scheduler = BBDMScheduler(num_train_timesteps=config.num_train_timesteps)
    noise_scheduler.set_timesteps(config.num_infer_timesteps, device="cuda",
                                  original_inference_steps=config.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.initial_learning_rate, eps=1e-6)
    milestones = [x * len(myTrainLoader) for x in config.milestones]
    optimizer_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if config.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    train_losses = []
    test_losses = []
    plt_train_loss_epoch = []
    plt_test_loss_epoch = []
    train_epoch_list = list(range(0, config.epoch_number))
    test_epoch_list = list(range(0, int(config.epoch_number / config.test_epoch_interval)))

    VQGAN = torch.load("YOUR DAE MODEL PATH").to(device).eval().requires_grad_(False)
    print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Training----------")
    for epoch in range(config.epoch_number):
        model.train()
        print(time.strftime("%H:%M:%S", time.localtime()),
              f"Epoch:{epoch},learning rate:{optimizer.param_groups[0]['lr']}")
        for i, batch in tqdm(enumerate(myTrainLoader)):
            cxr_i, bs_i = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                cxr = VQGAN.encode_stage_2_inputs(cxr_i)
                bs = VQGAN.encode_stage_2_inputs(bs_i)

            noise = torch.randn_like(cxr).to(device)

            timesteps = torch.randint(0, config.num_train_timesteps, (cxr.shape[0],), device=device).long()

            noisy_images = noise_scheduler.add_noise(bs, cxr, noise, timesteps)

            noisy_images = torch.cat(
                (noisy_images, cxr.clone() * torch.bernoulli(torch.full((cxr.shape[0], 1, 1, 1), 0.85)).to(device)),
                dim=1)

            pred = model(noisy_images, timesteps)[:, :4]

            if config.prediction_type == "noise":
                loss = F.mse_loss(pred.float(), noise.float())
            else:
                loss = F.mse_loss(pred.float(), ((timesteps / config.num_train_timesteps).view(-1, 1, 1, 1) * (
                        cxr - bs) + noise_scheduler.sqrd_sigma(timesteps).view(-1, 1, 1, 1) * noise).float())

            loss.backward()
            train_losses.append(loss.item())

            # 迭代模型参数
            optimizer.step()
            optimizer.zero_grad()
            optimizer_scheduler.step()
            ema.update()

        train_loss_epoch = sum(train_losses[-len(myTrainLoader):]) / len(myTrainLoader)
        print(time.strftime("%H:%M:%S", time.localtime()), f"Epoch:{epoch},train losses:{train_loss_epoch}")
        plt_train_loss_epoch.append(train_loss_epoch)

        if (epoch + 1) % config.test_epoch_interval == 0:
            model.eval()
            print(time.strftime("%H:%M:%S", time.localtime()), "----------Stop Training----------")
            print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Testing----------")
            with torch.no_grad():
                if config.ema:
                    ema.store()
                    ema.copy_to()
                for i, batch in tqdm(enumerate(myTestLoader)):
                    cxr_i, bs_i = batch[0].to(device), batch[1].to(device)

                    with torch.no_grad():
                        cxr = VQGAN.encode_stage_2_inputs(cxr_i)
                        bs = VQGAN.encode_stage_2_inputs(bs_i)

                    noise = torch.randn_like(cxr).to(device)

                    timesteps = torch.randint(0, config.num_train_timesteps, (cxr.shape[0],),
                                              device=device).long()

                    noisy_images = noise_scheduler.add_noise(bs, cxr, noise, timesteps)

                    # noisy_images = torch.cat((noisy_images, cxr.clone()), dim=1)
                    noisy_images = torch.cat((noisy_images, cxr.clone() * torch.bernoulli(
                        torch.full((cxr.shape[0], 1, 1, 1), 0.85)).to(device)), dim=1)

                    pred = model(noisy_images, timesteps)[:, :4]

                    if config.prediction_type == "noise":
                        loss = F.mse_loss(pred.float(), noise.float())
                    else:
                        loss = F.mse_loss(pred.float(), (
                                (timesteps / config.num_train_timesteps).view(-1, 1, 1, 1) * (
                                cxr - bs) + noise_scheduler.sqrd_sigma(timesteps).view(-1, 1, 1,
                                                                                       1) * noise).float())
                    test_losses.append(loss.item())

                if config.ema:
                    ema.restore()

                test_loss_epoch = sum(test_losses[-len(myTestLoader):]) / len(myTestLoader)
                print(time.strftime("%H:%M:%S", time.localtime()), f"Epoch:{epoch},test losses:{test_loss_epoch}")
                plt_test_loss_epoch.append(test_loss_epoch)
                print(time.strftime("%H:%M:%S", time.localtime()), "----------End Validation----------")
                print(time.strftime("%H:%M:%S", time.localtime()), "----------Continue to Train----------")
    print(time.strftime("%H:%M:%S", time.localtime()), "----------End Training Normally----------")
    # 查看损失曲线
    f, ([ax1, ax2]) = plt.subplots(1, 2)
    ax1.plot(train_epoch_list, plt_train_loss_epoch, color="red")
    ax1.set_title('Train loss')
    ax2.plot(test_epoch_list, plt_test_loss_epoch, color="blue")
    ax2.set_title('Test loss')  # 添加标题
    plt.savefig("./loss-S-ema-noise-2000-dl-8c-s1-dep1.png")
    if not config.use_server:
        plt.show()
    if config.ema:
        ema.copy_to()
    torch.save(model,
               "dit-" + str(config.epoch_number) + "-" + str(date.today()) + "-S-ema-noise-2000-dl-8c-s1-dep1.pth")


if __name__ == "__main__":
    train()
