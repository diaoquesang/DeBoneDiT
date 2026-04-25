from config import config

from transform import myTransform
from torch.utils.data import DataLoader
from BBDMScheduler import BBDMScheduler
from tqdm import tqdm
import cv2 as cv
import time
import os
from dataset import mySingleDataset
from monai.utils import set_determinism

import numpy as np
import torch

set_determinism(42)


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境
    output_path = "dit_output_N8"
    w = 1.5  # 1.5 for SZCH-X-Rays; 1.0 for JSRT
    cxr_path = os.path.join("SZCH-X-Rays-741", "CXR")

    model = torch.load("YOUR DEBONEDIT MODEL PATH").to(device).eval().requires_grad_(False)
    VQGAN = torch.load("YOUR DAE MODEL PATH").to(device).eval().requires_grad_(False)
    testset_list = "SZCH-X-Rays_testset.txt"
    myTestSet = mySingleDataset(testset_list, cxr_path, myTransform['testTransform'])
    myTestLoader = DataLoader(myTestSet, batch_size=1, shuffle=False)
    noise_scheduler = BBDMScheduler(num_train_timesteps=config.num_train_timesteps)

    with torch.no_grad():
        progress_bar = tqdm(enumerate(myTestLoader), total=len(myTestLoader), ncols=100)
        total_start = time.time()
        for step, batch in progress_bar:
            noise_scheduler.set_timesteps(config.num_infer_timesteps, device="cuda",
                                          original_inference_steps=config.num_train_timesteps)
            cxr = batch[0].to(device=device, non_blocking=True).float()
            filename = batch[1][0]
            cxr = VQGAN.encode_stage_2_inputs(cxr)
            sample = cxr.clone()

            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
                sample_uc = torch.cat((sample, torch.randn_like(cxr)), dim=1)
                residual_uc = model(sample_uc, torch.Tensor((t,)).to(device).long())[:, :4].to(device)

                sample = torch.cat((sample, cxr), dim=1)
                residual = model(sample, torch.Tensor((t,)).to(device).long())[:, :4].to(device)

                residual = w * residual + (1 - w) * residual_uc

                samples = noise_scheduler.step(residual, t, sample[:, :4], cxr)
                sample = samples.prev_sample

                if not config.use_server:
                    bs_show = np.transpose(np.squeeze(sample.cpu().detach().numpy())[0:3], (1, 2, 0))

                    bs_show = bs_show * 0.5 + 0.5
                    bs_show = np.clip(bs_show, 0, 1)
                    cv.imshow("win1", bs_show)
                    cv.waitKey(1)

            bs = VQGAN.decode(sample)
            bs = np.array(bs.detach().to("cpu"))
            bs = np.squeeze(bs)  # HW
            bs = bs * 0.5 + 0.5
            bs = np.clip(bs, 0, 1)
            bs *= 255
            bs = bs.astype(np.uint8)
            cv.imwrite(os.path.join(output_path, filename), bs)
        total_time = time.time() - total_start
        print(f"Total time: {total_time}.")


if __name__ == "__main__":
    eval()
