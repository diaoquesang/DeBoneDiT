![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:D22229,100:2B4FA3&text=DeBoneDiT:%20Depth-Driven%20Conditional%20Bridge-nl-Diffusion%20Transformers%20for%20Bone%20Suppression&reversal=false&fontSize=28&fontAlignY=28&desc=Yifei%20Sun,%20Fenglei%20Fan,%20Junhao%20Jia,%20Wenming%20Deng,%20Hongxia%20Xu,%20Changmiao%20Wang,%20Ruiquan%20Ge&descSize=12&descAlignY=55&fontColor=FFFFFF)

This code is a **pytorch** implementation of our paper "**DeBoneDiT: Depth-Driven Conditional Bridge Diffusion Transformers for Bone Suppression**".


## 🧑🏻‍🏫 Background

**Chest X-Ray (CXR)** is a primary modality for diagnosing pulmonary diseases, yet **the overlap of bone structures often obscures critical pathological details**, increasing the risk of diagnostic uncertainty. While **Dual-Energy Subtraction (DES)** is the clinical benchmark for bone suppression, its widespread use is limited by **specialized hardware requirements** and **increased radiation exposure**. 

<div align="center">

<img width="80%" alt="data" src="https://github.com/user-attachments/assets/9b521b12-31c6-43e5-8537-0991e921f889" />

  Figure 2: An illustration of MAs in fundus images.
</div>

## 😖 Current Challenges

1. CXR imaging inherently projects 3D anatomical structures onto a 2D plane, leading to irreversible loss of spatial depth information. This deficiency limits the ability of conventional latent space construction to capture the hierarchical anatomical relationships between bones and soft tissues, compromising the fidelity of subsequent diffusion-based soft tissue synthesis.

2. Existing diffusion-based methods frame bone suppression as a conditional generation task initialized from Gaussian noise. However, the substantial gap between the Gaussian prior and the target data distribution compromises anatomical consistency. Consequently, these methods inherently require a large number of sampling steps to bridge this gap, leading to increased computational burden.

3. While U-Net architectures remain the dominant backbone for noise estimation in diffusion models, their encoder-decoder design with dense skip connections suffers from suboptimal computational efficiency and poor scalability when performing high-resolution bone suppression. This limitation precludes practical integration into routine radiological workflows.

4. The scarcity of large-scale, high-quality paired datasets constitutes a fundamental bottleneck in bone suppression research. Currently, the largest existing publicly available dataset, JSRT, contains only 241 paired images with suboptimal quality marked by inadequate clarity and pronounced artifacts. These data constraints severely hinder the reliability of model training and evaluation, impeding the effective translation of methodological innovations into clinically applicable solutions.

## 🌟 Primary Contributions

To address these challenges, we propose a **W**avelet **D**iffusion **T**ransformer framework for **M**A **D**etection (**WDT-MD**). This is a supervised image-conditioned wavelet-domain model based on  Diffusion Transformers (DiTs). Our contributions can be summarized as follows:

1. In order to mitigate "identity mapping", we propose a ``noise-encoded image conditioning`` mechanism for diffusion-based MA detection. By perturbing the image condition with random intensities during training, the model is driven to capture the normal pattern.

2. To alleviate the issue of high false positives, we introduce pixel-level supervision signals in the training process through ``pseudo-normal pattern synthesis``. Specifically, we obtain the pseudo-normal labels align with the spatial distribution of real fundus images using inpainting techniques. This enables the model to distinguish MAs from other anomalies, thereby improving the detection performance.

3. To improve the reconstruction quality of normal features, we propose a ``wavelet diffusion Transformer`` architecture, which combines the global modelling capability of DiTs with the multi-scale analysis advantage of wavelet decomposition to better understand the overall structure and detailed information of fundus images.

4. Comprehensive experiments on the IDRiD and e-ophtha MA datasets demonstrate exceptional performance of our WDT-MD, holding significant promise for improving early DR screening.

## ⚙️ Prerequisties

- Linux/Windows
- Python>=3.7
- NVIDIA GPU + CUDA cuDNN

## 🧪 Implementation Details

All experiments were performed using PyTorch 2.5.1 on a single NVIDIA V100 32 GB GPU within Ubuntu 22.04. WDT-MD was trained from scratch over 600 epochs with a batch size of 4 utilizing the AdamW optimizer, complemented by a dynamic learning rate schedule initialized at $10^{-4}$. The noise scheduling parameter $\beta_t$ followed a scaled linear trajectory ranging from 0.00085 to 0.012 across $T=1000$ diffusion timesteps. The sampling steps $T_s$ was set to 50 using the LCM sampler. In pseudo-normal pattern synthesis, the inpainting radius $r$ is set to 3 pixels. For wavelet decomposition, the Daubechies 6 basis was selected to balance computational efficiency and time-frequency localization.

## 📦 Datasets

### Downloading

Two publicly available datasets, namely [IDRiD](https://ieee-dataport.org/openaccess/indian-diabetic-retinopathy-image-dataset-idrid) and [e-ophtha MA](https://www.adcis.net/en/third-party/e-ophtha), are adopted for extensive evaluation.

**The IDRiD dataset**, a benchmark resource for diabetic retinopathy analysis, was adapted for our study. For MA detection, we curated a subset of 249 samples, including 199 training cases, 24 validation cases, and 26 test cases. Specifically, the training set contains 134 normal images and 65 abnormal images. Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied with 8 $\times$ 8 tile grids and a 2.0 clip limit to enhance contrast. Considering  the computational overhead, we implemented dimension standardization through bilinear downsampling to 300 $\times$ 200 pixels. 

**The e-ophtha MA dataset** consists of 381 cases divided into 304 training, 38 validation, and 39 test samples. Specifically, the training set contains 188 normal images and 116 abnormal images. The preprocessing pipeline maintained strict consistency with IDRiD: (1) CLAHE (8 $\times$ 8 tile grids, 2.0 clip limit); (2) downsampling to 300 $\times$ 200 pixels.


### Pre-processing

```
python code/pre-processing.py
```

### Split

```
python code/split.py
```

## 🌵 Dependencies

```
pip install -r requirements.txt
```

<div align="center">

| Dependencies | Versions | Dependencies | Versions |
|--------------|----------|--------------|----------|
| diffusers    | 0.27.2   | timm         | 1.0.15   |
| matplotlib   | 3.7.2    | torch        | 2.0.1+cu117 |
| matplotlib-inline | 0.1.6 | torch-ema    | 0.3      |
| numpy        | 1.26.4   | torchaudio   | 2.0.2+cu117 |
| opencv-python | 4.8.1.78 | torchprofile | 0.0.4    |
| pandas       | 2.0.3    | torchsummary | 1.5.1    |
| pytorch-wavelets | 1.3.0 | torchvision  | 0.15.2+cu117 |
| PyWavelets   | 1.8.0    | tqdm         | 4.66.1   |

</div>

## 🍳 Training

```
python code/wdt_train.py
```

## 🚅 Inference

```
python code/wdt_eval.py
```
