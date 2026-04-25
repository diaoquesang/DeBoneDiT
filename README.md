![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:D22229,100:2B4FA3&text=DeBoneDiT:%20Depth-Driven%20Conditional%20Bridge-nl-Diffusion%20Transformers%20for%20Bone%20Suppression&reversal=false&fontSize=28&fontAlignY=28&desc=Yifei%20Sun,%20Fenglei%20Fan,%20Junhao%20Jia,%20Wenming%20Deng,%20Hongxia%20Xu,%20Changmiao%20Wang,%20Ruiquan%20Ge&descSize=12&descAlignY=55&fontColor=FFFFFF)

This code is a **pytorch** implementation of our paper "**DeBoneDiT: Depth-Driven Conditional Bridge Diffusion Transformers for Bone Suppression**".


## 🧑🏻‍🏫 Background

**Chest X-Ray (CXR)** is a primary modality for diagnosing pulmonary diseases, yet **the overlap of bone structures often obscures critical pathological details**, increasing the risk of diagnostic uncertainty. While **Dual-Energy Subtraction (DES)** is the clinical benchmark for bone suppression, its widespread use is limited by **specialized hardware requirements and increased radiation exposure**. 

<div align="center">

<img width="80%" alt="data" src="https://github.com/user-attachments/assets/9b521b12-31c6-43e5-8537-0991e921f889" />

  Figure 2: An illustration of MAs in fundus images.
</div>

## 😖 Current Challenges

- CXR imaging inherently projects 3D anatomical structures onto a 2D plane, leading to **irreversible loss of spatial depth information**. This deficiency limits the ability of conventional latent space construction to capture the hierarchical anatomical relationships between bones and soft tissues, compromising the fidelity of subsequent diffusion-based soft tissue synthesis.

- Existing diffusion-based methods frame bone suppression as a conditional generation task initialized from Gaussian noise. However, the **substantial gap between the Gaussian prior and the target data distribution** compromises anatomical consistency. Consequently, these methods inherently require a large number of sampling steps to bridge this gap, leading to increased computational burden.

- While U-Net architectures remain the dominant backbone for noise estimation in diffusion models, their encoder-decoder design with dense skip connections suffers from **suboptimal computational efficiency and poor scalability** when performing high-resolution bone suppression. This limitation precludes practical integration into routine radiological workflows.

- **The scarcity of large-scale, high-quality paired datasets** constitutes a fundamental bottleneck in bone suppression research. Currently, the largest existing publicly available dataset, JSRT, contains only 241 paired images with suboptimal quality marked by inadequate clarity and pronounced artifacts. These data constraints severely hinder the reliability of model training and evaluation, impeding the effective translation of methodological innovations into clinically applicable solutions.

## 🌟 Primary Contributions

To address these challenges, we propose **DeBoneDiT**, a depth-driven conditional bridge diffusion Transformer architecture designed for efficient high-resolution bone suppression. Drawing upon prior experience, we adopt the two-stage design of LDMs to ensure computational efficiency. In addition, we have constructed and released **SZCH-X-Rays**, a high-quality dataset containing 741 pairs of CXR and DES soft tissue images for bone suppression research. Our contributions can be summarized as follows:

- We introduce **Depth Auto-Encoder (DAE)**, a depth-driven vision tokenizer for perceptual compression into the latent space. By incorporating a depth loss derived from the features of a pretrained DINOv2 encoder fine-tuned on a comprehensive collection of depth estimation datasets, DAE effectively  preserves spatial depth information within the latent representations. This enables enhanced perceptual quality in both latent-space bone suppression and pixel-space reconstruction.

- We formulate bone suppression as a **Brownian bridge diffusion process**, departing from the Conditional Diffusion Model (CDM) paradigm prevalent in existing research. By replacing the prior with CXR data instead of Gaussian noise, inter-domain variations in the diffusion process are significantly reduced. Building upon this, we incorporate conditional guidance based on source domain information, further mitigating prediction difficulty and cumulative error at each sampling step.

- We demonstrate that **Diffusion Transformers (DiTs)** exhibit computational efficiency and scalability advantages over other denoising backbones in bone suppression, enabling high-resolution processing with reduced computational overhead.

- We have constructed and released **SZCH-X-Rays**, the largest publicly available high-quality paired dataset for bone suppression to date, comprising 741 pairs of posterior-anterior CXR and DES soft tissue images. With superior image clarity compared to the existing JSRT dataset, SZCH-X-Rays aims to address the critical data scarcity bottleneck, providing a reliable benchmark for the training and evaluation of future models.
  
- Extensive experiments conducted on both our self-constructed SZCH-X-Rays dataset and the public JSRT dataset demonstrate that DeBoneDiT achieves **state-of-the-art performance and efficiency** in bone suppression, highlighting its promise for clinical deployment.

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

| Dependencies | Versions   | Dependencies     | Versions    | Dependencies   | Versions     |
|--------------|------------|------------------|-------------|----------------|--------------|
| diffusers    | 0.27.0     | monai            | 1.2.0       | opencv-python  | 4.12.0.88    |
| lpips        | 0.1.4      | monai-generative | 0.2.2       | openpyxl       | 3.2.0b1      |
| matplotlib   | 3.7.2      | numpy            | 1.26.4      | pandas         | 2.0.3        |
| scikit-image | 0.22.0     | timm             | 1.0.15      | torch          | 2.2.1+cu118  |
| torch-ema    | 0.3        | torchvision      | 0.17.1+cu118| tqdm           | 4.67.1       |

</div>

## 🍳 Training

```
python code/wdt_train.py
```

## 🚅 Inference

```
python code/wdt_eval.py
```
