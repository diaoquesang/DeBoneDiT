<div align="center">

![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:D22229,100:2B4FA3&text=DeBoneDiT:%20Depth-Driven%20Conditional%20Bridge-nl-Diffusion%20Transformers%20for%20Bone%20Suppression&reversal=false&fontSize=28&fontAlignY=28&desc=Yifei%20Sun,%20Fenglei%20Fan,%20Junhao%20Jia,%20Wenming%20Deng,%20Hongxia%20Xu,%20Changmiao%20Wang,%20Ruiquan%20Ge&descSize=12&descAlignY=55&fontColor=FFFFFF)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
</div>

This code is a **pytorch** implementation of our paper "**DeBoneDiT: Depth-Driven Conditional Bridge Diffusion Transformers for Bone Suppression**".

<div align="center">

<div style="display:flex;gap:10px;">
  <img src="https://github.com/user-attachments/assets/1adea621-28dc-4b4f-ac69-096f2d02d7d2" style="width:85%;">
</div>

</div>

**Fig. 1.** Overview of the **DeBoneDiT** architecture. **Stage 1 (top)**: A **Depth Auto-Encoder (DAE)** is first pretrained to reconstruct images while preserving spatial depth information. **Stage 2 (bottom)**: The bone suppression task is formulated as a **Brownian bridge diffusion process** in the latent space, where a **DiT-based network** iteratively transforms the latent representation of the source CXR $z_T$ into the target soft tissue $z_0$.


## 🧑🏻‍🏫 Background

**Chest X-Ray (CXR)** is a primary modality for diagnosing pulmonary diseases, yet **the overlap of bone structures often obscures critical pathological details**, increasing the risk of diagnostic uncertainty. While **Dual-Energy Subtraction (DES)** is the clinical benchmark for bone suppression, its widespread use is limited by **specialized hardware requirements and increased radiation exposure**. 

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
- Python>=3.11
- NVIDIA GPU + CUDA cuDNN

## 🧪 Implementation Details

All experiments were performed using PyTorch 2.5.1 on an NVIDIA A800 GPU within Ubuntu 22.04. The DAE was trained from scratch for 200 epochs with a batch size of 4 using the Adam optimizer. Specifically, the learning rate was set to $10^{-4}$ for both the encoder $E$ and decoder $D$, and $5 \times 10^{-4}$ for the discriminator $\mathcal{D}$. Regarding the latent space, the downsampling factor $r$ was set to 4, resulting in a latent feature map of $256 \times 256$ for $1024 \times 1024$ inputs. To balance different training objectives, the trade-off weights for DAE training are set as follows: $\lambda_{adv}=10^{-2}$, $\lambda_{per}=10^{-3}$, and $\lambda_{dep}=1$. 

In addition, the DiT-based denoising network $\epsilon_\theta$ was trained from scratch for 2000 epochs with a batch size of 8 using the AdamW optimizer. Specifically, a multi-step learning rate decay strategy was employed, initialized at $10^{-4}$. Following established practices, Exponential Moving Average (EMA) and Classifier-Free Guidance (CFG) were adopted to ensure model robustness during training and inference, respectively. Furthermore, the number of diffusion timesteps $T$ was set to 1000 for training, whereas only 50 sampling steps were utilized for inference to facilitate rapid generation. 

## 📦 Datasets

We conducted comprehensive experiments across three distinct datasets, with each cohort assigned a specific role in evaluation: our self-constructed SZCH-X-Rays dataset and the publicly available JSRT dataset were utilized for performance evaluation, while the Asraf dataset was employed for downstream evaluation to benchmark clinical applicability. All the images were resized to 1024 $\times$ 1024 pixels for experimental consistency. 

- [**SZCH-X-Rays**](https://huggingface.co/datasets/diaoquesang/SZCH-X-Rays) comprises 741 pairs of posterior-anterior CXR and DES soft tissue images, acquired using a GE Discovery XR656 system in collaboration with our partner hospital. Initially stored in 14-bit DICOM format, the images were converted to PNG format to streamline the processing workflow. Data with operational errors, pronounced motion artifacts, pleural effusion or pneumothorax, were excluded to preclude disruption to analysis. Finally, the dataset was partitioned into 592 training, 74 validation and 75 test pairs.

- [**JSRT**](https://drive.google.com/file/d/1o-T5l2RKdT5J75eBsqajqAuHPfZnzPhj/view?usp=sharing) contains 241 pairs of CXR and synthetic soft tissue images, split into 192 training, 24 validation and 25 test pairs. The CXR images were sourced from the Japanese Society of Radiological Technology while the corresponding soft tissue images were algorithmically synthesized by researchers at the Budapest University of Technology and Economics.

- [**Asraf**](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset) encompasses 6939 posterior-anterior CXR images sourced from publicly available resources, evenly distributed across three diagnostic classes with 2313 images each: pneumonia, COVID-19, and normal. To evaluate clinical applicability, we performed downstream classification on this dataset using a 5-fold cross-validation strategy to ensure robustness.

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
python dae_train.py
```

```
python dit_train.py
```

## 🗃️ Checkpoints

Checkpoints are available on [Hugging Face](https://huggingface.co/diaoquesang/DeBoneDiT/tree/main/checkpoints).

## 🚅 Inference

```
python dit_eval.py
```

## 📄 License

This project is licensed under the terms of the [**MIT License**](https://opensource.org/licenses/MIT). 
