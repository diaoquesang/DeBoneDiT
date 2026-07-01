<div align="center">

![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:D22229,100:2B4FA3&text=DeBoneDiT:%20Depth-Driven%20Conditional%20Bridge-nl-Diffusion%20Transformers%20for%20Bone%20Suppression&reversal=false&fontSize=28&fontAlignY=28&desc=Yifei%20Sun,%20Fenglei%20Fan,%20Junhao%20Jia,%20Wenming%20Deng,%20Hongxia%20Xu,%20Changmiao%20Wang,%20Ruiquan%20Ge&descSize=12&descAlignY=55&fontColor=FFFFFF)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/license/apache-2.0)

</div>

This code is a **pytorch** implementation of our paper "**DeBoneDiT: Depth-Driven Conditional Bridge Diffusion Transformers for Bone Suppression**".

<div align="center">

<div style="display:flex;gap:10px;">
  <img src="https://github.com/user-attachments/assets/1adea621-28dc-4b4f-ac69-096f2d02d7d2" style="width:85%;">
</div>

</div>

**Fig. 1.** Overview of the **DeBoneDiT** architecture. **Stage 1 (top)**: A **Depth Auto-Encoder (DAE)** is first pretrained to reconstruct images while preserving depth-guided geometric cues by leveraging a frozen Depth Anything V2 encoder. **Stage 2 (bottom)**: The bone suppression task is formulated as a **source-conditioned Brownian bridge diffusion process** in the latent space, where a **DiT-based network** iteratively transforms the latent representation of the source CXR $z_T$ into the target soft tissue $z_0$.

## 🧑🏻‍🏫 Background

**Chest X-Ray (CXR)** is a primary modality for diagnosing pulmonary diseases, yet **the overlap of bony structures often obscures critical radiographic findings**, increasing the risk of diagnostic uncertainty. While **Dual-Energy Subtraction (DES)** is the clinical benchmark for bone suppression, its widespread adoption is primarily hindered by **specialized hardware requirements and increased radiation exposure**.

## 😖 Current Challenges

- While LDMs are widely adopted to handle high-resolution CXRs efficiently, their vision tokenizers are typically optimized for image reconstruction fidelity and therefore **fail to explicitly preserve depth-related geometric cues**, which are important for distinguishing overlapping bony and soft tissue structures.

- Existing diffusion-based methods formulate bone suppression as a conditional generation task initialized from Gaussian noise. The resulting distribution gap **increases the difficulty of learning anatomically consistent mappings** and often **requires extensive iterative denoising**.

- **The scarcity of large-scale, high-quality paired datasets** remains a primary constraint in bone suppression research. The largest existing publicly available dataset JSRT contains only 241 paired images with blurring and pronounced artifacts, limiting training reliability and clinical translation.

## 🌟 Primary Contributions

To address these challenges, we propose **DeBoneDiT**, a depth-driven conditional bridge diffusion Transformer architecture designed for efficient high-resolution bone suppression. Drawing upon prior experience, we adopt the two-stage design of LDMs to ensure computational efficiency. In addition, we have constructed and released **SZCH-X-Rays**, a high-quality dataset containing 741 pairs of CXR and DES soft tissue images for bone suppression research. Our contributions can be summarized as follows:

- We introduce **Depth Auto-Encoder (DAE)**, a depth-driven vision tokenizer for latent compression. By incorporating a depth loss derived from the multi-level features of a pretrained Depth Anything V2~\citep{yang2024depth} encoder, DAE preserves both visual fidelity and depth-guided geometric cues within the latent representations.

- We formulate bone suppression as a **source-conditioned Brownian bridge diffusion process**, leveraging CXRs as both the diffusion prior and structural guidance to narrow the source-to-target domain gap and preserve anatomical consistency. 

- We demonstrate that **Diffusion Transformers (DiTs)** exhibit computational efficiency and scalability advantages over other denoising backbones in bone suppression, enabling high-resolution processing with reduced computational overhead.

- We have constructed and released **SZCH-X-Rays**, the largest publicly available high-quality paired dataset for bone suppression to date, comprising 741 pairs of posterior-anterior CXR and DES soft tissue images.
  
- **Extensive experiments** and **downstream evaluation** demonstrate that DeBoneDiT achieves superior bone suppression performance with reduced computational overhead, underscoring its potential for clinical application.

## ⚙️ Prerequisties

- Linux/Windows
- Python>=3.11
- NVIDIA GPU + CUDA cuDNN

## 🧪 Implementation Details

All experiments were performed using PyTorch 2.5.1 on an NVIDIA A800 GPU within Ubuntu 22.04. The DAE was trained from scratch for 200 epochs with a batch size of 4 using the Adam optimizer. Specifically, the learning rate was set to $10^{-4}$ for both the encoder $E$ and decoder $D$, and $5 \times 10^{-4}$ for the discriminator $\mathcal{D}\_{\text{adv}}$. Regarding the latent space, the downsampling factor $r$ and the channel dimension $C$ were set to 4, resulting in a latent representation of $z \in \mathbb{R}^{4 \times 256 \times 256}$ for input $x \in \mathbb{R}^{1 \times 1024 \times 1024}$. The codebook $\mathcal{Z}$ was configured with a size of $K = 1024$. For the depth feature extractor $\phi$, we adopted the ViT-B/14 configuration, with input data resized to $518 \times 518$ pixels prior to depth feature extraction. Consequently, the sequence length of spatial patch tokens was $N_\text{p} = 1369$, and the feature dimensions $D_l$ for the $L = 4$ selected intermediate layers were 96, 192, 384, and 768, respectively. To balance different training objectives, the trade-off weights for DAE training were set to $\lambda_{\text{adv}}=10^{-2}$, $\lambda_{\text{per}}=10^{-3}$, and $\lambda_{\text{dep}}=1$. 

In addition, the DiT-based denoising network $\epsilon_\theta$ with $N = 12$ DiT blocks was trained from scratch for 2000 epochs with a batch size of 8 using the AdamW optimizer. Specifically, a multi-step learning rate decay strategy was employed, initialized at $10^{-4}$. Following established practices, Exponential Moving Average (EMA) and Classifier-Free Guidance (CFG) were adopted to ensure model robustness during training and inference, respectively. Furthermore, the number of diffusion timesteps $T$ was set to 1000 for training, whereas only 50 sampling steps were utilized for inference to facilitate rapid generation. 

For the downstream evaluation, all five classification models were fine-tuned from ImageNet-1K pretrained weights using the Adam optimizer with a weight decay of $10^{-4}$. The initial learning rate was set to $10^{-3}$ with a plateau-based learning rate scheduler. All these models were trained for 20 epochs with a batch size of 16 using the standard cross-entropy loss with a label smoothing coefficient of 0.15.

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

This code is licensed under the terms of the [Apache License 2.0](https://opensource.org/license/apache-2.0). 
