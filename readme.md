###  Multimodal Multi-Head Convolutional Attention with Various Kernel Sizes for Medical Image Super-Resolution (WACV 2023) - Official Repo

Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Andreea-Iuliana Miron, Olivian Savencu, Nicolae-Catalin Ristea, Nicolae Verga and Fahad Shahbaz Khan.

##### 🆕  This is the official repository of the "Multimodal Multi-Head Convolutional Attention with Various Kernel Sizes for Medical Image Super-Resolution" paper accepted at WACV 2023.


### 📜 Arxiv Link: https://arxiv.org/abs/2204.04218

### 🌟 Overview

We propose a novel multimodal multi-head convolutional attention module for super-resolution. **MHCA** is a spatial-channel attention module that **can be integrated into any neural network at any layer**.
We are also the first to perform medical image superresolution using a multimodal low-resolution input.

 
<img src="https://raw.githubusercontent.com/lilygeorgescu/MHCA/main/imgs/overview.png" width="800" >
 

### 🔒 License
The present code is released under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

### 💻 Code
We release the MHCA building block.

### 🚀 Results and trained models.
🌟 We obtained new SOTA results on **T2w modality on the IXI data set** for the scaling factor of 2x and 4x.

🤩 We release the pretrained models.
Check [EDSR folder](edsr) and try out our models. 

<table>
<tr>
    <td>Method</td> 
    <td>Scale</td>
    <td>PSNR/SSIM</td>  
</tr>
  
<tr>
    <td>EDSR + MCHA </td> 
    <td>2x</td>
    <td> <a href="edsr/pretrained_models/model_single_input_IXI_x2.pt">40.11/0.9871</a> </td>
</tr>

<tr>
    <td>EDSR + MMCHA </td> 
    <td>2x</td>
    <td> <a href="edsr/pretrained_models/model_multi_input_IXI_x2.pt">40.28/0.9874</a> </td>
</tr>


<tr>
    <td>EDSR + MCHA </td> 
    <td>4x</td>
    <td> <a href="edsr/pretrained_models/model_single_input_IXI_x4.pt">32.15/0.9418</a> </td>
</tr>

<tr>
    <td>EDSR + MMCHA </td> 
    <td>4x</td>
    <td> <a href="edsr/pretrained_models/model_multi_input_IXI_x4.pt">32.51/0.9452</a> </td>
</tr>


</table>

### 🔨 Installation
Please follow the instructions in [Install.md](install.md).

### 🖊Citation
Please cite our work if you use any material released in this repository.
```
@inproceedings{Georgescu-WACV-2023,
  title="{Multimodal Multi-Head Convolutional Attention with Various Kernel Sizes for Medical Image Super-Resolution}",
  author={Georgescu, Mariana-Iuliana and Ionescu, Radu Tudor and Miron, Andreea-Iuliana and Savencu, Olivian and Ristea, Nicolae-Catalin and Verga, Nicolae and Khan, Fahad Shahbaz},
  booktitle={Proceedings of WACV},
  year={2023},
  publisher={IEEE}
}
```
