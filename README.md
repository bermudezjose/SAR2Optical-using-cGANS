please contact with bermudez@ele.puc-rio.br for more information.

### Synthesis of Multispectral Optical Images from SAR-Optical Multitemporal Data using cGANS

TensorFlow implementation of [Synthesis of Multispectral Optical Images from SAR-Optical Multitemporal Data using cGANS] that uses cGANs to generate the missing optical image by exploiting the correspondent SAR data with an SAR-optical image pair from the same area in a different epoch.

The original code is an adaptation of https://github.com/yenchenlin/pix2pix-tensorflow pix2pix Tensorflow implementation.

For a new version, it is included two notebooks for monotemporal and multitemporal translations using Sentinel 1 and 2 data. Images can be downloaded using git. For these notebooks, the implementation is based on Keras and Tensorflow frameworks, and the Generator arquitecture was updated by incorporating Residual blocks. Notebooks can be executed on Google Colab.

## Setup

### Prerequisites
- Linux
- Python with numpy
- NVIDIA GPU + CUDA 9.2 + CuDNNv7.5
- TensorFlow 1.10

### Getting Started
- Clone this repo:
```bash
git clone git@github.com:bermudezjose/Synthesis-of-multispectral-optical-images-from-SAR-optical-multi-temporal-data-using-cGANS.git
cd pix2pix-tensorflow
