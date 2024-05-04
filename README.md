# ***DECIMER Image Transformer***: Deep Learning for Chemical Image Recognition using Efficient-Net V2 + Transformer

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/DECIMER-Image_Transformer.svg)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/DECIMER-Image_Transformer.svg)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/contributors/)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org)
[![DOI](https://zenodo.org/badge/293572361.svg)](https://zenodo.org/badge/latestdoi/293572361)
[![Documentation Status](https://readthedocs.org/projects/decimer-image-transformer/badge/?version=latest)](https://decimer-image-transformer.readthedocs.io/en/latest/?badge=latest)
[![GitHub release](https://img.shields.io/github/release/Kohulan/DECIMER-Image_Transformer.svg)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/releases/)
[![PyPI version fury.io](https://badge.fury.io/py/decimer.svg)](https://pypi.python.org/pypi/decimer/)

## Abstract

The DECIMER 2.2 [5] (Deep lEarning for Chemical ImagE Recognition) project [1] was launched to address the OCSR problem with the latest computational intelligence methods to provide an automated open-source software solution.

The original implementation of DECIMER[1] using GPU takes a longer training time when we use a bigger dataset of more than 1 million images. To overcome these longer training times, many implement the training script to work on multiple GPUs. However, we tried to step up and implemented our code to use Google's Machine Learning hardware [TPU(Tensor Processing Unit)](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) [2]. You can learn more about the hardware [here](https://en.wikipedia.org/wiki/Tensor_Processing_Unit).

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image_Transformer/blob/master/DECIMER_V2.png?raw=true)](https://github.com/Kohulan/DECIMER-Image_Transformer)

## Method and model changes
 - The DECIMER now uses EfficientNet-V2[3] for Image feature extraction and a transformer model [4] for predicting the SMILES.
 - The SMILES used during training and predictions

### Changes in the training method

 - We converted our datasets into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) Files, A binary file system the TPUs can read in a much faster way. Also, we can use these files to train on GPUs. Using the TFRecord helps us train the model fast by overcoming the bottleneck of reading multiple files from the hard disks.
 - We moved our data to [Google Cloud Buckets](https://cloud.google.com/storage/docs/json_api/v1/buckets). An efficient storage solution provided by the google cloud environment where we can access these files from any google cloud VMs easily and in a much faster way. (To get the highest speed, the cloud storage and the VM should be in the same region)
 - We adopted the TensorFlow data pipeline to load all TFRecord files to the TPUs from Google Cloud Buckets.
 - We modified the main training code to work on TPUs using [TPU strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy) introduced in Tensorflow 2.0.

## How to use DECIMER?
- Python package [Documentation](https://decimer-image-transformer.readthedocs.io/en/latest/?badge=latest)
- Model library could be found here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7624994.svg)](https://zenodo.org/record/7624994)

### We suggest using DECIMER inside a Conda environment, which makes the dependencies install easily.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) platforms (Python 3.7). We recommend installing miniconda3. Using Linux, you can get it with:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

### Instructions

### Python Package Installation

#### Use a conda environment for clean installation
```shell
$ sudo apt update
$ sudo apt install unzip
$ conda create --name DECIMER python=3.10.0
$ conda activate DECIMER
$ conda install pip
$ python3 -m pip install -U pip
```

Install the latest code from GitHub with:
```shell
$ pip install git+https://github.com/Kohulan/DECIMER-Image_Transformer.git
```

Install in development mode with:
```shell
$ git clone https://github.com/Kohulan/DECIMER-Image_Transformer.git decimer
$ cd decimer/
$ pip install -e.
```
- Where `-e` means "editable" mode.

Install from PyPi
```shell
$ pip install decimer
```
### How to use inside your own python script
```python
from DECIMER import predict_SMILES

# Chemical depiction to SMILES translation
image_path = "path/to/imagefile"
SMILES = predict_SMILES(image_path)
print(SMILES)
```

### Install tensorflow == 2.10.1 if you do not have an Nvidia GPU (On Mac OS)

## License:
- This project is licensed under the MIT License - see the [LICENSE](https://raw.githubusercontent.com/Kohulan/DECIMER-Image_Transformer/master/LICENSE?token=AHKLIF3EULMCUKCFUHIPBMDARSMDO) file for details

## Citation
- Rajan K, Brinkhaus HO, Agea MI, Zielesny A, Steinbeck C DECIMER.ai - An open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications. Nat. Commun. 14, 5045 (2023). https://doi.org/10.1038/s41467-023-40782-0
- Rajan, K., Zielesny, A. & Steinbeck, C. DECIMER 1.0: deep learning for chemical image recognition using transformers. J Cheminform 13, 61 (2021). https://doi.org/10.1186/s13321-021-00538-8

## References

1. Rajan, K., Zielesny, A. & Steinbeck, C. DECIMER: towards deep learning for chemical image recognition. J Cheminform 12, 65 (2020). https://doi.org/10.1186/s13321-020-00469-w
2. Norrie T, Patil N, Yoon DH, Kurian G, Li S, Laudon J, Young C, Jouppi N, Patterson D (2021) The Design Process for Google's Training Chips: TPUv2 and TPUv3. IEEE Micro 41:56–63
3. Tan M, Le QV (2021) EfficientNetV2: Smaller Models and Faster Training. arXiv [cs.CV]
4. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I (2017) Attention Is All You Need. arXiv [cs.CL]
5. Rajan, K., Zielesny, A. & Steinbeck, C. DECIMER 1.0: deep learning for chemical image recognition using transformers. J Cheminform 13, 61 (2021). https://doi.org/10.1186/s13321-021-00538-8

## Acknowledgement
- We thank [Charles Tapley Hoyt](https://github.com/cthoyt) for his valuable advice and help in improving the DECIMER repository.
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

<p align="center">
  <img src="https://user-images.githubusercontent.com/30716951/220350828-913e6645-6a0a-403c-bcb8-160d061d4606.png" width="500" class="center">
</p>

## Author: [Kohulan](https://kohulanr.com)

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://decimer.ai)

## Project Website:

- A web application implementation is available at [decimer.ai](https://decimer.ai), implemented by [Otto Brinkhaus](https://github.com/OBrink)


## Research Group
[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png)](https://cheminf.uni-jena.de)

![Alt](https://repobeats.axiom.co/api/embed/bf532b7ac0d34137bdea8fbb82986828f86de065.svg "Repobeats analytics image")
