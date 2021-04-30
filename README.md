# DECIMER 1.0: Deep Learning for Chemical Image Recognition using Transformers

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/DECIMER-Image_Transformer.svg)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/DECIMER-Image_Transformer.svg)](https://GitHub.com/Kohulan/DECIMER-Image_Transformer/graphs/contributors/)
[![DOI](https://zenodo.org/badge/293572361.svg)](https://zenodo.org/badge/latestdoi/293572361)
[![Documentation Status](https://readthedocs.org/projects/decimer-image-transformer/badge/?version=latest)](https://decimer-image-transformer.readthedocs.io/en/latest/?badge=latest)

## Abstract

- The DECIMER (Deep lEarning for Chemical ImagE Recognition) project [1] was launched to address the OCSR problem with the latest computational intelligence methods to provide an automated open-source software solution.

- The original implementation of DECIMER[1] using GPU does take a longer training time when we use a bigger dataset of images of more than 1 million. To overcome these longer training times, many implement the training script to work on multiple GPUs. But we tried to step up and implemented our code to use Google's Machine Learning hardware [TPU(Tensor Processing Unit)](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) [2]. You can learn more about the hardware [here](https://en.wikipedia.org/wiki/Tensor_Processing_Unit).

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image_Transformer/blob/master/DECIMER_8K_Red_.png?raw=true)](https://github.com/Kohulan/Smiles-TO-iUpac-Translator)

## Method and model changes
 - The DECIMER now uses EfficientNet-B3 [3],[4] for Image feature extraction and a transformer model [5] for predicting the SMILES.
 - The SMILES [6] are encoded to [SELFIES](https://github.com/aspuru-guzik-group/selfies) [7] during training and predictions

### Changes in training method

 - We converted our datasets into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) Files, A binary file system that can be read by the TPUs in a much faster way. Also, we can use these files to train on GPUs, by using the TFRecord helps us to train the model fast by overcoming the bottleneck of reading multiple files from the hard disks.
 - We moved our data to [Google Cloud Buckets](https://cloud.google.com/storage/docs/json_api/v1/buckets). An efficient storage solution provided by google cloud environment where we can access these files from any google cloud VMs easily and in a much faster way. (To get the highest speed the cloud storage and the VM should be in the same region)
 - We adopted the TensorFlow data pipeline to load all TFRecord files to the TPUs from Google Cloud Buckets.
 - We modified the main training code to work on TPUs using [TPU strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy) introduced in Tensorflow 2.0.

### Documentation

- Currently we are working on improving the [documentation](https://decimer-image-transformer.readthedocs.io/en/latest/)

## Usage:

### Deployment of VMs to train using TPUs
- We have to deploy VMs using [ctpu](https://cloud.google.com/tpu/docs/ctpu-reference) commands to launch VMs to work with TPUs.
- To launch VMs, type the following code on the Google cloud console shell environment.check the reference for more details.
```
ctpu up --vm-only --zone=europe-west4-a --name=tpu-test --machine-type=n1-highmem-8 --disk-size-gb=100 --project PROJECT_NAME
```
- TPUs can be launched by simply selecting the TPus hardware in Compute Engine in Google Cloud console.

## How to use DECIMER?

### We suggest to use DECIMER inside a Conda environment, which makes the dependencies to install easily.
- Conda can be downloaded as part of the [Anaconda](https://www.anaconda.com/) or the [Miniconda](https://conda.io/en/latest/miniconda.html) plattforms (Python 3.7). We recommend to install miniconda3. Using Linux you can get it with:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

### Instructions

```
$ sudo apt update
$ sudo apt install default-jdk # Incase if you do not have Java already installed
```

### Python Package Installation

Install the latest code from GitHub with:

```shell
$ pip install git+https://github.com/Kohulan/DECIMER-Image_Transformer.git
```

Install in development mode with:

```shell
$ git clone https://github.com/Kohulan/DECIMER-Image_Transformer.git decimer
$ cd decimer/
$ pip install -e .
```

where `-e` means "editable" mode.

### Install tensorflow==2.3.0 if you do not have an nVidia GPU (On Mac OS)

### CLI Usage

The Python package automatically installs the `decimer` command line tool.

```shell
$ decimer --help  # Use for help
```

- When you run the program for the first time the models will get automatically downloaded(Note: total size is ~ 1GB). Also, you can manually download the models from [here](https://storage.googleapis.com/iupac_models_trained/DECIMER_transformer_models/DECIMER_trained_models_v1.0.zip)
e.g.: 
```shell
$ decimer --model Canonical --image Sample_Images/caffeine.png       # Predict SMILES for a single image.
$ decimer --model Isomeric --dir Sample_Images         # Predict SMILES for all the Images inside a folder.
```
#### DECIMER automatically selects the Canonical model, but you can choose one of the following models

Available Models:
 - Canonical : Model trained on images depicted using canonical SMILES
 - Isomeric : Model trained on images depicted using isomeric SMILES, which includes stereochemical information + ions
 - Augmented: Model trained on images depicted using isomeric SMILES with augmentations 

## License:
- This project is licensed under the MIT License - see the [LICENSE](https://raw.githubusercontent.com/Kohulan/DECIMER-Image_Transformer/master/LICENSE?token=AHKLIF3EULMCUKCFUHIPBMDARSMDO) file for details

## Citation

- Rajan, Kohulan; Zielesny, Achim; Steinbeck, Christoph (2021): DECIMER 1.0: Deep Learning for Chemical Image Recognition using Transformers. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.14479287.v1 


## References

1. Rajan, K., Zielesny, A. & Steinbeck, C. DECIMER: towards deep learning for chemical image recognition. J Cheminform 12, 65 (2020). https://doi.org/10.1186/s13321-020-00469-w
2. Norrie T, Patil N, Yoon DH, Kurian G, Li S, Laudon J, Young C, Jouppi N, Patterson D (2021) The Design Process for Google’s Training Chips: TPUv2 and TPUv3. IEEE Micro 41:56–63
3. Tan M, Le Q (2019) Efficientnet: Rethinking model scaling for convolutional neural networks. In: International Conference on Machine Learning. PMLR, pp 6105–6114
4. Xie Q, Luong M-T, Hovy E, Le QV (2020) Self-training with noisy student improves imagenet classification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp 10687–10698
5. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I (2017) Attention Is All You Need. arXiv [cs.CL]
6. Weininger D (1988) SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. J Chem Inf Comput Sci 28:31–36
7. Krenn M, Häse F, Nigam A, Friederich P, Aspuru-Guzik A (2020) Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation. Mach Learn: Sci Technol 1:045024


## Author: [Kohulan](https://kohulanr.com)

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://decimer.ai)

## Project Website: [DECIMER](https://decimer.ai)

## Research Group
[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png)](https://cheminf.uni-jena.de)
