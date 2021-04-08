# DECIMER-TPU
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/Kohulan/DECIMER-TPU/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Kohulan/DECIMER-TPU.svg)](https://GitHub.com/Kohulan/DECIMER-TPU/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/Kohulan/DECIMER-TPU.svg)](https://GitHub.com/Kohulan/DECIMER-TPU/graphs/contributors/)

- The original implementation of DECIMER using GPU does take a longer training time when we use a bigger dataset of images of more than 1 million. To overcome these longer training times, many implement the training script to work on multiple GPUs. But we tried to step up and implemented our code to use Google's Machine Learning hardware [TPU(Tensor Processing Unit)](https://en.wikipedia.org/wiki/Tensor_Processing_Unit). You can learn more about the hardware [here](https://en.wikipedia.org/wiki/Tensor_Processing_Unit).

### Main changes made

 - We converted our datasets into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) Files, A binary file system which can be read by the TPUs in a much faster way. Also, we can use these files to train on GPUs, by using the TFRecord helps us to train the model fast by overcoming the bottleneck of reading multiple files from the hard disks.
 - We moved our data to [Google Cloud Buckets](https://cloud.google.com/storage/docs/json_api/v1/buckets). An efficient storage solution provided by google cloud environment where we can access these files from any google cloud VMs easily and in a much faster way. (To get the highest speed the cloud storage and the VM should be on the same region)
 - We adopted the TensorFlow data pipeline to load all TFRecord files to the TPUs from Google Cloud Buckets.
 - We modified the main training code to work on TPUs using [TPU strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy) introduced in Tensorflow 2.0.

#### Top-level directory layout
```bash

  ├── Network/                                    # Training script and the model
  +   ├ ─ TPU_Trainer_Image2Smiles.py             # Main training script - We can modify this according to our hardware
  +   └ ─ I2S_Model.py                            # Main Model
  +    
  ├── TF_RecordUtils/                                      # Utilities used to generate the data
  +   ├ ─ Create_TF_Record.py                     # Original script for creating TFRecords
  +   ├ ─ Create_TFrecord_From_Vectors.py         # Can be used to create multiple TFRecords same time
  +   ├ ─ Create_tokenizer.py                     # Can be used to create the tokenizer and the multiple path files which can be later used in creating TfRecords and Evaluation.
  +   └ ─ I2S_Data.py                             # To be used with Create_TF_Records.py
  + 
  ├── LICENSE
  ├── Python_Requirements                         # Python requirements needed to run the scripts without error
  └── README.md
  
  ```

## Usage of required dependencies:

### Deployment of VMs to use TPUs
- We have to deploy VMs using [ctpu](https://cloud.google.com/tpu/docs/ctpu-reference) commands to launch VMs to work with TPUs.
- To launch VMs, type the following code on the Google cloud console shell environment.check the reference for more details.
```
ctpu up --vm-only --zone=europe-west4-a --name=tpu-test --machine-type=n1-highmem-8 --disk-size-gb=100 --project PROJECT_NAME
```
- TPUs can be launched by simply selecting the TPus hardware in Compute Engine in Google Cloud console.

### Requirements
  - matplotlib
  - pillow
  - selfies
  
## License:
- This project is licensed under the MIT License - see the [LICENSE](https://github.com/Kohulan/Decimer-Python/blob/master/LICENSE) file for details

## Citation
- Use this BibTeX tp cite our paper published in Chemrxiv (-todo: update the BibTeX of peer-reviewed paper)
```
@article{Rajan2020,
author = "Kohulan Rajan and Achim Zielesny and Christoph Steinbeck",
title = "{DECIMER - Towards Deep Learning for Chemical Image Recognition}",
year = "2020",
month = "6",
url = "https://chemrxiv.org/articles/DECIMER_-_Towards_Deep_Learning_for_Chemical_Image_Recognition/12464420",
doi = "10.26434/chemrxiv.12464420.v1"
}
```

## Author:
- [Kohulan](github.com/Kohulan)

[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/raw/master/assets/DECIMER.gif)](https://kohulan.github.io/Decimer-Official-Site/)

## Project Website
- [DECIMER](https://kohulan.github.io/Decimer-Official-Site/)

## Research Group
[![GitHub Logo](https://github.com/Kohulan/DECIMER-Image-to-SMILES/blob/master/assets/CheminfGit.png)](https://cheminf.uni-jena.de)
