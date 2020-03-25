# RFS

Representations for Few-Shot Learning (RFS). This repo covers the implementation of the following paper:  

**"Rethinking few-shot image classification: a good embedding is all you need?"**

If you find this repo useful for your research, please consider citing the paper  
```
@article{tian2020rethink,
  title={Rethinking few-shot image classification: a good embedding is all you need?},
  author={Tian, Yonglong and Wang, Yue and Krishnan, Dilip and Tenenbaum, Joshua B and Isola, Phillip},
  journal={arXiv preprint},
  year={2020}
}
```

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. However, it should be compatible with recent PyTorch versions >=0.4.0

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), but we have
renamed the file. Our version of data can be downloaded from here:

[[DropBox]](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0)

## Running

Exemplar commands for running the code can be found in `scripts/run.sh`.

For unuspervised learning methods `CMC` and `MoCo`, please refer to the [CMC](http://github.com/HobbitLong/CMC) repo.

## Contacts
For any questions, please contact:

Yonglong Tian (yonglong@mit.edu)  
Yue Wang (yuewang@csail.mit.edu)

## Acknowlegements
Part of the code for distillation is from [RepDistiller](http://github.com/HobbitLong/RepDistiller) repo.


