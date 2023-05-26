# Data-Centric-Diet


This repository accompanies the paper [Data-Centric Diet: Effective Dataset Pruning for Medical Image Segmentation](TODO) and contains the basic code for replicating the training and computations in it.


## Usage
The main requirements are pytorch 1.4.0 with python 3.9.1.

### Training on single dataset
To train one independent run of 3D-UNet on MSD-Pancreas (the full dataset), first set up params `args.dataset='msd'` in [`params/params_msd_only`](params_msd_only), then from `<ROOT>` execute

```sh
python run/train_vnet.py 
```

### ATTENTION
This is a demo presentation that may encounter bugs or issues that may not run properly. The complete code will be released after the paper is received.

