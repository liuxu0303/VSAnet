![avatar](framework.jpg)

## Getting Started

#### Dependencies

- Python == 3.6.5
- torch == 1.8.1
- torchvision == 0.2.0
- cuda == 10.2.89
- cudnn == 7.6.5
- pytorch3d == 0.6.1
- imageio
- numpy
- pillow 
- path

### (1) Setup
This code has been tested with Python 3.6.5, Torch 1.8.1, CUDA 10.2 and cuDNN 7.6.5 on Ubuntu 16.04.

- Setup python environment
```
conda create -n VSAnet python=3.6.5
source activate VSAnet
pip install -r requirements.txt
conda install -c pytorch pytorch=1.8.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
#### (2) Preparing the dataset

VSA Estimation Dataset

- Dataset Directory

  ```
  |- dataset
  |   |---VSA Estimation Dataset
  |   |   |---Data
  |   |   |   |---V_chair.npy
  |   |   |   |---V_sofa.npy
  |   |   |   |---...
  |   |   |---train
  |   |   |   |---0000000.png
  |   |   |   |---0000000.txt
  |   |   |   |---...
  |   |   |---test
  |   |   |   |---0000000.png
  |   |   |   |---0000000.txt
  |   |   |   |---...
  ```
  
#### Start training

```bash
cd VSAestimator-Code
python3 -m torch.distributed.launch train_VSAestimator.py --n_bins 256 --num_layers 3
```
