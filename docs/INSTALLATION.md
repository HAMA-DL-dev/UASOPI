# Overview
It may be comfortable to install via `environment.yaml` or `requirements.txt`, but there exists dependency varying with CUDA version, which can be hard to reproduce our work.

So I decided to guide whole installation procedure. 

If you get unexpected error, please refer to [FAQs]() describing solutions for the error. 

Feel free to upload a git issue to ask questions, including an installation guide.

## Basic Setup
```bash
conda create -n uasopi python=3.8 
conda activate uasopi

git clone https://github.com/HAMA-DL-dev/UASOPI.git
cd UASOPI
pip install -ve .
```

## [Pytorch](https://pytorch.org/get-started/previous-versions/)
```bash
# Check your cuda to install desirable version
nvcc --version

# CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

PLEASE refer to the hyper link for installtaion. 

Our work tested under CUDA 11.8 with Pytorch 2.0.0.

So there can be an error if your environment has other CUDA version.


## [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
python setup.py develop
```
🙏🙏Please run `demo.py`🙏🙏 before install belows

`segmentation fault(core dumped)` would occur if there is version compatibility problem between torch and pcdet.

## [torchsparse](https://github.com/mit-han-lab/torchsparse) 
```bash
sudo apt-get install libsparsehash-dev
git clone https://github.com/mit-han-lab/torchsparse.git
pip install -r requirements.txt
pip install -e .
```

## [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels)
```bash
# Check your torch, cuda version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"

# Fill out 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch_geometric
```
## [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/), [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
🙏🙏PLEASE refer to [compatibility matrix](https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix) 🙏🙏

There also exists compatibility between belows,
- `lightning.pytorch`
- `pytorch_lightning`
- `lightning.fabric`
- `torch`
- `torchmetrics`
- Python

```bash
# EXAMPLE for torch 2.2.0 AND torch 2.0.0
pip install pytorch-lightning==1.9
pip install torch_geometric==2.2.0
pip install torchmetrics==0.11.4
```
