
This document includes solution for expected errors.

## pytorch-lightning version mismatch [(ref)](https://github.com/state-spaces/s4/issues/93#issuecomment-1493074874)
**Error**
```bash 
TypeError: Trainer.__init__() got an unexpected keyword argument 'gpus'
```
**Solution**
```bash
pip install pytorch-lightning==1.9.3
```

## torchsparse [(ref)](https://github.com/mit-han-lab/torchsparse/issues/248#issuecomment-1969084571)
**Error**
```bash
Error: cannot import name 'PointTensor' from 'torchsparse'
```
**Solution**
```bash
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## numpy
**Error**
```bash
AttributeError: module 'numpy' has no attribute 'int'
```
**Solution**
```bash
pip install numpy==1.23.0
```

There is no reference. this is common error.

## torch_geometrics [(ref)](https://github.com/pyg-team/pytorch_geometric/discussions/5889)
**Error**
```bash
TypeError: 'method' object is not iterable
```
**Solution**
```bash
pip install networkx==2.8.*
```

## pcdet and kornia [(ref)](https://github.com/open-mmlab/OpenPCDet/issues/1470#issuecomment-1747974237)
**Error**
```bash
return C.quaternion_to_rotation_matrix(
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    quat_wxyz, order=C.QuaternionCoeffOrder.WXYZ
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
)
```
**Solution**
```bash
# Solution
pip install kornia==0.6.1
```

## hydra [(ref)](https://github.com/facebookresearch/hydra/issues/919)
**Error**
```bash
AttributeError: module 'hydra' has no attribute 'main'
```
**Solution**
```bash
pip uinstall hydra
pip install hydra-core
```
