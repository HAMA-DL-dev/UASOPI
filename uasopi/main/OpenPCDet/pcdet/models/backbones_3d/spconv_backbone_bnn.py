from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
# from .....main.networks.adf import adf 
from main.networks.adf import adf 

from omegaconf import OmegaConf
conf = OmegaConf.load("/home/mmc-server4/Server/Users/hayeon/uasopi/main/configs/cfg/nuscenes_second_kitti.yaml")

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, dropout = False, use_adf = False):

    if use_adf:
        if conv_type == 'subm':
            conv = adf.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            conv = adf.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       bias=False, indice_key=indice_key)
        elif conv_type == 'inverseconv':
            conv = adf.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        else:
            raise NotImplementedError    
    else:
        if conv_type == 'subm':
            conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       bias=False, indice_key=indice_key)
        elif conv_type == 'inverseconv':
            conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        else:
            raise NotImplementedError
        
    if dropout and use_adf:
        print("Dropout in Backbone3D : SPCONV-{} with ADF".format(indice_key))
        m = adf.SparseSequential(
            conv,
            norm_fn(out_channels),
            adf.ReLU(keep_variance_fn = adf.keep_variance_fn),
            adf.Dropout(p=0.2, keep_variance_fn = adf.keep_variance_fn),
        )
    elif not dropout and use_adf:     
        m = adf.SparseSequential(
                conv,
                norm_fn(out_channels),
                adf.ReLU(keep_variance_fn = adf.keep_variance_fn),
        )
    elif dropout and not use_adf:
        print("Dropout in Backbone3D : SPCONV-{}".format(indice_key))
        m = spconv.SparseSequential(
            conv,
            norm_fn(out_channels),
            nn.ReLU(keep_variance_fn = adf.keep_variance_fn),
            nn.Dropout(p=0.2),
        )
        
    elif not dropout and not use_adf:
        m = spconv.SparseSequential(
                conv,
                norm_fn(out_channels),
                nn.ReLU(),
        )
    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        
        
        # self.relu = nn.ReLU()
        self.relu = adf.ReLU(keep_variance_fn = adf.keep_variance_fn)
        
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

# SECOND
class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        
        # TODO : ADF based model 
        norm_fn = partial(adf.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = adf.SparseSequential(
            adf.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            adf.ReLU(keep_variance_fn = adf.keep_variance_fn),
        )
        
        block = post_act_block

        self.conv1 = adf.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', dropout = False, use_adf = True),
        )
        self.conv2 = adf.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv',dropout = False, use_adf = True),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',dropout = False, use_adf = True),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',dropout = False, use_adf = True),
        )
        self.conv3 = adf.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            # [TODO] Activate dropout with prefix 
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv',dropout = False, use_adf = True),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',dropout = False, use_adf = True),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',dropout = False, use_adf = True),
        )
        self.conv4 = adf.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', 
                  conv_type='spconv',dropout = False, use_adf = True),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',dropout = False, use_adf = True),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',dropout = False, use_adf = True),
        )
        # TODO : Original Model 
        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.conv_input = spconv.SparseSequential(
        #     spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
        #     norm_fn(16),
        #     nn.ReLU(),
        # )
        # self.conv1 = spconv.SparseSequential(
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', dropout = False, use_adf = False),
        # )
        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv',dropout = False, use_adf = False),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',dropout = False, use_adf = False),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',dropout = False, use_adf = False),
        # )
        # self.conv3 = spconv.SparseSequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     # [TODO] Activate dropout with prefix 
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv',dropout = False, use_adf = False),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',dropout = False, use_adf = False),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',dropout = False, use_adf = False),
        # )
        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', 
        #           conv_type='spconv',dropout = False, use_adf = False),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',dropout = False, use_adf = False),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',dropout = False, use_adf = False),
        # )
        
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)

        # [TODO] Dropout Layer in Decoder : Conv_out
        if conf["network"]["decoder_params"]["dropout"]["usage"]:
            print("Dropout in Backbone3D : SPCONV - conv-out")
            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
                nn.Dropout(p = conf["network"]["decoder_params"]["dropout"]["rate"]),
            )
            self.conv_out = adf.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                adf.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                adf.ReLU(keep_variance_fn = adf.keep_variance_fn),
                adf.Dropout(p = conf["network"]["decoder_params"]["dropout"]["rate"], keep_variance_fn = adf.keep_variance_fn),
            )
            
        else:
            # self.conv_out = spconv.SparseSequential(
            #     # [200, 150, 5] -> [200, 150, 2]
            #     spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
            #                         bias=False, indice_key='spconv_down2'),
            #     norm_fn(128),
            #     nn.ReLU(),
            # )
            self.conv_out = adf.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                adf.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                adf.ReLU(keep_variance_fn = adf.keep_variance_fn),
            )            
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict, dropout_rate):
    # def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # TODO : Input Variance for Assumed Density Filtering (ADF)
        import torch
        # noise = torch.randn_like(voxel_features) + 2e-7
        noise = torch.zeros_like(voxel_features) + 2e-7
        input_var_tensor = spconv.SparseConvTensor(
            features = noise,
            indices = voxel_coords.int(),
            spatial_shape = self.sparse_shape,
            batch_size = batch_size 
        )
        x = input_sp_tensor, input_var_tensor

        x_input = self.conv_input(*x)
        x_conv1 = self.conv1(*x_input)
        x_conv2 = self.conv2(*x_conv1)
        x_conv3 = self.conv3(*x_conv2)
        x_conv4 = self.conv4(*x_conv3)
        
        # [TODO] Update dropout rate varying with optimalization
        if dropout_rate is not None:
            
            # [Deprecated] Illegal Memory Access 
            # self.conv_out = spconv.SparseSequential(
            #     spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=self.last_pad,
            #                         bias=False, indice_key='spconv_down2'),
            #     self.norm_fn(128),
            #     nn.ReLU(),
            #     nn.Dropout(p = batch_dict['dropout_rate']),
            # ) 
        
            for i, layer in enumerate(self.conv_out):
                if type(layer) == type(nn.Dropout(0.0)):
                    layer.p = dropout_rate
                
        x_out, var_out = self.conv_out(*x_conv4)
        
        batch_dict.update({
            'encoded_spconv_tensor': x_out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        #[TODO] batch dictionary update for input variance
        batch_dict.update({
            'variance': {
                'var_init': input_var_tensor,
                # 'var_conv1': var_conv1,
                # 'var_conv2': var_conv2,
                # 'var_conv3': var_conv3,
                # 'var_conv4': var_conv4,
                'var_output': var_out,
            }
        })
        
        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        noise = torch.randn_like(voxel_features) + 2e-7
        input_var_tensor = spconv.SparseConvTensor(
            features = noise,
            indices = voxel_coords.int(),
            spatial_shape = self.sparse_shape,
            batch_size = batch_size 
        )
        
        x = self.conv_input(input_sp_tensor)
        var = self.conv_input(input_var_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        var_conv1 = self.conv1(var)
        var_conv2 = self.conv2(var_conv1)
        var_conv3 = self.conv3(var_conv2)
        var_conv4 = self.conv4(var_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        out_var = self.conv_out(var_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        batch_dict.update({
            'variance': {
                'var_init': input_var_tensor,
                'var_conv1': var_conv1,
                'var_conv2': var_conv2,
                'var_conv3': var_conv3,
                'var_conv4': var_conv4,
                'var_output': out_var,
            }
        })

        return batch_dict
