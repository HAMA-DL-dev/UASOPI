import numpy as np
import torch
import torch.nn as nn

# from .....main.networks.adf import adf 
from main.networks.adf import adf 

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.dropout = self.model_cfg.DROPOUT
        if self.dropout:
            print("Dropout in Backbone2D : BaseBEVBackbone")
            
        # FIXME : 원본 for self.blocks
        for idx in range(num_levels):
            if idx == num_levels-1:
                c_in_list[idx] *= 2
            cur_layers = [
                # nn.ZeroPad2d(1),
                # nn.Conv2d(
                #     c_in_list[idx], num_filters[idx], kernel_size=3,
                #     stride=layer_strides[idx], padding=0, bias=False
                # ),
                # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                # nn.ReLU(),
                adf.ZeroPad2d(1),
                adf.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                adf.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                adf.ReLU(None, type_conv = "backbone2d"),
            ]
            
            for k in range(layer_nums[idx]):
                    cur_layers.extend([
                        # nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        # nn.ReLU(),
                        adf.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        adf.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        adf.ReLU(None, type_conv = "backbone2d"),
                    ])      
            # self.blocks.append(nn.Sequential(*cur_layers))   
            self.blocks.append(adf.Sequential(*cur_layers))   

        # FIXME : 만약 central dropout 을 backbone2d-self.blocks에서 쓴다면.     
        # for idx in range(num_levels):
        #     # [TODO] central dropout layer : dropout is added for only last level   
        #     if idx >= num_levels / 2: 
        #         cur_layers = [
        #             # nn.ZeroPad2d(1),
        #             # nn.Conv2d(
        #             #     c_in_list[idx], num_filters[idx], kernel_size=3,
        #             #     stride=layer_strides[idx], padding=0, bias=False
        #             # ),
        #             # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #             # nn.ReLU(),
        #             # # nn.Dropout(p = self.dropout),

        #             adf.ZeroPad2d(1),
        #             adf.Conv2d(
        #                 c_in_list[idx], num_filters[idx], kernel_size=3,
        #                 stride=layer_strides[idx], padding=0, bias=False
        #             ),
        #             adf.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #             adf.ReLU(None, type_conv = "backbone2d"),
        #             # adf.Dropout(p = self.dropout, keep_variance_fn = adf.keep_variance_fn),
        #         ]
        #         for k in range(layer_nums[idx]):
        #             # [TODO] Add dropout for only last layer : 마지막 level 중에서도 마지막 레이어에 dropout 배치시 사용 
        #             if k == layer_nums[idx] - 1:
        #                 cur_layers.extend([
        #                     # nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #                     # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #                     # nn.ReLU(),
        #                     # nn.Dropout(p = self.dropout)
        #                     adf.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #                     adf.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #                     adf.ReLU(None, type_conv = "backbone2d"),
        #                     adf.Dropout(p = self.dropout, keep_variance_fn = adf.keep_variance_fn)
        #                 ])
        #             else:
        #                 cur_layers.extend([
        #                     # nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #                     # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #                     # nn.ReLU(),
        #                     adf.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #                     adf.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #                     adf.ReLU(None, type_conv = "backbone2d"),
        #                 ])
                    
        #             # cur_layers.extend([
        #             #     nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #             #     nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #             #     nn.ReLU(),
        #             # ])
        #     else:
        #         cur_layers = [
        #             # nn.ZeroPad2d(1),
        #             # nn.Conv2d(
        #             #     c_in_list[idx], num_filters[idx], kernel_size=3,
        #             #     stride=layer_strides[idx], padding=0, bias=False
        #             # ),
        #             # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #             # nn.ReLU(),
        #             adf.ZeroPad2d(1),
        #             adf.Conv2d(
        #                 c_in_list[idx], num_filters[idx], kernel_size=3,
        #                 stride=layer_strides[idx], padding=0, bias=False
        #             ),
        #             adf.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #             adf.ReLU(None, type_conv = "backbone2d"),
        #         ]
        #         for k in range(layer_nums[idx]):
        #                 cur_layers.extend([
        #                     # nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #                     # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #                     # nn.ReLU(),
        #                     adf.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #                     adf.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #                     adf.ReLU(None, type_conv = "backbone2d"),
        #                 ])      
        #     # self.blocks.append(nn.Sequential(*cur_layers))
        #     self.blocks.append(adf.Sequential(*cur_layers))

            # Deblocks
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(adf.Sequential(
                        adf.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        adf.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        adf.ReLU(),
                    ))

                    # FIXME : self.deblocks에 central dropout을 쓴다면 주석 해제
                    # if idx >= num_levels / 2:
                    #     # self.deblocks.append(nn.Sequential(
                    #     #     nn.ConvTranspose2d(
                    #     #         num_filters[idx], num_upsample_filters[idx],
                    #     #         upsample_strides[idx],
                    #     #         stride=upsample_strides[idx], bias=False
                    #     #     ),
                    #     #     nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    #     #     nn.ReLU(),
                    #     #     # nn.Dropout(p = self.dropout), ### Last Layer of deblocks 
                    #     # ))
                    #     self.deblocks.append(adf.Sequential(
                    #         adf.ConvTranspose2d(
                    #             num_filters[idx], num_upsample_filters[idx],
                    #             upsample_strides[idx],
                    #             stride=upsample_strides[idx], bias=False
                    #         ),
                    #         adf.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    #         adf.ReLU(),
                    #         # adf.Dropout(p = self.dropout, keep_variance = adf.keep_variance_fn), ### Last Layer of deblocks 
                    #     ))
                    # else:
                    #     # self.deblocks.append(nn.Sequential(
                    #     #     nn.ConvTranspose2d(
                    #     #         num_filters[idx], num_upsample_filters[idx],
                    #     #         upsample_strides[idx],
                    #     #         stride=upsample_strides[idx], bias=False
                    #     #     ),
                    #     #     nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    #     #     nn.ReLU(),
                    #     # ))
                    #     self.deblocks.append(adf.Sequential(
                    #         adf.ConvTranspose2d(
                    #             num_filters[idx], num_upsample_filters[idx],
                    #             upsample_strides[idx],
                    #             stride=upsample_strides[idx], bias=False
                    #         ),
                    #         adf.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    #         adf.ReLU(),
                    #     ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(adf.Sequential(
                        adf.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        adf.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        adf.ReLU(),
                    ))
                    
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            # self.deblocks.append(nn.Sequential(
            #     nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
            #     nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
            #     nn.ReLU(),
            # ))
            self.deblocks.append(adf.Sequential(
                adf.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                adf.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                adf.ReLU(),
            ))
            
        self.num_bev_features = c_in
        
    def forward(self, data_dict, dropout_rate):
    # def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        # [TODO] propagated variance 
        var = data_dict['variance']
        
        ups = []
        ret_dict = {}
        x = spatial_features

        # [TODO] Update dropout rate varying with optimalization
        if dropout_rate is not None:
            
            # [Deprecated] DO NOT add dropout layer
            # self.deblocks.append(nn.Sequential(
            #     nn.Dropout(p = dropout_rate),
            # )) 
            
            self.deblocks[-1][-1].p = dropout_rate
    
        for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #     var = self.blocks[i](var)
        #     stride = int(spatial_features.shape[2] / x.shape[2])
        #     ret_dict['spatial_features_%dx' % stride] = x
        #     if len(self.deblocks) > 0:
        #         ups.append(self.deblocks[i](x))
        #         ups.append(self.deblocks[i](var))
        #     else:
        #         ups.append(x)
        #         ups.append(var)
        # #[TODO] 이거 수상한데, 확인필요함 
        # if len(ups) > 2:
        #     # x = torch.cat(ups), dim=1)
        #     x = torch.cat(ups[:int(len(ups) / 2)], dim=1)
        #     var = torch.cat(ups[int(len(ups) / 2):], dim=1)
        # elif len(ups) == 2:
        #     x = ups[0]
        #     var = ups[1]
        

        # if len(self.deblocks) > len(self.blocks):
        #     x = self.deblocks[-1](x)
        #     var = self.deblocks[-1](var)
            
        # data_dict['spatial_features_2d'] = x
        # data_dict['variance'] = var 
            
            x = spatial_features, var
            x = self.blocks[i](*x)
            stride = int(spatial_features.shape[2] / x[0].shape[2])
            ret_dict['spatial_features_%dx' % stride] = x[0]
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](*x))
            else:
                ups.append(x)

        # ups의 first dim.은 두 개의 (self.blocks + self.deblocks)에서 각각 나온 결과. 즉, index가 된다. 
        if len(ups) > 1:
            mean = torch.cat((ups[0][0], ups[1][0]), dim = 1)
            var = torch.cat((ups[0][1], ups[1][1]), dim = 1)
        elif len(ups) == 1:
            mean = (ups[0][0], ups[1][0])
            var = (ups[0][1], ups[1][1])
        if len(self.deblocks) > len(self.blocks):
            mean = self.deblocks[-1](mean)
            var = self.deblocks[-1](var)
            
        data_dict['spatial_features_2d'] = mean
        data_dict['variance'] = var         
        
        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
