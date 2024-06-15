import torch
import torch.nn as nn
import logging

from torch_geometric.nn import radius as search_radius, knn as search_knn, avg_pool_x
import torch.nn.functional as F

from functools import partial

# from ..adf import adf 
from main.networks.adf import adf 

# from omegaconf import OmegaConf
# conf = OmegaConf.load("/home/mmc-server4/Server/Users/hayeon/uasopi/main/configs/cfg/nuscenes_second_kitti.yaml")

class InterpNet(torch.nn.Module):

    def __init__(self, latent_size, out_channels, K=1, radius=1.0,  spatial_prefix="", 
            intensity_loss=False,
            radius_search=True,
            column_search=False,
            uncertainty_loss=True,
            # dropout=conf["network"]["decoder_params"]["dropout"],
            dropout=0.2,
            ):
        super().__init__()

        self.intensity_loss = intensity_loss
        self.out_channels = out_channels
        self.column_search = column_search
        self.dropout_rate = dropout["rate"]
        self.uncertainty_loss = uncertainty_loss
        
        logging.info(f"InterpNet - radius={radius} - out_channels={self.out_channels} - dropout ratio = {self.dropout_rate}")
        print("Dropout in Decoder")
        
        # TODO : layers of the decoder
        # self.fc_in = torch.nn.Linear(latent_size+3, latent_size)
        # mlp_layers = [torch.nn.Linear(latent_size, latent_size) for _ in range(2)]
        # self.mlp_layers = nn.ModuleList(mlp_layers)
        # self.fc_out = torch.nn.Linear(latent_size, self.out_channels)
        # self.activation = torch.nn.ReLU()
        # self.spatial_prefix = spatial_prefix
        # self.dropout = torch.nn.Dropout(p=self.dropout_rate) 
        self.fc_in = adf.Linear(latent_size+3, latent_size)
        mlp_layers = [adf.Linear(latent_size, latent_size) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = adf.Linear(latent_size, self.out_channels)
        self.activation = adf.ReLU()
        self.spatial_prefix = spatial_prefix
        self.dropout = adf.Dropout(p=self.dropout_rate, keep_variance_fn = adf.keep_variance_fn)

        # search function
        if radius_search:
            self.radius = radius
            self.K = None
            self.search_function = partial(search_radius, r=self.radius)
        else:
            self.K = int(K)
            self.radius = None
            self.search_function = partial(search_knn, k = self.K)

    def forward(self, data, dropout_rate):

        # get the data
        if "latents_pos" in data:
            pos_source = data["latents_pos"]
            batch_source = data["latents_batch"]
        else:
            pos_source = data["pos"]
            batch_source = data["batch"]
        
        pos_target = data["pos_non_manifold"]
        batch_target = data["pos_non_manifold_batch"]
        latents = data["latents"]                              # torch.Size([563200, 128])
        
        #[TODO] variance totally propagated 
        var = data["variance"]                                 # torch.Size([563200, 128])

        # neighborhood search
        if self.column_search:
            row, col = self.search_function(x=pos_source[:,:2], y=pos_target[:,:2], batch_x=batch_source, batch_y=batch_target)
        else:
            row, col = self.search_function(x=pos_source, y=pos_target, batch_x=batch_source, batch_y=batch_target)

        # compute reltive position between query and input point cloud
        # and the corresponding latent vectors
        pos_relative = pos_target[row] - pos_source[col]       # torch.Size([654697, 3])
        latents_relative = latents[col]                        # torch.Size([654697, 128])
        x = torch.cat([latents_relative, pos_relative], dim=1) # torch.Size([654697, 131])
        var_relative = var[col]
        var = torch.cat([var_relative, pos_relative], dim=1)
    
        x = x.contiguous(), var
        x = self.fc_in(*x)
        for i, mlp in enumerate(self.mlp_layers):
            x = self.activation(*x)
            # x = self.dropout(*x)
            x = mlp(*x)

        if dropout_rate is not None:
            self.dropout.p = dropout_rate 

        x = self.dropout(*x)
        result = self.fc_out(*x)

        x = result[0]
        var = result[1]
        
        return_data = {"predictions":x[:, 0],} # occupancy_preds with shape [# of features, (data, index)]

        var = abs(var[:,0]).mean()
        ###################### Aleatoric Uncertainty ###########################
        if self.uncertainty_loss:
            # return_data["aleatoric_uncertainty"] = abs(var[:,0])
            return_data["aleatoric_uncertainty"] = var
        
            # uncert_loss = 0.5 * torch.log(abs(var[:,0])) + (x[:,0] - occupancies_gt.float())**2 / (2 * abs(var[:,0]))
            # uncert_loss = 0.5 * torch.log(abs(var[:,0])) + (x[:,0].mean() - occupancies_gt.float())**2 / (2 * abs(var[:,0])) # MSE loss 
            # return_data["uncertainty_loss"] = uncert_loss.mean()
        ########################################################################
        
        if "occupancies" in data:
            occupancies_gt = data["occupancies"][row]
            return_data["occupancies"] = occupancies_gt
            return_data["occupancies_preds"] = x[:,0]

            #### Reconstruction loss
            recons_loss = F.binary_cross_entropy_with_logits(x[:,0], occupancies_gt.float())
            # recons_loss = 0.5 * F.binary_cross_entropy_with_logits(x[:,0], occupancies_gt.float()) / var + torch.log(var)
            return_data["recons_loss"] = recons_loss

        #### intensity_loss
        if self.intensity_loss:
            intensities = data["intensities_non_manifold"][row].squeeze(-1)
            intensity_mask = (intensities >= 0)
            return_data["intensity_loss"] = F.l1_loss(x[:,1][intensity_mask], intensities[intensity_mask])
            # return_data["intensity_loss"] = 0.5 * F.l1_loss(x[:,1][intensity_mask], intensities[intensity_mask]) / var + torch.log(var)
            
        return return_data


