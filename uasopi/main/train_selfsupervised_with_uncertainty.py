import os
import yaml
import logging
import warnings
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np

from sklearn.metrics import confusion_matrix


from torch_geometric.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

import datasets
import networks.decoder

import utils.metrics as metrics
from utils.utils import wblue, wgreen
from utils.callbacks import CustomProgressBar
from transforms import get_transforms, get_input_channels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config_file(config, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def logs_file(filepath, epoch, log_data):

    
    if not os.path.exists(filepath):
        log_str = f"epoch"
        for key, value in log_data.items():
            log_str += f", {key}"
        log_str += "\n"
        with open(filepath, "a+") as logs:
            logs.write(log_str)
            logs.flush()

    # write the logs
    log_str = f"{epoch}"
    for key, value in log_data.items():
        log_str += f", {value}"
    log_str += "\n"
    with open(filepath, "a+") as logs:
        logs.write(log_str)
        logs.flush()


class LightningSelfSupervisedTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config
        
        if config["network"]["backbone_params"] is None:
            config["network"]["backbone_params"] = {}
        config["network"]["backbone_params"]["in_channels"] = get_input_channels(config["inputs"])
        config["network"]["backbone_params"]["out_channels"] = config["network"]["latent_size"]

        backbone_name = "networks.backbone."
        if config["network"]["framework"] is not None:
            backbone_name += config["network"]["framework"]
        importlib.import_module(backbone_name)
        backbone_name += "."+config["network"]["backbone"]

        logging.info(f"Backbone - {backbone_name}")

        self.backbone = eval(backbone_name)(**config["network"]["backbone_params"])

        logging.info(f"Decoder - {config['network']['decoder']}")

        config["network"]["decoder_params"]["latent_size"] = config["network"]["latent_size"]
        self.decoder = eval("networks.decoder."+config["network"]["decoder"])(**config["network"]["decoder_params"])

        self.train_cm = np.zeros((2,2))
        self.val_cm = np.zeros((2,2))

        self.mc_samples = config["monte-carlo_method"]["mc_samples"]
        self.init_drop_rate = config["monte-carlo_method"]["drop_rate"]
        self.au = 0
        self.eu = 0
        self.total_uncertainty = 0
        self.uncertainty_quantification = config["uncertainty_quantification"]

    def forward(self, data, drop_rate):
        
        outputs = self.backbone(data,drop_rate)
        
        if isinstance(outputs, dict):
            for k,v in outputs.items():
                data[k] = v
        else:
            data["latents"] = outputs
            
        return_data = self.decoder(data, drop_rate)

        return return_data

    
    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"])(self.parameters(), **self.config["optimizer_params"])
        return optimizer


    def compute_confusion_matrix(self, output_data):
        outputs = output_data["predictions"].squeeze(-1)
        occupancies = output_data["occupancies"].float()
        
        output_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
        target_np = occupancies.cpu().numpy().astype(int)
        cm = confusion_matrix(
            target_np.ravel(), output_np.ravel(), labels=list(range(2))
        )
        return cm

    #[TODO] Epistemic Uncertainty Quantification
    def compute_epistemic_uncertainty(self, preds, gt):
        return (preds - gt.float())**2
        
    def compute_loss(self, output_data, prefix):

        loss = 0
        loss_values = {}
        for key, value in output_data.items():
            if "loss" in key and (self.config["loss"][key+"_lambda"] > 0):
                value *= self.config["loss"][key+"_lambda"] 
                loss = loss + value
                # self.log(prefix+"/"+key, value.item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)
                self.log(prefix+"/"+key, value.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
                loss_values[key] = value.item()
                
        ## [Deprecated - TODO] NLL loss for AU : decoder에서 AU loss를 받아오기로 함.
        # if self.uncertainty_quantification:
            # [OOM error] CUDA Out Of Memory 
            # self.uncertainty_loss = torch.log(self.au) / 2 + (output_data["occupancies_preds"]- output_data["occupancies"])**2 / (2 * self.au)
            # self.log("uncertainty/loss",self.uncertainty_loss.mean(), on_step = False, on_epoch = True, prog_bar = True, logger = True)
            # loss += self.uncertainty_loss.mean()
            # loss_values["uncertainty_loss"] = self.uncertainty_loss.mean()
            
            # [OOM error] CUDA Out Of Memory 
            # self.uncertainty_loss = torch.log(self.au.mean()) / 2 + (output_data["occupancies_preds"].float().mean()- output_data["occupancies"].float().mean())**2 / (2 * self.au.mean())
            # self.log("uncertainty/loss",self.uncertainty_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
            # loss += self.uncertainty_loss
            # loss_values["uncertainty_loss"] = self.uncertainty_loss

            # [Succeed, but no gradient upate]
            # self.au += 1e-6 # exception for zero division 
            # self.au = self.au.detach().cpu().numpy()
            # preds = output_data["occupancies_preds"].detach().cpu().numpy()
            # gt = output_data["occupancies"].detach().cpu().numpy()
            
            # self.uncertainty_loss = np.log(self.au) / 2 + (preds.mean() - gt)**2 / (2 * self.au)
            # self.uncertainty_loss *= self.config["loss"]["uncertainty_loss_lambda"]
            # self.log("uncertainty/loss",self.uncertainty_loss.mean(), on_step = False, on_epoch = True, prog_bar = True, logger = True)
            # loss += self.uncertainty_loss.mean()
            # loss_values["uncertainty_loss"] = self.uncertainty_loss.mean()
            
        if self.train_cm.sum() > 0:
            self.log(prefix+"/iou", metrics.stats_iou_per_class(self.train_cm)[0], on_step=True, on_epoch=False, prog_bar=True, logger=False)
        
        # [TODO] log also the total loss : train, val
        self.log(prefix+"/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss, loss_values #loss, individual_losses

    def on_train_epoch_start(self) -> None:
        self.train_cm = np.zeros((2,2))
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.val_cm = np.zeros((2,2))
        return super().on_validation_epoch_start()

    def training_step(self, data, batch_idx):

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        output_data = self.forward(data, self.init_drop_rate)

        #[TODO] uncertainty quantification 
        if self.uncertainty_quantification:
            self.au = output_data["aleatoric_uncertainty"]
        
        loss, individual_losses = self.compute_loss(output_data, prefix="train")
        cm = self.compute_confusion_matrix(output_data)
        self.train_cm += cm
        
        individual_losses["loss"] = loss

        return individual_losses

    def validation_step(self, data, batch_idx):
        
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        output_data = self.forward(data, self.init_drop_rate)
        
        #[TODO] Uncertainty Quantification 
        if self.uncertainty_quantification:
            # self.eu = 0
            # for i in range(self.mc_samples):
            #     output_data = self.forward(data, self.init_drop_rate)
            #     self.eu += self.compute_epistemic_uncertainty(output_data["occupancies_preds"],output_data["occupancies"])
            self.au = output_data["aleatoric_uncertainty"]
            # self.eu = self.eu / self.mc_samples
            # self.total_uncertainty = self.eu + self.au
            
        loss, individual_losses = self.compute_loss(output_data, prefix="val")
        cm = self.compute_confusion_matrix(output_data)
        self.val_cm += cm
        
        individual_losses["loss"] = loss

        return individual_losses

    def predict_step(self, data, batch_idx):
            
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        # [TODO] Activate Dropout Layer to quantify EU
        # torch.set_grad_enabled(True) # CUDA OOM error 
        self.backbone.train()
        self.decoder.train()

        output_data = self.forward(data, self.init_drop_rate)
        
        #[TODO] Uncertainty Quantification 
        if self.uncertainty_quantification:
            self.eu = 0
            for i in range(self.mc_samples):
                output_data = self.forward(data, self.init_drop_rate)
                self.eu += self.compute_epistemic_uncertainty(output_data["occupancies_preds"],output_data["occupancies"])
            self.au = output_data["aleatoric_uncertainty"]
            self.eu = self.eu / self.mc_samples
            self.total_uncertainty = self.eu + self.au
        self.log("epistemic_uncertainty",self.eu.mean(), on_step = False, on_epoch = True, prog_bar=True, logger = True)
        # torch.set_grad_enabled(False)
        self.backbone.eval()
        self.decoder.eval()

        #[TODO] Optimal Dropout Rate (ref. SalsaNext)
        from scipy.optimize import minimize,Bounds
        
        def cost_func(dropout_rate):
            dropout_rate = dropout_rate[0]
            return torch.log(self.total_uncertainty) / 2 + (self.forward(data, dropout_rate)["occupancies_preds"]- output_data["occupancies"])**2 / (2 * self.total_uncertainty)
        
        self.opt_dropout_rate = minimize(cost_func, [self.init_rate], method="BFGS", bounds = Bounds([0.0],[1.0]))

        # Can NOT log anything at prediction 
        # self.log("optimal_dropout_rate", self.opt_dropout_rate.x[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            "Epistemic Uncertainty" : self.eu,
            "Aleatoric Uncertainty" : self.au,
            "Total Uncertainty" : self.total_uncertainty,
            "Optimal Dropout Rate" : self.opt_dropout_rate.x[0]
        }
    
    def compute_log_data(self, outputs, cm, prefix):
        
        # compute iou
        iou = metrics.stats_iou_per_class(cm)[0]
        self.log(prefix+"/iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("epistemic_uncertainty",self.eu, on_step = False, on_epoch = True, prog_bar=True, logger = True)
        self.log("aleatoric_uncertainty",self.au, on_step = False, on_epoch = True, prog_bar=True, logger = True)
        # self.log("total_uncertainty",self.au.mean()+self.eu, on_step = False, on_epoch = True, prog_bar=True, logger = True)
        
        log_data = {}
        keys = outputs[0].keys()
        for key in keys:
            if "loss" not in key:
                continue
            if key == "loss":
                loss = np.mean([d[key].item() for d in outputs])
            else:
                loss = np.mean([d[key] for d in outputs])
            log_data[key] = loss

        log_data["iou"] = iou
        log_data["steps"] = self.global_step

        return log_data

    def get_description_string(self, log_data):
        desc = f"Epoch {self.current_epoch} |"
        for key, value in log_data.items():
            if "iou" in key:
                desc += f"{key}:{value*100:.2f} |"
            elif "steps" in key:
                desc += f"{key}:{value} |"
            else:
                desc += f"{key}:{value:.3e} |"
        return desc


    def training_epoch_end(self, outputs):


        log_data = self.compute_log_data(outputs, self.train_cm, prefix="train")

        try:
            os.makedirs(self.logger.log_dir, exist_ok=True)
            logs_file(os.path.join(self.logger.log_dir, "logs_train.csv"), self.current_epoch, log_data)
        except:
            os.makedirs(self.logger.save_dir, exist_ok=True)
            logs_file(os.path.join(self.logger.save_dir, "logs_train.csv"), self.current_epoch, log_data)

        if (self.global_step > 0) and (not self.config["interactive_log"]):
            desc = "Train "+ self.get_description_string(log_data)
            print(wblue(desc))


    def validation_epoch_end(self, outputs):
        
        if self.global_step > 0:

            log_data = self.compute_log_data(outputs, self.val_cm, prefix="val")

            try:
                os.makedirs(self.logger.log_dir, exist_ok=True)
                logs_file(os.path.join(self.logger.log_dir, "logs_val.csv"), self.current_epoch, log_data)
            except:
                os.makedirs(self.logger.save_dir, exist_ok=True)
                logs_file(os.path.join(self.logger.save_dir, "logs_val.csv"), self.current_epoch, log_data)

            if (not self.config["interactive_log"]):
                desc = "Val "+ self.get_description_string(log_data)
                print(wgreen(desc))



def get_savedir_name(config):
    
    ##########################################################################
    # [TODO] rename save directory 
    savedir = f"{config['network']['backbone']}_{config['network']['decoder']}_one_drop0.2_tail_of_backbone3d_decoder_demo_adf"
    ##########################################################################
    
    if config["network"]['framework'] is not None:
        savedir += f"_{config['network']['framework']}"
    savedir += f"_{config['manifold_points']}_{config['non_manifold_points']}"
    savedir += f"_{config['train_split']}Split"
    savedir += f"_radius{config['network']['decoder_params']['radius']}"
    if ("desc" in config) and config["desc"]:
        savedir += f"_{config['desc']}"

    return savedir

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config : DictConfig):

    config = OmegaConf.to_container(config)["cfg"]

    warnings.filterwarnings("ignore", category=UserWarning) 
    logging.getLogger().setLevel(config["logging"])

    logging.info("Getting the dataset and dataloader")
    DatasetClass = eval("datasets."+config["dataset_name"])
    train_transforms = get_transforms(config, train=True)
    test_transforms = get_transforms(config, train=False)

    # build the dataset
    train_dataset = DatasetClass(config["dataset_root"], 
                split=config["train_split"], 
                transform=train_transforms, 
                )
    val_dataset = DatasetClass(config["dataset_root"],
                split=config["val_split"], 
                transform=test_transforms, 
                )

    # build the data loaders
    train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["threads"],
            follow_batch=["pos_non_manifold", "voxel_coords", "voxel_proj_yx"]
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["threads"],
        follow_batch=["pos_non_manifold", "voxel_coords", "voxel_proj_yx"]
    )

    logging.info("Creating trainer")

    savedir_root = get_savedir_name(config)
    savedir_root = os.path.join(config["save_dir"], "Pretraining", savedir_root)

    logging.info(f"Savedir_root {savedir_root}")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # tb_logger
    logger = pl_loggers.TensorBoardLogger(save_dir=savedir_root)

    if config["wandb"]["usage"]:
        import wandb
        from pytorch_lightning.loggers.wandb import WandbLogger
        logger = WandbLogger(
            project = config["wandb"]["project"],
            save_dir = savedir_root, # config["wandb"]["save_dir"],
            entity = config["wandb"]["entity"],
            name = config["wandb"]["name"]
        )
        
    trainer = pl.Trainer(
            gpus= config["num_device"],
            check_val_every_n_epoch=config["training"]["val_interval"],
            logger=logger,
            max_epochs=config["training"]["max_epochs"],
            callbacks=[
                CustomProgressBar(refresh_rate=int(config["interactive_log"])),
                lr_monitor
                ]
            )

    try:
        # save the config file
        logging.info(f"Saving at {trainer.logger.log_dir}")
        os.makedirs(trainer.logger.log_dir, exist_ok=True)
        yaml.dump(config, open(os.path.join(trainer.logger.log_dir, "config.yaml"), "w"), default_flow_style=False)
        
    except:
        logging.info(f"Saving at {trainer.logger.save_dir}")
        os.makedirs(trainer.logger.save_dir, exist_ok=True)
        yaml.dump(config, open(os.path.join(trainer.logger.save_dir, "config.yaml"), "w"), default_flow_style=False)
    
    model = LightningSelfSupervisedTrainer(config)
    trainer.fit(model, train_loader, val_loader, ckpt_path=config["resume"])

    predictions = trainer.predict(model, val_loader, ckpt_path=config["resume"])
    for pred in predictions:
        print(f"Epistemic Uncertainty: {pred['Epistemic Uncertainty']} \n Aleatoric Uncertainty : {pred['Aleatoric Uncertainty']} \n Optimal Dropout Rate: {pred['Optimal Dropout Rate']}")
        
    
if __name__ == "__main__":
    main()