#For Reading the arguments
from datetime import datetime

from numpy import arange, float32, int32
from numpy.random import choice

#Base PyTorch imports
from torch import from_numpy, ones
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR

#Dataset and data utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

#PyTorch Lightning
import pytorch_lightning as pl
from torchmetrics import PeakSignalNoiseRatio as PSNR
from pytorch_msssim import SSIM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

#Model
from models.laplacian_pyramid import LapPyramid
from models.pl_wrappers import PLIPWrapper
from models.archs_nafnet import BlurSimul

#Augmentation
from transforms import CenterCrop, Compose, HorizontalFlip, Norm01, RandomCrop, RandomTransposeHW, VerticalFlip, TimeFlip, ToTensor
from datasets import VideoFolderPair

#Color
from utils import  RGB2YCBCR
from losses import CharbonnierLoss

import torch
import os

W_DECAY     = 0
INIT_LR     = 1e-3
MIN_LR      = 1e-6
NBK         = 50
BATCH_SIZE  = 32
ITER_STEPS  = 5000#//BATCH_SIZE: each epoch has 5k steps, total number in one epoch is 32*5000
T_MAX       = ITER_STEPS*160 # number of epoches is 160
IM_SIZE     = 128
K_SIZE      = 15
N_GPUS      = 1
NUM_WORKERS = 16

# Set the CUDA device you want to use
# gpu_ids = "0, 1, 2, 3, 4, 5"
gpu_ids = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids



class BlurSimulPLWrapper(PLIPWrapper):
    def __init__(self):
        models = {
            'blur_simulator': BlurSimul(3, NBK, K_SIZE),
            'rgb2ycbcr':     RGB2YCBCR(only_y=True)      # Needed for PSNR and SSIM metrics.
        }
        
        super().__init__(models)

        optimizer_conf = {
            'class': optim.Adam,
            'lr': INIT_LR,
            'weight_decay': W_DECAY,
            'betas':(0.9, 0.9)
        }
        lr_scheduler_conf = {
            'class': CosineAnnealingLR,
            'T_max': T_MAX,
            'eta_min': MIN_LR,
            'verbose': False,
        }

        self.init_optimizer(optimizer_conf, lr_scheduler_conf)

        metrics = {
            'psnr': [PSNR(data_range=1.0).cuda(), {'preds': 'pY_Y', 'target': 'y_Y'}],
            'ssim': [SSIM(data_range=1.0, size_average=True, channel=1).cuda(), {'X': 'pY_Y', 'Y': 'y_Y'}],
        }
        metrics_train_val_only = {}

        self.loss = CharbonnierLoss(eps=1e-5)

        self.set_metrics(metrics, metrics_train_val_only)

        self.add_module('lap', LapPyramid(max_levels=2, channels=3))

    def degradation(self, batch_input: dict, mode: str, **kwargs): 
        return batch_input

    def run_loss(self, batch_input: dict, mode: str, **kwargs):
        
        x, y = batch_input['x'].squeeze(2), batch_input['y'].squeeze(2) #1, 3, 128, 128 or 32, 3, 128, 128

        pY = self.blur_simulator([x, y]) #32, 3, 128, 128/64/32
        hi_y, lw_y = self.lap(y)

        y = [y, lw_y[0], lw_y[1]] #32, 3, 128, 128 (/64/32)

        loss = 0.

        for i in range(3):
            loss += (self.loss(pY[i], y[i]))*2**(-i)
        
        loss = loss / 1.75
        
        batch_output = {
            'pY': pY[0], 
            'y': y[0], 
            'pY_Y': self.rgb2ycbcr(pY[0]), 
            'y_Y': self.rgb2ycbcr(y[0])
        }

        return loss, batch_output

    def forward(self, inp):
        pY = self.blur_simulator(inp)
        
        return pY

    def configure_optimizers(self):
        opt_a = {'optimizer': self.optimizer, }

        if self.lr_scheduler:
            opt_a['lr_scheduler'] = {
                "scheduler":  self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
                'monitor': 'train_loss_a'
            }

        return opt_a


if __name__=='__main__':
    #Parameters
    train_path_blur      = '/home/zhg9257/REDS/train/blur'
    train_path_label     = '/home/zhg9257/REDS/train/sharp'
    train_seq_list       = '/home/zhg9257/REDS/train/seq_list.txt'
    val_path_blur        = '/home/zhg9257/REDS/val/blur'
    val_path_label       = '/home/zhg9257/REDS/val/sharp'
    val_seq_list         = '/home/zhg9257/REDS/val/seq_list.txt'
    save_path_pattern    = 'weights_char/blur_simul_nbk_%d_k_size_%d_batch_size_%d_w_decay_%f_ini_lr_%f_date_%s'

    #Read Data
    train_transform = [
        RandomCrop(size=[IM_SIZE, IM_SIZE], keys=['y', 'x']),
        Norm01(min_val=0.0, max_val=255.0, new_type=float32, keys=['y', 'x']),
        HorizontalFlip(p=0.5, keys=['y', 'x']),
        VerticalFlip(p=0.5, keys=['y','x']),
        RandomTransposeHW(p=0.5, keys=['y','x']),
        ToTensor(keys=['y', 'x'], contiguous=True),
    ]
    train_transform = Compose(train_transform)
    val_transform = [
        CenterCrop(size=[IM_SIZE, IM_SIZE], keys=['y', 'x']), 
        Norm01(min_val=0.0, max_val=255.0,
               new_type=float32, keys=['y','x']),
        ToTensor(keys=['y', 'x'], contiguous=True),
    ]
    val_transform = Compose(val_transform)

    #Loaders
    train_dataset = VideoFolderPair(
        train_path_blur, 
        train_path_label, train_seq_list, 
        '%08d.png', transform=train_transform,
        max_l=1,
    )
    train_sampler = WeightedRandomSampler(weights=ones(
        len(train_dataset)).double(), num_samples=BATCH_SIZE//N_GPUS*ITER_STEPS) #num_samples: num of pics of one gpu=32*5000/1; 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE//N_GPUS, #batch_size=32; len(train_dataset)=240 environs
                              pin_memory=True, num_workers=NUM_WORKERS, sampler=train_sampler, persistent_workers=True)
    
    for targetss in train_loader: # torch.Size([32, 3, 1, 128, 128])
        break
        # print("Target shape:", targets)

    val_dataset = VideoFolderPair(
        val_path_blur,  
        val_path_label, val_seq_list, 
        '%08d.png', transform=val_transform,
        max_l=1,
    )
    val_sampler = WeightedRandomSampler(weights=ones(
        len(val_dataset)).double(), num_samples=int(len(val_dataset)/6)) #num_samples: num of pics of one gpu=5? 
    val_loader = DataLoader(val_dataset,  batch_size=1,  #batch_size=1; len(train_dataset)=30 environs
                            pin_memory=True, num_workers=NUM_WORKERS, sampler=val_sampler, persistent_workers=True) 

    for targets in val_loader: # torch.Size([1, 3, 1, 128, 128]) or torch.Size([2, 3, 1, 128, 128])
        break
        # print("Target shape:", targets['y'][0].shape)

    #Init model
    date  = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    save_path = save_path_pattern % (NBK, K_SIZE, BATCH_SIZE, W_DECAY, INIT_LR, date)
    model = BlurSimulPLWrapper()

    #Train
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_psnr", mode="max", save_top_k=5)
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", 
                         devices=N_GPUS, 
                         callbacks=[lr_monitor, checkpoint_callback], 
                         default_root_dir=save_path,  max_epochs=T_MAX//ITER_STEPS, # ITER_STEPS  = 5000, T_MAX = ITER_STEPS*160
                        #  replace_sampler_ddp=False,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=1)

    # Get the index of the current device
    current_device = torch.cuda.current_device()

    # Get the name of the current device
    device_name = torch.cuda.get_device_name(current_device)

    print(f"The current network is running on GPU {current_device}, device name: {device_name}")
    
    trainer.fit(model, train_loader, val_loader)
