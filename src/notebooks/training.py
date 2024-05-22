import os
import sys
from omegaconf import OmegaConf
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('../')

from model.unet import get_unet
from data_utils import get_pmri_data_loaders
from trainer.unet_trainer import get_unet_trainer

### Load basic config
DATA_KEY = 'prostate'
ITERATION = 0
AUGMENT = False
SUBSET = True
VALIDATION = False
cfg = OmegaConf.load('../configs/conf.yaml')
OmegaConf.update(cfg, 'run.iteration', ITERATION)
OmegaConf.update(cfg, 'run.data_key', DATA_KEY)

unet_name = 'monai-64-4-4'
extra_description = '_noaug_overfit'
cfg.wandb.project = f'{DATA_KEY}_{unet_name}_{ITERATION}{extra_description}'
args = unet_name.split('-')
cfg.unet[DATA_KEY].pre = unet_name
cfg.unet[DATA_KEY].arch = args[0]
cfg.unet[DATA_KEY].n_filters_init = None if unet_name == 'swinunetr' else int(args[1])
# The following three are set to try to achieve overfitting
cfg.unet[DATA_KEY].training.augment = AUGMENT
cfg.unet[DATA_KEY].training.validation = VALIDATION # Makes validation set be the training one
cfg.unet[DATA_KEY].training.subset = SUBSET # Subsets validation
cfg.format = 'torch'
if args[0] == 'monai':
    cfg.unet[DATA_KEY].depth = int(args[2])
    cfg.unet[DATA_KEY].num_res_units = int(args[3])

wandb.init(
    project=cfg.wandb.project,
    config={
        "learning_rate": cfg.unet[DATA_KEY].training.lr,
        "architecture": unet_name,
        "dataset": DATA_KEY
    }
)

unet = get_unet(cfg, update_cfg_with_swivels=False, return_state_dict=False)
train_loader, val_loader = get_pmri_data_loaders(cfg=cfg)
pmri_trainer = get_unet_trainer(cfg=cfg, train_loader=train_loader, val_loader=val_loader, model=unet)

try:
    pmri_trainer.fit()
finally:
    wandb.finish()
