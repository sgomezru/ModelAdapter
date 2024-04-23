
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

base_path = Path('/workspace/repositories/ModelAdapter/src').resolve()
# src_path = base_path / 'src'
cfg_path = base_path / 'configs'
pmri_data_path = Path('/data/Data/PMRI').resolve()
sys.path.append(str(base_path))

from model.unet import get_unet
from data_utils import get_pmri_data_loaders
from trainer.unet_trainer import get_unet_trainer

#####################CONFIG LOADING########################################
DATASET_KEY = 'prostate'
DATASET_SUBKEY = 'pmri'
ARCH = 'monai-unet-64-4-4'
PROJECT_NAME = 'modeladapt'
VALIDATION = True
LOG = True
LOAD_MODEL = False
LOAD_DATASETS = True
NUM_EPOCHS = 200
BATCH_SIZE = 32
TBPE = 250
VBPE = 25
RUN_ITERATION = 0
cfg = OmegaConf.load(cfg_path / 'conf.yaml')
OmegaConf.update(cfg, 'run.iteration', RUN_ITERATION)
OmegaConf.update(cfg, 'run.dataset_key', DATASET_KEY)
OmegaConf.update(cfg, 'run.data_key', DATASET_KEY)
OmegaConf.update(cfg, 'run.dataset_subkey', DATASET_SUBKEY)
OmegaConf.update(cfg, 'run.arch', ARCH)
OmegaConf.update(cfg, 'run.validation', VALIDATION)
OmegaConf.update(cfg, 'wandb.project', PROJECT_NAME)
OmegaConf.update(cfg, 'wandb.log', LOG)
OmegaConf.update(cfg, f'unet.{DATASET_KEY}.{DATASET_SUBKEY}.training.epochs', NUM_EPOCHS)
OmegaConf.update(cfg, f'unet.{DATASET_KEY}.{DATASET_SUBKEY}.training.batch_size', BATCH_SIZE)
OmegaConf.update(cfg, f'unet.{DATASET_KEY}.{DATASET_SUBKEY}.training.train_batches_per_epoch', TBPE)
OmegaConf.update(cfg, f'unet.{DATASET_KEY}.{DATASET_SUBKEY}.training.val_batches_per_epoch', VBPE)
### If want to load a pre-trained model
OmegaConf.update(cfg, 'run.load', LOAD_MODEL)
### If want to load the datasets
OmegaConf.update(cfg, 'run.load_dataset', LOAD_DATASETS)
###########################################################################
# For compatibility with previous version
args = ARCH.split('-')
cfg.unet[DATASET_KEY].pre = ARCH
cfg.unet[DATASET_KEY].arch = args[0]
cfg.unet[DATASET_KEY].n_filters_init = None if ARCH == 'swinunetr' else int(args[1])
if args[0] == 'monai':
    cfg.unet[DATASET_KEY].depth = int(args[2])
    cfg.unet[DATASET_KEY].num_res_units = int(args[3])

unet = get_unet(cfg, update_cfg_with_swivels=False, return_state_dict=False)
train_loader, val_loader = get_pmri_data_loaders(cfg=cfg)
pmri_trainer = get_unet_trainer(cfg=cfg, train_loader=train_loader, val_loader=val_loader, model=unet)
pmri_trainer.fit()
