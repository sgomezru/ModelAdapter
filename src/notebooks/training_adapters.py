import os
import sys
from omegaconf import OmegaConf
import wandb
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append('../')

from model.unet import get_unet
from data_utils import get_pmri_data_loaders
from trainer.unet_trainer import get_unet_trainer
from adapters import PCA_Adapter, PCAModuleWrapper
import torch.nn as nn

### Load basic config
DATA_KEY = 'prostate'
ITERATION = 0
AUGMENT = True
SUBSET = False # Whether the validation is a subset or the whole set
VALIDATION = True # If false makes validation set be the training one
N_DIMS = 16
cfg = OmegaConf.load('../configs/conf.yaml')
OmegaConf.update(cfg, 'run.iteration', ITERATION)
OmegaConf.update(cfg, 'run.data_key', DATA_KEY)

unet_name = 'monai-64-4-4'
extra_description = '_ipca_adapted'
cfg.wandb.project = f'{DATA_KEY}_{unet_name}_{ITERATION}{extra_description}'
args = unet_name.split('-')
cfg.unet[DATA_KEY].pre = unet_name
cfg.unet[DATA_KEY].arch = args[0]
cfg.unet[DATA_KEY].n_filters_init = None if unet_name == 'swinunetr' else int(args[1])
cfg.unet[DATA_KEY].training.augment = AUGMENT
cfg.unet[DATA_KEY].training.validation = VALIDATION
cfg.unet[DATA_KEY].training.subset = SUBSET
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

# Possible layers
# Probably bottleneck on encoder
# model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv
# model.1.submodule.1.submodule.1.submodule.0.conv.unit
# Probably bottleneck on decoder
# model.1.submodule.1.submodule.2.0.conv
# model.1.submodule.1.submodule.2.0

layer_names = [
    'model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv',
    'model.1.submodule.1.submodule.2.0.conv'
]

adapters = [PCA_Adapter(swivel, N_DIMS, cfg.unet[DATA_KEY].training.batch_size) for swivel in layer_names]
for adapter in adapters:
    adapter.training = True
adapters = nn.ModuleList(adapters)

unet = get_unet(cfg, update_cfg_with_swivels=False, return_state_dict=False)
train_loader, val_loader = get_pmri_data_loaders(cfg=cfg)

unet_adapted = PCAModuleWrapper(model=unet, adapters=adapters)
unet_adapted.hook_adapters()

pmri_trainer = get_unet_trainer(cfg=cfg, train_loader=train_loader, val_loader=val_loader, model=unet_adapted)

try:
    pmri_trainer.fit()
    for adapter in unet_adapted.adapters:
        name = adapter.swivel.replace('.', '_')
        with open(f'/workspace/src/out/pca/{name}.pkl',  'wb') as f:
            pickle.dump(adapter.pca, f)
finally:
    wandb.finish()
