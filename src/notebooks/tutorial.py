# %% [markdown]
# ## Basic config, package loading

# %%
### Set CUDA device
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# %%
import sys
from omegaconf import OmegaConf
import wandb

sys.path.append('../')
from model.unet import get_unet

# %%
### Load basic config
cfg = OmegaConf.load('../configs/conf.yaml')
OmegaConf.update(cfg, 'run.iteration', 0)

# %% [markdown]
# ## Model selection

# %%
### Set dataset, either brain, heart or prostate
DATA_KEY = 'prostate'
OmegaConf.update(cfg, 'run.data_key', DATA_KEY)

### get model
# available models:
#     - default-8
#     - default-16
#     - monai-8-4-4
#     - monai-16-4-4
#     - monai-16-4-8
#     - monai-32-4-4
#     - monai-64-4-4
#     - swinunetr

unet_name = 'monai-64-4-4'
# unet_name = 'swinunetr'
args = unet_name.split('-')
cfg.unet[DATA_KEY].pre = unet_name
cfg.unet[DATA_KEY].arch = args[0]
cfg.unet[DATA_KEY].n_filters_init = None if unet_name == 'swinunetr' else int(args[1])
if args[0] == 'monai':
    cfg.unet[DATA_KEY].depth = int(args[2])
    cfg.unet[DATA_KEY].num_res_units = int(args[3])

# unet, state_dict = get_unet(
#     cfg,
#     update_cfg_with_swivels=False,
#     return_state_dict=True)
# unet.load_state_dict(state_dict)
# _ = unet.cuda()

wandb.init(
    project=cfg.wandb.project,
    config={
        "learning_rate": cfg.unet[DATA_KEY].training.lr,
        "architecture": unet_name,
        "dataset": DATA_KEY
    }
)


# %% [markdown]
# ## Train model

# %%
from data_utils import get_pmri_data_loaders
from trainer.unet_trainer import get_unet_trainer

unet = get_unet(cfg, update_cfg_with_swivels=False, return_state_dict=False)
train_loader, val_loader = get_pmri_data_loaders(cfg=cfg)
pmri_trainer = get_unet_trainer(cfg=cfg, train_loader=train_loader, val_loader=val_loader, model=unet)

# %%
try:
    pmri_trainer.fit()
finally:
    wandb.finish()
