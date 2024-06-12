import os
import sys
import wandb
import pickle
import torch
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append('../')

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from model.unet import get_unet
from data_utils import get_pmri_data_loaders, get_eval_data
from trainer.unet_trainer import get_unet_trainer
from adapters import PCA_Adapter, PCAModuleWrapper
from torch.utils.data import DataLoader

### Load basic config
DATA_KEY = 'prostate'
ITERATION = 1
LOG = False
AUGMENT = False
LOAD_ONLY_PRESENT = True
SUBSET = 'training' # Whether the validation is a subset or the whole set, normally bool, but for eval must be string 'training'
VALIDATION = True # If false makes validation set be the training one
N_DIMS = [2, 4, 8, 16, 32, 64]
cfg = OmegaConf.load('../configs/conf.yaml')
OmegaConf.update(cfg, 'run.iteration', ITERATION)
OmegaConf.update(cfg, 'run.data_key', DATA_KEY)

unet_name = 'monai-64-4-4'
extra_description = 'moredata'
cfg.wandb.project = f'{DATA_KEY}_{unet_name}_{ITERATION}{extra_description}'
args = unet_name.split('-')
cfg.unet[DATA_KEY].pre = unet_name
cfg.unet[DATA_KEY].arch = args[0]
cfg.unet[DATA_KEY].n_filters_init = None if unet_name == 'swinunetr' else int(args[1])
cfg.unet[DATA_KEY].training.augment = AUGMENT
cfg.unet[DATA_KEY].training.validation = VALIDATION
cfg.unet[DATA_KEY].training.subset = SUBSET
cfg.unet[DATA_KEY].training.load_only_present = LOAD_ONLY_PRESENT

if args[0] == 'monai':
    cfg.unet[DATA_KEY].depth = int(args[2])
    cfg.unet[DATA_KEY].num_res_units = int(args[3])

if LOG:
    wandb.init(
        project=cfg.wandb.project,
        config={
            "learning_rate": cfg.unet[DATA_KEY].training.lr,
            "architecture": unet_name,
            "dataset": DATA_KEY
        }
    )

layer_names = [
    'model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv',
    'model.1.submodule.1.submodule.2.0.conv'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if AUGMENT:
    cfg.format = 'torch'
    train_loader, val_loader = get_pmri_data_loaders(cfg=cfg)
else:
    cfg.format = 'numpy'
    data = get_eval_data(train_set=False, val_set=False, eval_set=True, cfg=cfg)
    dataset = data['eval']
    print(f'Length of non-aug dataset: {len(dataset)}')

try:
    for n_dims in N_DIMS:
        cfg.unet[DATA_KEY].training.batch_size = 58 if n_dims <= 58 else n_dims
        adapters = [PCA_Adapter(swivel, n_dims, cfg.unet[DATA_KEY].training.batch_size) for swivel in layer_names]
        for adapter in adapters:
            adapter.training = True
        adapters = nn.ModuleList(adapters)
        unet, state_dict = get_unet(cfg, update_cfg_with_swivels=False, return_state_dict=True)
        unet_adapted = PCAModuleWrapper(model=unet, adapters=adapters)
        unet_adapted.hook_adapters()
        unet_adapted.to(device);
        if AUGMENT:
            print(f'Training for PCA with {n_dims} dims on augmented data')
            pmri_trainer = get_unet_trainer(cfg=cfg, train_loader=train_loader, val_loader=val_loader, model=unet_adapted)
            pmri_trainer.fit_adapter()
        else:
            print(f'Training for PCA with {n_dims} dims on non-augmented data')
            unet_adapted.eval()
            dataloader = DataLoader(dataset,
                                    batch_size=cfg.unet[DATA_KEY].training.batch_size,
                                    shuffle=False, drop_last=False)

            for i, batch in enumerate(tqdm(dataloader)):
                input_ = batch['input'].to(device)
                if input_.size(0) < n_dims:
                    print(f'Skipped because batch size smaller than n_components')
                    continue
                unet_adapted(input_)

        for adapter in unet_adapted.adapters:
            name = adapter.swivel.replace('.', '_')
            if AUGMENT: name += '_aug'
            with open(f'/workspace/src/out/pca/{cfg.wandb.project}_{name}_{n_dims}dim.pkl',  'wb') as f:
                pickle.dump(adapter.pca, f)
finally:
    if LOG:
        wandb.finish()
