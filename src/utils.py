import os, sys
import random
from typing import (
    Iterable, 
    Dict, 
    Callable, 
    Tuple, 
    Union, 
    List
)
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, auc
import nibabel as nib
import wandb
import matplotlib.pyplot as plt



### Evaluation
class UMapScorePerSlice(nn.Module):
    """Calculates scores for each input slice.
    """
        
    def __init__(
        self, 
        reduction: str
    ):
        super().__init__()
        self.r = reduction
        # TODO: default to keep only first dim,
        #       flexible where possible (e.g. mean)
        self.dim = (1,2,3)
        
    @torch.no_grad()
    def forward(
        self, 
        umap:   Tensor = None, 
        pred:   Tensor = None, # original segmentation prediction
        pred_r: Tensor = None  # re-sampled segmentation prediction
    ) -> Tensor:
        
        if self.r == 'mean':
            assert umap is not None, "reduction is set to mean over umap, but umap is None"
            return umap.mean()
        
        elif self.r == 'norm':
            # The frobenius (i.e. euclidean) norm is simply the sum of squared elements, square rooted
            assert umap is not None, "reduction is set to norm of umap, but umap is None"
            # assert self.dim == (1,2,3), "setting the dim parameter doesn't do anything for norm reduction"
            return torch.norm(umap)
        
        elif self.r == 'nflips':
            assert pred is not None, "reduction is set to fraction of flipped predictions, but pred is None"
            assert pred_r is not None, "reduction is set to fraction of flipped predictions, but pred_r is None"
            flip_mask = (torch.argmax(pred, dim=1, keepdim=True) != torch.argmax(pred_r, dim=1, keepdim=True)) * 1.
            return flip_mask.sum()
        
        elif self.r == 'nNflips':
            assert pred is not None, "reduction is set to number of flipped predictions, but pred is None"
            assert pred_r is not None, "reduction is set to number of flipped predictions, but pred_r is None"
            flip_mask = (torch.argmax(pred, dim=1, keepdim=True) != torch.argmax(pred_r, dim=1, keepdim=True)) * 1.
            return flip_mask.mean()

        

class Thresholder(nn.Module):
    """
    Thresholds uncertainty maps.
    
    PyTorch module to threshold uncertainty maps at n_taus thresholds.
    Placement of thresholds is done outside the class, mostly because 
    quantile thresholding needs lots of information that are not part of
    this class.
    """
    
    def __init__(
        self, 
        taus
    ):
        super().__init__()
        self.register_buffer("taus", taus, persistent=False)
    
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return (x.detach() >= self.taus).float()
    
    
    

class UMapGenerator(nn.Module):
    """
    Calculates uncertainty maps from UNets in different ways.
    
    PyTorch Module to generate uncertainty maps from
    * VAE samples
    * Entropy in drop out samples
    * Entropy in model outputs
    """
    
    def __init__(
        self,
        method: str,  # 'ae'
        net_out: str,  # 'mms' or 'calgary'
    ):
        super().__init__()
        self.method  = method
        self.net_out = net_out
        self.m       = nn.Softmax(dim=1) if net_out=='heart' else nn.Sigmoid()
        self.ce      = nn.CrossEntropyLoss(reduction='none') if net_out=='heart' else nn.BCEWithLogitsLoss(reduction='none')
    
    @torch.no_grad()
    def forward(self, x: Tensor, batch_size: int = 1) -> Tensor:
        
        if self.method == 'none':
            return None
        
        x = x.detach()

        #################################
        ### Cross and regular Entropy ###
        #################################

        if self.method == 'cross_entropy':
            umap = self.ce(x[:batch_size], self.m(x[batch_size:]))
            if len(umap.shape) == 3:
                umap = umap.unsqueeze(1)
            # umap = umap.mean(dim=1, keepdims=True)
            
        elif self.method == 'entropy':          
            x_prob = self.m(x[:batch_size])
            umap = torch.special.entr(x_prob).sum(dim=1, keepdims=True)
            # umap = torch.distributions.Categorical(x_prob.permute(0,2,3,1)).entropy().unsqueeze(1)
        
        #################################
        ### experimental / M&M only   ###
        #################################
    
        elif self.method == 'kl_divergence':
            x_in = F.log_softmax(x[:batch_size], dim=1)
            umap = self.kl(x_in, self.m(x[batch_size:]))
            umap = umap.sum(dim=(1), keepdims=True)
            
        elif self.method == 'mse':
            x      = self.m(x)
            x     -= x.min(dim=1, keepdims=True).values
            x     /= x.sum(dim=1, keepdims=True)
            umap   = torch.pow(x[:batch_size] - x[batch_size:], 2).mean(1, keepdim=True)
            # umap   = umap.mean(dim=1, keepdims=True)            
            # print(umap.shape)
        #################################
        ###   old umaps from MICCAI   ###
        #################################
        
#         if self.method == 'ae':
#             if self.net_out == 'mms':                
#                 umap = self.ce(x[:1], self.m(x[1:]))
#                 #umap = umap.mean(dim=(0, 1), keepdims=True)
#                 #print(umap.shape)
#                 umap = umap.mean(dim=0, keepdims=True)
# #                 x      = self.m(x)
# #                 x     -= x.min(dim=1, keepdims=True).values
# #                 x     /= x.sum(dim=1, keepdims=True)
# #                 umap   = torch.pow(x[:1] - x[1:], 2).mean(0, keepdim=True)
# #                 umap   = umap.mean(dim=1, keepdims=True)
                
#             elif self.net_out == 'calgary':
#                 x    = torch.sigmoid(x)
#                 umap = torch.pow(x[:1] - x[1:], 2).mean(0, keepdim=True)
# #                 umap = self.ce(x[:1] - self.m(x[1:]))
# #                 umap = 
                
                
#         elif self.method == 'entropy':          

#             if self.net_out == 'mms':
#                 #print('x', x.shape)
#                 #x_argmax  = torch.argmax(x, dim=1)
#                 #print('2',x_argmax.shape)
#                 #x_one_hot = F.one_hot(x_argmax, num_classes=4).permute(0,3,1,2).float()
#                 #print('3',x_one_hot.shape)
#                 x_softmax = F.softmax(x, dim=1)
#                 #print('soft',x_softmax.shape)
#                 #x_mean    = x_one_hot.mean(dim=0, keepdims=True)
#                 x_mean    = x_softmax.mean(dim=0, keepdims=True)
#                 #print('4',x_mean.shape)
#                 umap = torch.distributions.Categorical(x_mean.permute(0,2,3,1)).entropy()
#                 #print('5',umap.shape)
#                 #umap      = - x_mean * torch.log(x_mean)
#                 #umap      = umap.sum(dim=1, keepdims=True)

#             elif self.net_out == 'calgary':
#                 x_probs = torch.sigmoid(x[1:])
#                 x_mean  = x_probs.mean(dim=0, keepdims=True)
#                 umap    = - x_mean * torch.log(x_mean) - (1-x_mean) * torch.log(1-x_mean)
                
#         elif self.method == 'probs':
#             if self.net_out == 'mms':
#                 x_probs = F.softmax(x, dim=1)
#                 umap = torch.distributions.Categorical(x_probs.permute(0,2,3,1)).entropy()
#                 #umap    = - x_probs * torch.log(x_probs)
#                 #umap    = umap.sum(dim=1, keepdims=True)
                
#             elif self.net_out == 'calgary':
#                 x_probs = torch.sigmoid(x)
#                 #print(x_probs.min(), x_probs.max())
#                 #umap = torch.distributions.Categorical(x_probs.permute(0,2,3,1)).entropy()
#                 umap    = - x_probs * torch.log(x_probs+1e-6) - (1-x_probs) * torch.log(1-x_probs+1e-6)
        
        assert umap.shape[1] == 1, f"umap shape is {umap.shape}, but should be (n, 1, h, w)"
        return umap
    
    
    
class Metrics(object):
    """
    Object to collect, post-process and log metrics w.r.t. thresholded uncertainty maps.
    
    From binary uncertainty maps and ground truth segmentation, this object calculates
    an adaption of accuracy, precision and recall for uncertainty evaluation. This is used
    mostly to plot Precision-Recall curves, which are the main tool to assess uncertainty
    maps from different origins. All metrics and the plots can then be logged to WandB.
    """

    def __init__(
        self, 
        n_taus
    ):
        self.taus = torch.linspace(0, 1, n_taus).numpy()
        self.reset()
        
    @torch.no_grad()
    def reset(self) -> None:
        self.mse  = 0.
        self.acc  = 0.
        self.rec  = 0.
        self.pre  = 0.
        self.error_rate = 0.
        
        self.tp = 0.
        self.tpfp = 0.
        self.tpfn = 0.
        
    @torch.no_grad()    
    def scale(self, factor: float) -> None:
        self.mse /= factor
        self.acc /= factor
        #self.rec /= factor
        #self.pre /= factor
        self.error_rate /= factor
        
        self.pre = self.tp / self.tpfp
        self.rec = self.tp / self.tpfn
    
    @torch.no_grad()
    def update(self, binary_umaps: Tensor, errmap: Tensor, output=None) -> None:
        if output is not None:
            self.mse += ((output[:1] - output[1:])**2).mean()
            
        self.acc += self._get_accuracy(binary_umaps, errmap)
        #self.rec += self._get_recall(binary_umaps, errmap)
        #self.pre += self._get_precision(binary_umaps, errmap)
        self.error_rate += errmap.sum() / errmap.size(-1)**2

        tp, tpfp, tpfn = self._get_stats(binary_umaps, errmap)
        self.tp += tp
        self.tpfp += tpfp
        self.tpfn += tpfn
        
    @torch.no_grad()   
    def summary_stats(self) -> None:
        self.auc_acc = self.acc.mean()
        self.auc_rec = self.rec.mean()
        self.auc_pre = self.pre.mean()
        self.auc_pr  = auc(self.rec, self.pre) #torch.abs((self.pre[1:] * torch.diff(self.rec, 1))).sum()
        
    @torch.no_grad()    
    def _get_accuracy(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
    
        with torch.no_grad():
            t = (binary_umaps == errmap).sum(dim=(0, 2, 3))
            return t / binary_umaps.size(-1)**2
        
    @torch.no_grad()    
    def _get_recall(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
    
        with torch.no_grad():
            tp = (binary_umaps * errmap).sum(dim=(0, 2, 3))
            return tp / (errmap == 1).sum().clamp(1)
        
    @torch.no_grad()
    def _get_precision(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
    
        with torch.no_grad():
            tp = (binary_umaps * errmap).sum(dim=(0, 2, 3))
            return tp / (binary_umaps == 1).sum(dim=(0, 2, 3)).clamp(1)
        
    @torch.no_grad()    
    def _get_stats(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
        
        with torch.no_grad():
            tp   = (binary_umaps * errmap).sum(dim=(0, 2, 3))
            tpfp = (binary_umaps == 1).sum(dim=(0, 2, 3)).clamp(1)
            tpfn = (errmap == 1).sum().clamp(1)
            
            return tp, tpfp, tpfn
    
    @torch.no_grad()
    def log(self):
        data  = [[x, y] for (x, y) in zip(self.rec, self.pre)]
        table = wandb.Table(data=data, columns = ["recall", "precision"])
        xs    = self.taus
        ys    = [self.acc, self.rec, self.pre]
        keys  = ["accuracy", "recall", "precision"]

        wandb.log({
            "pr_auc" :    wandb.plot.line(table, "recall", "precision", 
                                          title="PR curve"),
            "acc_all":    wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
                                                 title="ROC curves", xname='tau'),
            'AUC ACC':    self.auc_acc,
            'AUC REC':    self.auc_rec, 
            'AUC PRE':    self.auc_pre,
            'AUC PR' :    self.auc_pr,
            'Error Rate': self.error_rate
        })

        
        
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=7):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

            
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        
        return False


    
def volume_collate(batch: List[dict]) -> dict:
    return batch[0]



def _activate_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
    
    
    
def reject_randomness(manualSeed):
    np.random.seed(manualSeed)
    random.rand.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None



def epoch_average(losses, counts):
    losses_np = np.array(losses)
    counts_np = np.array(counts)
    weighted_losses = losses_np * counts_np
    
    return weighted_losses.sum()/counts_np.sum()



def average_metrics(metric_list, counts):
    metrics = dict(metric_list[0])
    counts_np = np.array(counts)
    
    metrics[list(metrics.keys())[0]] = 0
    for layer in list(metrics.keys())[1:]:
        metrics[layer] = dict.fromkeys(metrics[layer], 0)
        
        for metric, count in zip(metric_list, counts_np):
            
            metrics[list(metrics.keys())[0]] += metric[list(metrics.keys())[0]] * count
            for m in metric[layer]:
                metrics[layer][m] += metric[layer][m] * count
        
        metrics[list(metrics.keys())[0]] /= counts_np.sum()
        for m in metrics[layer]:
            metrics[layer][m] /= counts_np.sum()
            
    return metrics
            
    
        
def slice_selection(dataset: Dataset, indices: Tensor, n_cases: int = 10) -> Tensor:
    
    slices = dataset.__getitem__(indices)['input']
    kmeans_in = slices.reshape(len(indices), -1)
    kmeans = KMeans(n_clusters=n_cases).fit(kmeans_in)
    idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, kmeans_in)
    return indices[idx]



def dataset_from_indices(dataset: Dataset, indices: Tensor) -> DataLoader:
    data = dataset.__getitem__(indices)
    
    class CustomDataset(Dataset):
        
        def __init__(self, input: Tensor, labels: Tensor, 
                     voxel_dim: Tensor):
            self.input = input
            self.labels = labels
            self.voxel_dim = voxel_dim
            
        def __getitem__(self, idx):
            return {'input': self.input[idx],
                    'target': self.labels[idx],
                    'voxel_dim': self.voxel_dim[idx]}
        
        def __len__(self):
            return self.input.size(0)
        
    return CustomDataset(*data.values())


 
def load_state_dict_for_modulelists(model, state_dict):

    seg_model_dict = model.seg_model.state_dict()
    seg_model_dict_pretrained = {
        k.replace('seg_model.', ''): v for k, v in state_dict.items() if k.replace('seg_model.', '') in seg_model_dict
    }
    model.seg_model.load_state_dict(seg_model_dict_pretrained)

    counter = 0
    for i in range(4):
        transformation_state_dict = model.transformations[i].state_dict()
        for j in range(4):
            try:
                transformation_state_dict_pretrained = {
                    k.replace(f'transformations.{j}.', ''): v for k, v in state_dict.items() if k.replace(f'transformations.{j}.', '') in transformation_state_dict
                }
                model.transformations[i].load_state_dict(transformation_state_dict_pretrained)
                counter += 1
            except:
                pass
    if counter == 4:
        print('All transformations loaded')
    else:
        sys.exit()

    return model
