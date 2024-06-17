import torch
import torch.nn as nn
import numpy as np
import pickle
from torch import Tensor
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import IncrementalPCA, PCA
from copy import deepcopy
from typing import Tuple, Callable

# Possible layers
# Probably bottleneck on encoder
# model.1.submodule.1.submodule.1.submodule.0.conv.unit3.conv
# model.1.submodule.1.submodule.1.submodule.0.conv.unit
# Probably bottleneck on decoder
# model.1.submodule.1.submodule.2.0.conv
# model.1.submodule.1.submodule.2.0

class PoolingMahalanobisDetector(nn.Module):
    def __init__(
        self,
        swivel: str,
        pool: str = 'avg2d',
        sigma_algorithm: str = 'default',
        sigma_diag_eps: float = 2e-1,
        transform: bool = False,
        dist_fn: str = 'squared_mahalanobis',
        lr: float = 1e-3,
        device: str  = 'cuda:0'
    ):
        super().__init__()
        # init args
        self.swivel = swivel
        self.pool = pool
        self.sigma_algorithm = sigma_algorithm
        self.sigma_diag_eps = sigma_diag_eps
        # self.hook_fn = self.register_forward_pre_hook if hook_fn == 'pre' else self.register_forward_hook
        self.transform = transform
        self.dist_fn = dist_fn
        self.lr = lr
        self.device = device
        # other attributes
        self.active = True
        self.training_representations = []
        # methods
        if self.pool == 'avg2d':
            self._pool = nn.AvgPool2d(
                kernel_size=(2,2), 
                stride=(2,2)
            )
        elif self.pool == 'avg3d':
            self._pool = nn.AvgPool3d(
                kernel_size=(2,2,2),
                stride=(2,2,2)
            )
        elif self.pool == 'none':
            self._pool = None
        else:
            raise NotImplementedError('Choose from: avg2d, avg3d, none')
        
        self.to(device)


    ### private methods ###

    def _reduce(self, x: Tensor) -> Tensor:
        if 'avg' in self.pool:
            # reduce dimensionality with 3D pooling to below 1e4 entries
            while torch.prod(torch.tensor(x.shape[1:])) > 1e4:
                x = self._pool(x)
            x = self._pool(x)
        elif self.pool == 'none':
            pass
        # reshape to (batch_size, 1, n_features)
        x = x.reshape(x.shape[0], 1, -1)
        return x


    @torch.no_grad()
    def _collect(self, x: Tensor) -> None:
        # reduces dimensionality as per self._pool, moves to cpu and stores
        x = self._reduce(x.detach()).cpu()
        self.training_representations.append(x)


    @torch.no_grad()
    def _merge(self) -> None:
        # concatenate batches from training data
        self.training_representations = torch.cat(
            self.training_representations,
            dim=0
        )


    @torch.no_grad()
    def _estimate_gaussians(self) -> None:
        self.register_buffer(
            'mu',
            self.training_representations.mean(0, keepdims=True).detach().to(self.device)
        )
        
        if self.sigma_algorithm == 'diagonal':
            self.register_buffer(
                'var',
                torch.var(self.training_representations.squeeze(1), dim=0).detach()
            )
            sigma = torch.sqrt(self.var)
            sigma = torch.max(sigma, torch.tensor(self.sigma_diag_eps))
            self.register_buffer(
                'sigma_inv', 
                1 / sigma.detach().to(self.device)
            )

            # self.sigma_inv = torch.max(self.sigma_inv, torch.tensor(self.sigma_diag_eps).to(self.device))
        
        elif self.sigma_algorithm == 'default':
            assert self.pool in ['avg2d', 'avg3d'], 'default only works with actual pooling, otherwise calculation sigma is infeasible'

            self.register_buffer(
                'sigma',
                torch.cov(self.training_representations.squeeze(1).T)
            )
            self.register_buffer(
                'sigma_inv', 
                torch.linalg.inv(self.sigma).detach().unsqueeze(0).to(self.device)
            )

        elif self.sigma_algorithm == 'ledoitWolf':
            assert self.pool in ['avg2d', 'avg3d'], 'default only works with actual pooling, otherwise calculation sigma is infeasible'

            self.register_buffer(
                'sigma', 
                torch.from_numpy(
                    LedoitWolf().fit(
                        self.training_representations.squeeze(1)
                    ).covariance_
                )
            )
            self.register_buffer(
                'sigma_inv', 
                torch.linalg.inv(self.sigma).detach().unsqueeze(0).to(self.device)
            )

        else:
            raise NotImplementedError('Choose from: lediotWolf, diagonal, default')



    def _distance(self, x: Tensor) -> Tensor:
        assert self.sigma_inv is not None, 'fit the model first'
        x_reduced  = self._reduce(x)
        x_centered = x_reduced - self.mu
        if self.sigma_algorithm == 'diagonal':
            dist = x_centered**2 * self.sigma_inv
            dist = dist.sum((1,2))
        else:
            dist = x_centered @ self.sigma_inv @ x_centered.permute(0,2,1)
            
        assert len(dist.shape) == 1, 'distances should be 1D over batch dimension'
        assert dist.shape[0] == x.shape[0], 'distance and input should have same batch size'

        if self.dist_fn == 'squared_mahalanobis':
            return dist
        elif self.dist_fn == 'mahalanobis':
            return torch.sqrt(dist)
        else:
            raise NotImplementedError('Choose from: squared_mahalanobis, mahalanobis')


    ### public methods ###

    def fit(self):
        self._merge()
        self._estimate_gaussians()
        # del self.training_representations


    def on(self):
        self.active = True


    def off(self):
        self.active = False


    def forward(self, x: Tensor) -> Tensor:
        if self.active:
            if self.training:
                self._collect(x)
            
            else:
                self.batch_distances = self._distance(x).detach().view(-1)
                if self.transform:
                    x = x.clone().detach().requires_grad_(True)
                    dist = self._distance(x).sum()
                    dist.backward()
                    x.grad.data = torch.nan_to_num(x.grad.data, nan=0.0)
                    x.data.sub_(self.lr * x.grad.data)
                    x.grad.data.zero_()
                    x.requires_grad = False
        else:
            pass
        return x


class PoolingMahalanobisWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        adapters: nn.ModuleList,
        copy: bool = True,
        sequential: bool = False
    ):
        super().__init__()
        self.model           = deepcopy(model) if copy else model
        self.adapters        = adapters
        self.sequential      = sequential
        self.adapter_handles = {}
        self.transform       = True
        self.model.eval()

    def hook_adapters(
        self,
    ) -> None:
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer  = self.model.get_submodule(swivel)
            hook   = self._get_hook(adapter)
            self.adapter_handles[
                swivel
            ] = layer.register_forward_pre_hook(hook)

    def _get_hook(
        self,
        adapter: nn.Module
    ) -> Callable:
        def hook_fn(
            module: nn.Module, 
            x: Tuple[Tensor]
        ) -> Tensor:
            # x, *_ = x # tuple, alternatively use x_in = x[0]
            # x = adapter(x)
            return adapter(x[0])
        
        return hook_fn

    def fit(self):
        for adapter in self.adapters:
            adapter.fit()

    def set_transform(self, transform: bool):
        self.transform = transform
        for adapter in self.adapters:
            adapter.transform = transform

    def set_lr(self, lr: float):
        for adapter in self.adapters:
            adapter.lr = lr

    def turn_off_all_adapters(self):
        for adapter in self.adapters:
            adapter.off()

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        if self.sequential and self.transform and not self.training:
            out = []
            for adapter in self.adapters:
                self.turn_off_all_adapters()
                adapter.on()
                out.append(self.model(x).detach().cpu())
            return out

        else:
            return self.model(x).detach().cpu()

class PCA_Adapter(nn.Module):
    def __init__(self, swivel, n_dims, batch_size, pre_fit=False, name='', device='cuda:0'):
        super().__init__()
        self.swivel = swivel
        self.n_dims = n_dims
        self.bs = batch_size
        # self.pca = PCA(n_components=self.n_dims)
        self.device = device
        self.pre_fit = pre_fit
        self.project = name
        self.pca_path = f'/workspace/src/out/pca/{name}'
        self.activations = []
        self._set_pca()

    def _set_pca(self):
        if self.training is True:
            print('Instantiated new IPCA')
            self.pca = IncrementalPCA(n_components=self.n_dims, batch_size=self.bs)
        else:
            print('Loading trained IPCA')
            self.pca_path += f'_{self.swivel.replace(".", "_")}'
            self.pca_path += f'_{self.n_dims}dim.pkl'
            print(f'Loading path{self.pca_path}')
            with open(self.pca_path, 'rb') as f:
                self.pca = pickle.load(f)

    def forward(self, x):
        # X must be of shape (n_samples, n_features), thus flattened, and comes as a torch tensor
        x = x.view(x.size(0), -1)
        x_np = x.detach().cpu().numpy()
        if self.pre_fit is False:
            self.pca.partial_fit(x_np)
        elif self.pre_fit is True:
            x_np = self.dim_reduce(x_np)
            self.activations.append(x_np)

    def _collect(self, x):
        # Collect activations, probably assert x type and shape
        ...

    def _merge_save(self):
        # Probably concat activations
        print('Saving activations...')
        save_path = f'/workspace/src/out/activations/{self.project}_{self.swivel.replace(".", "_")}_activations_{self.n_dims}dims.npy'
        np.save(save_path, np.vstack(self.activations))

    def dim_reduce(self, x):
        # Probably assert x shape and type
        return self.pca.transform(x)

class PCAModuleWrapper(nn.Module):
    def __init__(self, model, adapters, copy=True):
        super().__init__()
        self.model = deepcopy(model) if copy else model
        self.adapters = adapters
        self.adapter_handles = {}
        self.model.eval()

    def hook_adapters(self):
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer  = self.model.get_submodule(swivel)
            hook   = self._get_hook(adapter)
            self.adapter_handles[swivel] = layer.register_forward_pre_hook(hook)

    def _get_hook( self, adapter):
        def hook_fn( module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            adapter(x[0])
            # return adapter(x)
        return hook_fn

    def forward(self, x):
        return self.model(x)
