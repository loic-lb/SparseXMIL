import torch
import torch.nn as nn
from models.model_xmil import XMIL
from models.model_sparseconvmil import SparseConvMIL
from models.model_average import AverageMIL
from models.model_attention import GatedAttention
from models.model_transmil import TransMIL
from models.model_dgcn import DGCNMIL


class ModelEmaV2(nn.Module):
    def __init__(self, model, model_name, perf_aug, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = self.get_module(model, model_name, perf_aug)
        self.module.load_state_dict(model.state_dict())
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    @staticmethod
    def get_module(model, model_type, perf_aug):
        """
        Returns the model to use for the EMA
        @param model: model used for training (to get the parameters)
        @param model_type: name of the model to use for the EMA
        @param perf_aug: boolean indicating if test-time augmentation is used
        @return: a function that returns the model to use for the EMA
        """
        if model_type == "average":
            model_fct = AverageMIL(model.nb_layers_in, model.n_classes)
        elif model_type == "attention":
            model_fct = GatedAttention(model.nb_layers_in, model.n_classes)
        elif model_type == "transmil":
            model_fct = TransMIL(model.n_classes, model.transmil_size)
        elif model_type == "dgcn":
            model_fct = DGCNMIL(num_features=model.num_features, n_classes=model.n_classes)
        elif model_type == "sparseconvmil":
            model_fct = SparseConvMIL(model.nb_layers_in, sparse_map_downsample=model.sparse_map_downsample,
                                      perf_aug=perf_aug, num_classes=model.num_classes)
        else:
            model_fct = XMIL(nb_layers_in=model.nb_layers_in, sparse_map_downsample=model.sparse_map_downsample,
                             D=model.D, num_classes=model.num_classes, perf_aug=perf_aug)
        return model_fct

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
