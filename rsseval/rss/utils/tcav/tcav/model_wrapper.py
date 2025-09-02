from copy import deepcopy
from torch.autograd import grad
import torch
import numpy as np
from typing import Callable


class ModelWrapper(object):
    def __init__(self, model, layers, is_boia):
        self.model = deepcopy(model)
        self.intermediate_activations = {}
        self.is_boia = is_boia

        def save_activation(name):
            """create specific hook by module name"""

            def hook(module, input, output):
                self.intermediate_activations[name] = output

            return hook

        for name, module in self.model._modules.items():
            if name in layers:
                # register the hook
                module.register_forward_hook(save_activation(name))

        if hasattr(self.model, "encoder"):
            for name, module in self.model.encoder._modules.items():
                if name in layers:
                    # register the hook
                    module.register_forward_hook(save_activation(name))

        if hasattr(self.model, "net"):
            for name, module in self.model.net._modules.items():
                if name in layers:
                    # register the hook
                    module.register_forward_hook(save_activation(name))

    def save_gradient(self, grad):
        self.gradients = grad

    def generate_gradients(self, c, layer_name):
        activation = self.intermediate_activations[layer_name]
        activation.register_hook(self.save_gradient)
        if self.is_boia:
            gradients = None
            for i in range(0, self.output.shape[1], 2):
                logit = self.output[:, i]
                logit.backward(torch.ones_like(logit), retain_graph=True)
                if gradients is None:
                    gradients = self.gradients.cpu().detach().numpy()
                else:
                    gradients += self.gradients.cpu().detach().numpy()
        else:
            logit = self.output[:, c]
            logit.backward(torch.ones_like(logit), retain_graph=True)
            gradients = self.gradients.cpu().detach().numpy()
        return gradients

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        # if hasattr(self.model, 'to'):
        #     self.model = self.model.to(device)
        return self

    def __call__(self, x):
        self.output = self.model(x)["YS"]
        return self.output

    def get_downstream_head(self, layer) -> Callable:
        """
        Return the downstream head of the model starting from the given layer.

        **NOTE:** given layer not included in the downstream head.
        """
        found = False
        modules = []
        if hasattr(self.model, "encoder"):
            for name, module in self.model.encoder._modules.items():
                if found:
                    modules.append(module)
                if name == layer:
                    found = True
        elif hasattr(self.model, "net"):
            for name, module in self.model.net._modules.items():
                if found:
                    modules.append(module)
                if name == layer:
                    found = True
        else:
            for name, module in self.model._modules.items():
                if found:
                    modules.append(module)
                if name == layer:
                    found = True
        return torch.nn.Sequential(*modules)