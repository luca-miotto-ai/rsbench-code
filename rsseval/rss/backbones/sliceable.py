import torch
from collections import OrderedDict
from typing import Optional, Union

IndexLike = Union[str, int]

class SliceableModule(torch.nn.Module):
    """A module that can be sliced into sub-networks by layer names or indices."""
    
    def __init__(self, layers: OrderedDict[str, torch.nn.Module]):
        super().__init__()
        if not isinstance(layers, OrderedDict):
            raise TypeError("layers must be an OrderedDict[name -> nn.Module]")
        self._layer_sequence = list(layers.keys())
        for layer in layers.items():
            name, module = layer
            setattr(self, name, module)

    def forward(self, x):
        for name in self._layer_sequence:
            x = getattr(self, name)(x)
        return x

    def _resolve(self, key: IndexLike, *, allow_end: bool = False) -> int:
        names = self._layer_sequence
        if isinstance(key, str):
            if key not in names:
                raise ValueError(f"Layer name '{key}' not found. Available: {names}")
            return names.index(key)
        elif isinstance(key, int):
            n = len(names)
            idx = key if key >= 0 else n + key
            upper = n if allow_end else n - 1
            if not (0 <= idx <= upper):
                raise IndexError(f"Index {key} out of range [0,{upper}]")
            return idx
        else:
            raise TypeError("Slice keys must be str (name) or int (index)")

    def slice(self, from_: IndexLike = None, to_: IndexLike = None) -> torch.nn.Module:
        """
        Return a sub-network as `SliceableModule` itself.

        * from_: return sub-network starting AFTER `from_` (excluded) to the end.
        * to_:   return sub-network from the beginning UP TO `to_` (included).
        * both:  return sub-network BETWEEN them (from_ excluded, to_ included).

        `from_` and `to_` can be a layer NAME (str) or an INDEX (int, supports negatives).
        
        At least one of them must be provided; if both, they cannot refer to same layer.
        """
        names = self._layer_sequence
        n = len(names)
        
        if from_ is None and to_ is None:
            raise ValueError("Provide at least one slicing point")

        start = -1 if from_ is None else self._resolve(from_)
        end = n - 1 if to_ is None else self._resolve(to_)

        if start == end:
            raise ValueError("Invalid slice range: start must differ from end")
        if start > end:
            raise ValueError(f"Invalid slice range: start exceeds end")

        selected = OrderedDict(
            (name, getattr(self, name)) 
            for name in names[start + 1 : end + 1]
        )
        
        if len(selected) == 0:
            raise ValueError("Invalid slice range: slice would be empty")
        
        return SliceableModule(selected)
