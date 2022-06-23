from typing import Set
import torch.nn as nn


def find_all_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        found_keys.update(find_all_keys(child, prefix=new_prefix))

    return found_keys
