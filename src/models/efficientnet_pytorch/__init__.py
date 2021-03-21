# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/__init__.py
__version__ = "0.7.0"
from src.models.efficientnet_pytorch.model import EfficientNet, VALID_MODELS
from src.models.efficientnet_pytorch.utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)
