from .cnn1d import SimpleCNN1D
from .attention_cnn import AttentionCNN1D
from .tcn import TemporalConvNet
from .multiscale_cnn import MultiScaleCNN1D
from .lstm import BiLSTM, BiLSTMWithAttention
from .gru import BiGRU
from .cnn_lstm import CNNLSTM, CNNGRUWithAttention
from .resnet1d import ResNet1D
from .spectral_transformer import SpectralTransformer

__all__ = [
    'SimpleCNN1D',
    'AttentionCNN1D',
    'TemporalConvNet',
    'MultiScaleCNN1D',
    'BiLSTM',
    'BiLSTMWithAttention',
    'BiGRU',
    'CNNLSTM',
    'CNNGRUWithAttention',
    'ResNet1D',
    'SpectralTransformer',
    'register_model',
    'get_model',
]

# --- Custom model registry ---
_CUSTOM_REGISTRY: dict = {}


def register_model(name: str, cls: type) -> None:
    """Register a custom model class by name for use with spec_runner."""
    _CUSTOM_REGISTRY[name.lower()] = cls


def get_model(name: str):
    """Get a registered custom model class. Returns None if not found."""
    return _CUSTOM_REGISTRY.get(name.lower())