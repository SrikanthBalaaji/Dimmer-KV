"""DimmerKV: per-token-precision KV cache compression for chat and stored conversations."""

from .codec import DimmerKVCodec
from .engine import CompressionSettings, DimmerKV
from .rope import RoPE
from .store import ConversationStore, Match

__all__ = ["CompressionSettings", "ConversationStore", "DimmerKV", "DimmerKVCodec", "Match", "RoPE"]
__version__ = "0.1.0"
