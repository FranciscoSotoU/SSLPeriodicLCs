"""
TimeHandlerParallel Module - Main Interface

This module provides imports for the modular time handling and encoding components.
All classes have been reorganized into separate modules for better maintainability.
"""

# Import handlers
from .handlers.time_handler import TimeHandler
from .handlers.time_handler_parallel import TimeHandlerParallel

# Import encoders
from .encoders.time_film import TimeFilm
from .encoders.positional_encoder import PositionalEncoder
from .encoders.positional_encoder_claude import PositionalEncoderClaude
from .encoders.positional_encoder_enhanced import PositionalEncoderEnhanced

# For backward compatibility, expose all classes at module level
__all__ = [
    'TimeHandler',
    'TimeHandlerParallel',
    'TimeFilm',
    'PositionalEncoder',
    'PositionalEncoderClaude',
    'PositionalEncoderEnhanced'
]