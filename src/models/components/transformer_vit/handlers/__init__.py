"""
Time Handlers Module

This module contains time handling classes for processing time series data
with different strategies and encoder configurations.
"""

from .time_handler import TimeHandler
from .time_handler_parallel import TimeHandlerParallel

__all__ = [
    'TimeHandler',
    'TimeHandlerParallel'
]
