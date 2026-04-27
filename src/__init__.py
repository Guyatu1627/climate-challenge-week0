"""
EthioClimate Analytics Engine - Core Processing Module

This package contains the core data processing and analysis functions
for the Climate Analytics Engine for COP32.

Author: Climate Analytics Team
Purpose: Evidence-based climate policy analysis
"""

__version__ = "1.0.0"
__author__ = "Climate Analytics Team"

from .processor import ClimateDataProcessor
from .analyzer import ClimateAnalyzer

__all__ = ['ClimateDataProcessor', 'ClimateAnalyzer']
