#!/usr/bin/env python3
"""
Test script for Enhanced Dual Pathway GRU
Sets up proper Python path and imports
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced model
from causal_architectures.enhanced_dual_pathway_gru import (
    EnhancedDualPathwayCausalGRU,
    create_enhanced_dual_pathway_model,
    test_enhanced_model_compatibility
)

if __name__ == "__main__":
    print("🧪 TESTING ENHANCED DUAL PATHWAY MODEL")
    print("=" * 50)

    success = test_enhanced_model_compatibility()

    if success:
        print("\n✅ ENHANCED MODEL READY FOR PHASE 1 INTEGRATION!")
    else:
        print("\n❌ Enhanced model needs debugging")