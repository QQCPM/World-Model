#!/usr/bin/env python3
"""
Test script for Enhanced Structure Learner
Sets up proper Python path and tests Phase 2 enhancements
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_architectures.enhanced_structure_learner import (
    EnhancedCausalStructureLearner,
    create_enhanced_structure_learner,
    test_enhanced_structure_learner
)

if __name__ == "__main__":
    print("🔥 TESTING ENHANCED STRUCTURE LEARNER (PHASE 2)")
    print("=" * 60)

    success = test_enhanced_structure_learner()

    if success:
        print("\n✅ ENHANCED STRUCTURE LEARNER READY FOR PHASE 2 INTEGRATION!")
    else:
        print("\n❌ Enhanced structure learner needs debugging")