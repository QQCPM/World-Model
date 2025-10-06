# Archived Files

This directory contains old/superseded files that were moved during the Phase 1 cleanup to maintain a clean codebase.

## Directory Structure

### `old_scripts/`
Numbered pipeline scripts that were superseded by the new Phase 1 joint training system:
- `01_generate_causal_data.py` - Old data generation script
- `05_train_continuous_models.py` - Old individual model training (superseded by joint training)
- `10_causal_intervention_testing.py` - Old intervention testing (superseded by validation framework)
- `11_production_inference_server.py` - Old production server (Phase 4 will implement new version)
- `12_complete_system_demo.py` - Old system demo (Phase 4 will implement new version)

### `old_implementations/`
Previous implementation attempts that have been superseded:
- `causal_rnn/` - Old causal RNN implementation (replaced by dual-pathway GRU)
- `causal_intervention_tester.py` - Old intervention testing (superseded by CausalReasonerTester)
- `severe_causal_validation.py` - Old validation script (superseded by validation framework)
- `live_monitor.py` - Old training monitoring script
- `production/` - Empty old production directory (Phase 4 will recreate)

### `old_experiments/`
Old experimental scripts and utilities:
- `experiments/` - Old experimental framework

### `old_utils/`
Previous utility functions:
- `utils/` - Old utility functions (Phase 1 implements new utilities in core modules)

### `old_models/`
Pre-Phase 1 trained model checkpoints:
- Various `.pth` files from individual model training experiments

### `old_results/`
Previous experimental results and logs:
- `results/` - Old training results and evaluation reports
- `data/` - Old datasets and generated data
- `analysis/` - Previous analysis results
- `logs/` - Old training logs
- `navigation_training.log` - Old navigation training log

### `old_documentation/`
Previous documentation that became outdated after Phase 1:
- Various `.md` files documenting old approaches and interim findings

### `shell_scripts/`
Old shell scripts for automation:
- `quick_check.sh` - Old system check script
- `quick_start.sh` - Old startup script

## Current Active System

The current active system uses only these components:
- `causal_architectures/` - Phase 1 dual-pathway architecture
- `training/` - Phase 1 joint training system
- `validation/` - Phase 1 comprehensive validation framework
- `causal_envs/` - Continuous campus environment
- `continuous_models/` - Baseline models for comparison
- `CAUSAL_ARCHITECTURE_IMPLEMENTATION_PLAN.md` - Current implementation plan
- `README.md` - Current project overview

## Archive Date
Files archived: September 16, 2025
Phase 1 implementation completed:

## Important Notes
- These files are preserved for reference but should not be used
- The Phase 1 system replaces all functionality from archived files
- If you need to reference old approaches, check the documentation in `old_documentation/`
- Do not mix archived imports with current Phase 1 code