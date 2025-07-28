# Installation Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for accelerated simulations

## Basic Installation
```bash
pip install sapiens-sim
```

## Development Installation
```bash
git clone https://github.com/yourusername/sapiens-sim.git
cd sapiens-sim
pip install -e .[dev]
```

## Dependencies
Core dependencies:
- numpy>=1.20.0
- numba>=0.58.1
- neat-python>=0.92
- pandas>=1.3.0

Development dependencies:
- pytest>=6.0.0
- black>=21.0
- mypy>=0.900

## Configuration
1. Copy `config.example.py` to `config.py`
2. Adjust parameters as needed
3. Run validation:
```python
from sapiens_sim.config import validate_config
validate_config()
```

## Troubleshooting
Common issues and solutions:
1. Numba compilation errors
   - Ensure LLVM is properly installed
   - Update to latest Numba version

2. Performance issues
   - Enable CUDA if available
   - Adjust OPTIMIZATION_THRESHOLDS in config
