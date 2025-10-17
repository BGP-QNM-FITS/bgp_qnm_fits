## Overview

This package implements Bayesian likelihood fits of quasinormal modes (QNMs) to Cauchy-characteristic evolved (CCE) numerical relativity (NR) data, using a Gaussian process model for the NR noise. In principle the functions in this package can be generalised to other NR data. 

This repo accompanies the paper [Quasinormal modes from numerical relativity with Bayesian inference (Dyer & Moore, submitted)](https://arxiv.org/pdf/2510.11783). For further code that uses this package, see [BGP methods](https://github.com/Richardvnd/bgp_methods). 

## Installation

```bash
# Clone the repository
git clone https://github.com/username/bgp_qnm_fits.git

# Move to folder containing .toml file
cd bgp_qnm_fits

# Install the package (editable version)
pip install -e .

```

## Usage

```python
import bgp_qnm_fits as bgp

```

For example usage see demo.ipynb

## Requirements

- NumPy
- SciPy
- JAX
- [qnmfits](https://github.com/sxs-collaboration/qnmfits)

## Authors

- Richard Dyer (richard.dyer@ast.cam.ac.uk)
- Christopher Moore (christopher.moore@ast.cam.ac.uk)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{dyer2025quasinormalmodesnumericalrelativity,
      title={Quasinormal modes from numerical relativity with Bayesian inference}, 
      author={Richard Dyer and Christopher J. Moore},
      year={2025},
      eprint={2510.11783},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2510.11783}, 
}
```