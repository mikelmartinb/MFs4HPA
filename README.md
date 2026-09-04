# MFs4HPA
This repository contains the code necessary to reproduce the results in https://arxiv.org/abs/2603.22449 

Minkowski Functionals are computed with ``pynkowski`` 

## Installation

Clone the repository:

```bash
git clone https://github.com/mikelmartinb/MFs4HPA.git
cd MFs4HPA
```

Then install the required Python packages:

```bash
pip install -r requirements.txt
```

### LaTeX

Some figures use Matplotlib's LaTeX rendering through:

```python
plt.rcParams['text.usetex'] = True
```

Therefore, reproducing these figures requires a working LaTeX installation. Alternatively, LaTeX rendering can be disabled by setting:

```python
plt.rcParams['text.usetex'] = False
```

## Citation

If you make use of the software or analysis in this repository, please consider citing the relevant papers and software.

### Associated paper

[More Than Power: Revisiting the CMB Hemispherical Power Asymmetry with Morphological Descriptors](https://arxiv.org/abs/2603.22449)

```bibtex
@article{CarronDuque:2026unb,
    author = "Carr{\'o}n Duque, Javier and Martin Barandiaran, Mikel and Mart{\'\i}nez-Arrizabalaga, Joseba",
    title = "{More Than Power: Revisiting the CMB Hemispherical Power Asymmetry with Morphological Descriptors}",
    eprint = "2603.22449",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "IFT-UAM/CSIC-26-036",
    month = "3",
    year = "2026"
}
```

### `pynkowski`

The Minkowski Functionals in this repository are computed using [`pynkowski`](https://github.com/javicarron/pynkowski). If you use this software, please also cite the original `pynkowski` repository.



