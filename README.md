# MFs4HPA
This repository contains the code necessary to reproduce the results in https://arxiv.org/abs/2603.22449 

Minkowski Functionals are computed with ``pynkowski`` 

## Installation

Clone the repository:

```bash
git clone https://github.com/mikelmartinb/MFs4HPA.git
cd MFs4HPA
```
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
