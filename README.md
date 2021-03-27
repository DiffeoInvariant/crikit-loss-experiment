[![DOI](https://zenodo.org/badge/352093813.svg)](https://zenodo.org/badge/latestdoi/352093813)

This repository contains code comparing the quadratic Wasserstein distance to the regular squared `L^2` norm (`crikit.integral_loss`), where the quadratic Wasserstein distance is approximated with the sliced quadratic Wasserstein distance with 30 slices (`crikit.SlicedWassersteinDistance`). The problem is inferring the p-Stokes CR (from `examples/p-stokes/p-stokes.py`) where the learned CR is (line 324 of `loss-experiment.py`)
```
def cr_func(scalar_invts, theta):
    a, p = theta[0], theta[1]
    scale = a * (scalar_invts[1] + eps2) ** ((p - 2) / 2)
    return jnp.array([0, scale])
```
where `eps2` is a regularization constant and the true value of `theta` is `[1.0, 1.2]`. The mesh is a 5-by-5 unit square, and the problem is discretized with low-order Taylor-Hood elements (second-order in the velocity space and first-order in the pressure space). Optimization is done with `L-BFGS-B` with the following bounds for `theta`:
```
bounds = [array([0.9, 1.05]), array([1.05, 1.5])]
```
using the same 30 random seeds for each loss function and standard deviation (so all generated random numbers are exactly the same across loss functions in each iteration of the experiment). The noise model is additive, mean-zero, i.i.d. Normal distributions (`crikit.AdditiveRandomFunction`), and observations are taken on the outflow boundary with `crikit.SubdomainObserver`. You can generate the plots from this experiment (which were used in Zane Jakobs's MS thesis) with `make-plots.sh`. The experiment requires considerable computational resources to fully replicate all of the data in the `data` directory (given that it requires 60 optimizations per standard deviation), so we recommend generating the plots first and choosing one or a few standard deviations of interest and trying those.
    
