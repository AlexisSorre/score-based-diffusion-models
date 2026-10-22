# Score Estimator Comparison: Memorization and Computation Time

This notebook (`comparaison-estimateur.ipynb`) compares several score estimators for diffusion models, following the setup of Stuart et al. It measures how much each estimator **memorizes** the training data and how long it takes to **train and sample**.

## Estimators compared

1. **Empirical score** (explicit, exact minimizer of the empirical loss)
2. **Ridge-regularized empirical score** (Tikhonov, $c = 10^{-2}$)
3. **Neural network score** (ReLU MLP, width 256, random Fourier time features, EMA weights)
4. **Regularized neural network** (trained with a Tikhonov-penalized loss)
5. **Leave-One-Out cross-validation score** (explicit), in two implementations:
   - naive, $O(N^2)$ per evaluation
   - optimized, $O(N)$, with a numerically stable fix for catastrophic cancellation at small $\sigma$
6. **K-fold cross-validation score** ($K = 5$), $O(KN)$

An extra section reproduces the effect of network **width** (8, 32, 128) and **number of training epochs** (5k, 20k, 50k) on memorization.

## Experimental setup

- Data: $N = 20$ i.i.d. points from $\mathcal N(0, I_2)$
- Forward process: VE SDE with $g(t) = 10t$, $T = 1$, so $\sigma(t) = \sqrt5\, t$
- Sampling: deterministic reverse ODE, solved with fixed-step RK4 (400 steps) after the change of variable $\sigma = e^{-s}$, down to $\sigma_{\min} = 10^{-3}$
- 500 generated samples, all estimators share the same initial conditions $x(T)$
- **Memorization rate**: fraction of samples within distance $\tau$ of a training point, reported for $\tau = 10^{-2}$ and $\tau = 10^{-1}$

## Outputs

- Voronoi diagram of the dataset, with sample trajectories for each estimator
- Training loss curves for the neural networks
- A summary table with training time, generation time, mean/max distance to the data, and memorization rates

## Requirements

```
numpy pandas scipy matplotlib torch
```

## Notes

- The notebook text and comments are in French.

