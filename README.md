# USFS\_Hessian

Unsupervised Feature Selection with Hessian Regularization (`USFS_Hessian`) is a MATLAB implementation of an **augmented‑Lagrangian optimisation** framework that jointly learns

- a sparse projection matrix **W**
- an orthogonal basis **F**
- a low‑dimensional embedding **G**

while selecting the top‑`m` informative features from high‑dimensional data.\
The method combines **manifold preservation** (via a Hessian energy term) with **mixed \$\ell\_{2,1\text{–}2}\$ sparsity** to encourage row‑wise and global shrinking of **W**.

---

## Key Ideas

1. **Hessian regularisation** captures second‑order manifold curvature, outperforming Laplacian‑based graph terms on data with non‑linear structure.
2. **\$\ell\_{2,1\text{–}2}\$ sparsity** simultaneously enforces row sparsity (feature selection) and Frobenius‑norm control (model compactness).
3. **Orthogonality constraints** on **F** and **G** promote stable, rotation‑invariant embeddings.
4. **Augmented Lagrangian** updates are applied in an outer–inner loop:\
   ‑ inner loop alternates **G** ↔ **K** (ReLU‑like split) to satisfy non‑negativity,\
   ‑ outer loop updates **F**, **W**, **E** (error), and the Lagrange multiplier **Y**.

---

## Function Signature

```matlab
[feature_idx, W, obj_values] = USFS_Hessian(X, m, W_init, F_init, G_init,
                                            alpha, lambda, gamma, mu,
                                            maxIter, innerIter, k, TanParam)
```

### Required Inputs

- **X** – *(n\_samples × n\_features)* data matrix (rows = samples, columns = features).
- **m** – number of features to retain (set `m = 0` to return a full ranking).
- **W\_init, F\_init, G\_init** – initial guesses for the projection, basis and embedding matrices. Random orthonormal initialisations work well.

### Hyper‑parameters

| symbol      | variable | role                                                                 |
| ----------- | -------- | -------------------------------------------------------------------- |
| \$\alpha\$  | `alpha`  | weight of Hessian term (manifold preservation)                       |
| \$\lambda\$ | `lambda` | weight of \$\ell\_{2,1\text{–}2}\$ sparsity on **W**                 |
| \$\gamma\$  | `gamma`  | weight of splitting term $\|G-K\|\_F^2\$                             |
| \$\mu\$     | `mu`     | augmented‑Lagrangian penalty (multiplied each outer loop if desired) |

Other controls:

- **maxIter** – outer iterations (default ≈ 100–200).
- **innerIter** – inner G/K repeats per outer step (default = 5).
- **k** – nearest neighbours for Hessian graph construction.
- **TanParam** – intrinsic tangent‑space dimensionality.

---

## Outputs

- **feature\_idx** – indices of the `m` selected features (descending importance).
- **W** – learned projection matrix (can be reused for out‑of‑sample data).
- **obj\_values** – objective value at each outer iteration (for convergence diagnostics).

---

## Quick Start

```matlab
% Load or create data X (samples × features)
X = randn(500, 1024);

% Desired dimensionality
n_components = 20;        % latent dimension
m_features   = 100;       % features to keep

% Random orthonormal inits
[U,~,~] = svd(randn(size(X,2), n_components), 'econ');  W0 = U;
[U,~,~] = svd(randn(size(X,1), n_components), 'econ');  F0 = U;
[U,~,~] = svd(randn(size(X,1), n_components), 'econ');  G0 = U;

% Hyper‑parameters
alpha  = 1e-2;   lambda = 1e-1;  gamma = 1;  mu = 1;
maxIter = 150;   innerIter = 5;   k = 10;    TanParam = 5;

[idx, W, obj] = USFS_Hessian(X, m_features, W0, F0, G0,
                             alpha, lambda, gamma, mu,
                             maxIter, innerIter, k, TanParam);

fprintf('Top‑5 features: %s\n', mat2str(idx(1:5)))
```

---

## Practical Tips

- **Scaling/centering**: the function internally centres X. Standardise features beforehand if scale varies widely.
- **Parameter tuning**: start with small \$\alpha\$ and moderate \$\lambda\$, then adjust to balance manifold fidelity vs. sparsity.
- **Convergence**: monitor `obj_values`. A plateau or relative change < 1e‑4 typically indicates convergence.
- **Large datasets**: computing the full Hessian graph scales with \$\mathcal{O}(n^2)\$. Consider approximate k‑NN or subsampling.
- **Out‑of‑sample projection**: project unseen data `X_new` via `X_new * W(:,1:m)`.

---

## File List

- `USFS_Hessian.m` – main algorithm.
- `ConstructHessian.m` – builds the Hessian energy matrix.
- `L21_2.m` – mixed \$\ell\_{2,1\text{–}2}\$ norm utility.
- `demo_usfs.m` – minimal reproducible example (optional).

---

## Citation

If you use this code, please cite:

> *Dang, Z., Jazayeri, A., Arino, R., Briskina, A., & Moslemi, A.* (2025). **Unsupervised Feature Selection Using Orthogonally Constrained Matrix Factorization with Hessian Regularization and Non-Convex Sparsity**. *Proceedings of the Canadian Conference on Artificial Intelligence*. Retrieved from https://caiac.pubpub.org/pub/et3rmuvx
---

## License

Distributed under the MIT License. See `LICENSE` for details.

