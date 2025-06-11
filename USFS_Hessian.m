function [feature_idx, W, obj_values] = USFS_Hessian(X, m, W_init, F_init, G_init, alpha, lambda, gamma, mu, maxIter, innerIter, k, TanParam)
% USFS_Hessian: Unsupervised Feature Selection with Hessian Regularization 
% and L_{2,1-2}-norm Sparsity Regularization using the Augmented Lagrangian
%
% Inputs:
%   - X: Data matrix (n_samples x n_features)
%   - m: Number of selected features
%   - W_init, F_init, G_init: Initial matrices (W, F, G)
%   - alpha, lambda, gamma, mu: Hyperparameters
%   - maxIter: Maximum number of outer iterations
%   - innerIter: Number of inner iterations for G and K updates
%   - k: Number of nearest neighbors for Hessian computation
%   - TanParam: Tangent space dimensionality
%
% Outputs:
%   - feature_idx: Indices of selected features
%   - W: Learned projection matrix
%   - obj_values: Objective function values over iterations

%% Initialization
[n_samples, n_features] = size(X);
n_components = size(F_init, 2);

% Center the data
X = X - mean(X, 1);

% Compute Hessian matrix H
XX = sum(X.^2, 2);                     
D_squared = bsxfun(@plus, XX, XX') - 2*(X*X'); 
D_squared(D_squared < 0) = 0;          
D = sqrt(D_squared);

[~, idx] = sort(D, 2);  
NNIdx = idx(:, 1:k);    
[H, ~] = ConstructHessian(X, NNIdx, TanParam);

% Initialize variables
W = W_init;
F = F_init;
G = G_init;
K = max(G, 0);
E = zeros(n_samples, n_components);
Y = zeros(n_samples, n_components);
obj_values = zeros(maxIter, 1);

% Precompute X^T X
XtX = X' * X;

% Initialize auxiliary variables for W and E
d_W = ones(n_features, 1);
d_E = ones(n_samples, 1);

%% Main Loop
for iter = 1:maxIter
    % Update G and K
    for inner = 1:innerIter
        % Update G
        A = mu * (X * W - E) * F - Y * F + 2 * gamma * K;
        B = mu * (F' * F) + 2 * gamma * eye(n_components);
        G = A / B;
        % Enforce orthogonality constraint G^T G = I via SVD
        [U_G, ~, V_G] = svd(G, 'econ');
        G = U_G * V_G';
        
        % Update K
        K = 0.5 * (G + abs(G));
    end

    % Update F
    M = G' * (X * W - E + Y / mu);
    [U_F, ~, V_F] = svd(M, 'econ');
    F = V_F * U_F';

    % Update W
    % Compute Q and scalar_c based on current W
    d_W = 1 ./ (2 * sqrt(sum(W.^2, 2) + eps));
    D_W = spdiags(d_W, 0, n_features, n_features);
    Q = D_W;
    %Compute the scalar for the L21_2-norm
    W_fro = norm(W, 'fro');
    scalar_c = lambda / (W_fro + eps);

    W_num = X' * (mu * (E - G * F' + Y / mu));
    W_den = mu * XtX + 2 * alpha * X' * H * X + lambda * Q + scalar_c * eye(n_features);

    W = W_den \ W_num;


    % Update E
    D_E = spdiags(d_E, 0, n_samples, n_samples);
    E = (mu * eye(n_samples) + 2 * D_E) \ (mu * (X * W - G * F' + Y / mu));

    % Update Y
    Residual = E - X * W + G * F';
    Y = Y + mu * Residual;

    % Update weights d_W and d_E for L_{2,1}-style updates
    d_W = 1 ./ (2 * sqrt(sum(W.^2, 2) + eps));
    d_E = 1 ./ (2 * sqrt(sum(E.^2, 2) + eps));

    % Compute objective function value (Augmented Lagrangian)
    % According to the given augmented objective:
    %
    % L(W,G,F,K,E) = ||E||_{2,1} + αTr(W^T X^T H X W) + λ||W||_{2,1-2} + γ||G-K||_F^2 
    %                + Tr[Y^T (E - XW + GF^T)] + (μ/2)||E - XW + GF^T||_F^2

    obj_values(iter) = sum(sqrt(sum(E.^2, 2))) ...                    % ||E||_{2,1}
                     + alpha * trace(W' * X' * H * X * W) ...         % αTr(W^T X^T H X W)
                     + lambda * L21_2(W) ...                           % λ||W||_{2,1-2}
                     + gamma * norm(G - K, 'fro')^2 ...               % γ||G - K||_F^2
                     + trace(Y' * (E - X * W + G * F')) ...           % Tr[Y^T (E - XW + GF^T)]
                     + (mu/2) * norm(E - X * W + G * F', 'fro')^2;    % (μ/2)||E - XW + GF^T||_F^2

    % Display progress (optional)
    fprintf('Iteration %d: Objective Value = %f\n', iter, obj_values(iter));
end

%% Select Features
[~, idx] = sort(sum(W.^2, 2), 'descend');
if nargin >= 2 && m > 0
    feature_idx = idx(1:m);
else
    feature_idx = idx;
end

end
