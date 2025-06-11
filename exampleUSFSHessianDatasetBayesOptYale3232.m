addpath('datasets/');
clear all;  
dataset = 'Yale_32x32';
load(dataset);  % Loads variables X and Y

fea = fea;
gnd = gnd;

% Parameters
m = 50;
n_components = 10;
[n_samples, n_features] = size(fea);
W_init = randn(n_features, n_components);
F_init = orth(randn(n_components, n_components));
G_init = orth(randn(n_samples, n_components));

maxIter = 50;
innerIter = 5;
k = 10;  % Number of nearest neighbors

TanParam = struct();
TanParam.DimGiven = 1;           
TanParam.NCoordDim = 5;          
TanParam.EValueTolerance = 0.95; 

X = fea;

% Define ranges for parameters (simulate Bayesian optimization by random search)
alphaRange = [1e-3, 10];
lambdaRange = [1e-3, 10];
gammaRange = [1e-3, 10];
muRange = [1e-3, 10];

numTrials = 20; % Number of random samples

best_obj = Inf;
best_params = struct('alpha',[],'lambda',[],'gamma',[],'mu',[]);

for trial = 1:numTrials
    % Log-scale sampling for parameters (similar idea to bayesopt’s transforms)
    alpha_val = 10^(log10(alphaRange(1)) + rand*(log10(alphaRange(2))-log10(alphaRange(1))));
    lambda_val = 10^(log10(lambdaRange(1)) + rand*(log10(lambdaRange(2))-log10(lambdaRange(1))));
    gamma_val = 10^(log10(gammaRange(1)) + rand*(log10(gammaRange(2))-log10(gammaRange(1))));
    mu_val = 10^(log10(muRange(1)) + rand*(log10(muRange(2))-log10(muRange(1))));

    [~, ~, obj_values_tmp] = USFS_Hessian(X, m, W_init, F_init, G_init, ...
        alpha_val, lambda_val, gamma_val, mu_val, maxIter, innerIter, k, TanParam);

    final_obj = obj_values_tmp(end);
    if final_obj < best_obj
        best_obj = final_obj;
        best_params.alpha = alpha_val;
        best_params.lambda = lambda_val;
        best_params.gamma = gamma_val;
        best_params.mu = mu_val;
    end

    fprintf('Trial %d/%d: alpha=%.4f lambda=%.4f gamma=%.4f mu=%.4f FinalObj=%.4f\n', ...
        trial, numTrials, alpha_val, lambda_val, gamma_val, mu_val, final_obj);
end

disp('Best parameters found by random search (heuristic):');
disp(best_params);
disp(['Best final objective value: ', num2str(best_obj)]);

[feature_idx, W, obj_values] = USFS_Hessian(X, m, W_init, F_init, G_init, ...
    best_params.alpha, best_params.lambda, best_params.gamma, best_params.mu, maxIter, innerIter, k, TanParam);

selected_features = feature_idx;

nKmeans = 15;
[nSmp,mFea] = size(fea);
nClass = length(unique(gnd));
label = litekmeans(fea, nClass, 'Replicates', 15);
MIhat = NMI_sqrt_lei(gnd, label);
disp(['Clustering using all the ', num2str(size(fea,2)), ' features. Clustering NMI: ', num2str(MIhat)]);

disp('Selected Feature Indices:');
disp(feature_idx);
save('selected_features.mat', 'feature_idx');

PlotExampleUSFSHessianDataset;
