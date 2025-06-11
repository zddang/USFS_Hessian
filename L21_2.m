function a = L21_2(X)
% L212 computes the L_{2,1-2} norm of matrix X
%
% Input:
% X: a (d x n) matrix
%
% Output:
% a = ||X||_{2,1-2} = ||X||_{2,1} - ||X||_{2,2}
%
%   - ||X||_{2,1} = sum_j ||X(:,j)||_2
%   - ||X||_{2,2} = ||X||_F = sqrt(sum_{i,j} X(i,j)^2)

% Compute L_{2,1}-norm: sum of column norms
colNorms = sqrt(sum(X.^2, 1));
L21 = sum(colNorms);

% Compute L_{2,2}-norm: Frobenius norm
L22 = norm(X, 'fro');

% Compute L_{2,1-2} norm
a = L21 - L22;

end
