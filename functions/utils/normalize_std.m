%returns a normalized version of X where
%the mean value of each feature is 0 and the standard deviation
%is 1. Use this as PCA input
function [A_norm] = normalize_std(A)

	%subtract by the mean
	mu = mean(A);
	A_norm = bsxfun(@minus, A, mu);

	%divide by the standard deviation
	sigma = std(A_norm);
	A_norm = bsxfun(@rdivide, A_norm, sigma);

end