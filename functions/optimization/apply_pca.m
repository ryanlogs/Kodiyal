function [Z, U] = apply_pca(X)

	%first normalize
	[X_norm, mean, sigma] = feature_normalize(X);
	
	%apply PCA
	[Z, U] = pca(X_norm);
	
	%give the difference between X_norm and X_approx
	X_approx = recover(Z, U);
	m = size(Z,1) * size(Z,2);
	diff = abs(X_norm - X_approx);
	disp(sprintf('\nDifference between original and approx data = %f\n', sum(diff(:))/m));
end