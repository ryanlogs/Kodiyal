% Applies PCA  on input features
% use flag=0 if you want to find optimal features
% use flag=1 and features to any number u want the result to reduce to

function [Z, U] = apply_pca(X,flag,features)

	%first normalize
	[X_norm, mean, sigma] = feature_normalize(X);
	 
	X_norm(isinf(X_norm)) = 0;
	X_norm(isnan(X_norm)) = 0; 
	
	%apply PCA
	[Z, U] = pca(X_norm, flag, features);
	
	%give the difference between X_norm and X_approx
	X_approx = recover(Z, U);
	m = size(Z,1) * size(Z,2);
	diff = abs(X_norm - X_approx);
	disp(sprintf('\nDifference between original and approx data = %f\n', sum(diff(:))/m));
end