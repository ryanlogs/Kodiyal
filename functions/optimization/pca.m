function [Z, U] =  pca(X_norm)

	m = size(X_norm,1);
	n = size(X_norm,2);
	% first calculate the covariance matrix;
	Sigma = (X_norm' * X_norm) ./ m;

	[ U, S, V] = svd(Sigma);	

	% choose the least k, from n features such information is preserved
	k = 1;
	rating = 0;
	diagonal = diag(S);
	
	while rating < 0.99
		rating = sum(diagonal(1:k)) ./ sum(diagonal);
		disp(sprintf('Rating for %d features = %f',k,rating));
		k++;
	end
	
	Z = X_norm * U(:,1:k);

end 