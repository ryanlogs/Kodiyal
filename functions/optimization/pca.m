function [Z, U] =  pca(X_norm,flag,features)

	m = size(X_norm,1);
	n = size(X_norm,2);
	% first calculate the covariance matrix;
	Sigma = (X_norm' * X_norm) ./ m;

	[ U, S, V] = svd(Sigma);	

	% choose the least k, from n features such information is preserved
	diagonal = diag(S);
	
	if(flag==0)
		k = 0;
		rating = 0;
		
		while (rating < 0.99 && n ~= k)
			k = k + 1;
			rating = sum(diagonal(1:k)) ./ sum(diagonal);
			disp(sprintf('Rating for %d features = %f',k,rating));	
		end
	
		disp(sprintf('\nInput reduced to %d features...\n',k));
	else
		k = features;
	end	
	
	Z = X_norm * U(:,1:k);

end 