function rank = get_rank(Theta, X)
	%PREDICT Predict the label of an input given a trained neural network
	%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	%   trained weights of a neural network (Theta1, Theta2)

	% Useful values
	m = size(X, 1);
	p = zeros(size(X, 1), 1);
	
	% You need to return the following variables correctly 
	rank = zeros(size(X, 1), 1);

	
	H = X;
	for i = 1:size(Theta)
		H = hyperbolic([ones(m,1) H] * (Theta{i})');
	end

	% Get confidence values
	for i = 1:size(X,1)
		if(H(i,2) > H(i,1))
			p(i) = H(i,2);
		elseif(H(i,2) < H(i,1))
			p(i) = -1 * H(i,1);
		else 
			p(i) = 0;	
		end	
	end
	
	% adding numbers to sort by index, to retrieve original ordering
	p = [ (1:m)', p]; 
	
	%sort based on confidence
	p = sortrows(p,2);
	
	% adding rank
	p = [ p, (1:m);]; 	
	
	%resort based on original order
	p = sortrows(p,1);	
	rank = p(:,3);
end