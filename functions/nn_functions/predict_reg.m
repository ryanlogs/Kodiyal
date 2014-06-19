function p = predict(Theta, X)
	%PREDICT Predict the label of an input given a trained neural network
	%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	%   trained weights of a neural network (Theta1, Theta2)

	% Useful values
	m = size(X, 1);

	% You need to return the following variables correctly 
	p = zeros(size(X, 1), 1);

	H = X;
	for i = 1:size(Theta)-1
		H = hyperbolic([ones(m,1) H] * (Theta{i})');
	end
	
	H = [ones(m,1) H] * (Theta{size(Theta,1)})';
	
	p = H;
	% [dummy, p] = max(H, [], 2);
	% p = p - 1; 

end
