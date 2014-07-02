function p = predict(Theta, X)
	%PREDICT Predict the label of an input given a trained neural network
	%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	%   trained weights of a neural network (Theta1, Theta2)

	% Useful values
	m = size(X, 1);

	% You need to return the following variables correctly 
	p = zeros(size(X, 1), 1);

	H = X;
	for i = 1:size(Theta)
		H = hyperbolic([ones(m,1) H] * (Theta{i})');
	end

	p(H>=0) = 1;
	p(H<0)  = -1;

end
