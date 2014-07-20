function rank = get_rank(ensemlble,X,num_experts)
	%PREDICT Predict the label of an input given a trained neural network
	%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	%   trained weights of a neural network (Theta1, Theta2)

	% Useful values
	m = size(X, 1);
	p = zeros(size(X, 1), 1);
	
	% You need to return the following variables correctly 
	rank = zeros(size(X, 1), 1);

	H = zeros(size(X,1),1);
	for i = 1:num_experts
		pred = predict(ensemlble{i,1},X);
		pred(pred==0) = -1;
		H = H + ensemlble{i,2} .* pred;
	end

	p = H;
	
	% adding numbers to sort by index, to retrieve original ordering
	p = [ (1:m)', p]; 
	
	%sort based on confidence
	p = sortrows(p,2);
	
	% adding rank
	p = [ p, (1:m)']; 	
	
	%resort based on original order
	p = sortrows(p,1);	
	rank = p(:,3);
end