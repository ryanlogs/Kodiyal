function [J grad] = nn_adv_cost_function(nn_params, X, y, lambda)

	
	addpath('functions\nn_functions');
	addpath('functions\utils');
	m = size(X,1);	
	n = size(X,2);
	
	% compute cost
	H = X * nn_params;
	J = sum((H-y).^2) /(2*m) - sum(nn_params.^2) * lambda;
	
	% gradient
	temp = sum(((H-y) * ones(1,n)) .* X) / m;
	temp(:,2:end) = temp(:,2:end) + nn_params' .* (lambda/m)
	
	grad = temp';
end
