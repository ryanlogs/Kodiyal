% used for stochastic learning
function [Theta] = learn(	network, ...
							X, y, lambda, batch, batch_iter)
							
%function takes all the parameters and trains the network							

	addpath('function\nn_functions');
	
	num_layers = size(network,1);
	
	nn_params = [];
	for i = 1: num_layers-1
		parm = randInitializeWeights(network(i),network(i+1));
		nn_params = [ nn_params ; parm(:) ];
	end	
	
	fprintf('\n\nTraining Neural Network for digit %d\n',digit);
	options = optimset('MaxIter', batch_iter);
	
	%costFunction = @(p) nn_adv_cost_function(p, network, X,y, lambda);

	for iter = 0:batch:size(X,1)-batch;

		costFunction = @(p) nnCostFunction(p, network, X(iter+1:iter+batch,:),y(iter+1:iter+batch,:), lambda);
		[nn_params, cost] = fmincg(costFunction, nn_params, options);	
	%end	
	
	Theta = cell(num_layers-1,1);
	read = 0;
	for i = 1:num_layers - 1
		Theta{i} = reshape(nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
						network(i+1), network(i)+1);
						
		read = 	read + network(i+1) * (network(i) + 1);
	end	
	
end	