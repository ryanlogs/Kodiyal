%Neural networks with boosting to build NN ensemble
	function ensemble = nn_boosting(experts, network, X_train, Y_train, lambda)
	% The first cell in each row stores the theta 
	% and the second value stores the alpha for each NN expert
	ensemble = cell(experts,2);

	% set initial weights to 1/Num_examples
	m = size(X_train,1);
	weights = ones(m,1) ./ m;

	
	num_layers = size(network,1);
	options = optimset('MaxIter', 100);
	
	fprintf('\nIterations | Accuracy\n');
	% start training experts
	for i = 1:experts
		% Setting initialize network parameters i.e. thetas or weights.
		initial_nn_params = [];
		for j = 1:num_layers-1
			parm = rand_initialize_weights(network(j),network(j+1));
			initial_nn_params = [ initial_nn_params; parm(:)];
		end
		
		% training the ith expert
		cost_function = @(p) nn_ensemble_cost_function(p, network, X_train, Y_train, lambda, weights);
		[nn_params, cost] = fmincg(cost_function, initial_nn_params, options);
		
		% unrolling theta	
		Theta = cell(num_layers-1,1);
		read = 0;
		for j = 1:num_layers - 1
			Theta{j} = reshape(nn_params(read + 1: read + network(j+1) * (network(j) + 1)), ...
							network(j+1), network(j)+1);
								
			read = 	read + network(j+1) * (network(j) + 1);
		end
		
		% calculate error rate 
		pred = predict(Theta,X_train);
		error = sum(weights(pred~=Y_train));
		
		% update weights
		weights(pred~=Y_train) = weights(pred~=Y_train) ./ (2*error);
		weights(pred==Y_train) = weights(pred==Y_train) ./ (2*(1-error));
		
		% calculate alpha
		alpha = log((1-error)/error) / 2;
		if(alpha < 0)
			alpha = 0;
		end		
	
		ensemble{i,1} = Theta;
		ensemble{i,2} = alpha;
			
		
		% calculate predictions and accuracy
		hypothesis = predict_ensemble(ensemble,X_train,i);
		accuracy = mean(double(hypothesis == Y_train)) * 100;
%		accuracy = 1-sum(hypothesis~=Y_train)/m;
		
		% print results
		a = mean(double(pred == Y_train)) * 100;
		fprintf('%10d | %5f | %f %f %f\n', i, accuracy, alpha, error, a);
		plot(i,accuracy);
	end
	
end