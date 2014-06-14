% Assume that generate data function has been run.
% Load all data in the form of .mat files.

load('data\train\train.mat');
load('data\train\cv.mat');
load('data\test\test.mat');

% Replace -999.0 with 0.

X_train(X_train = -999.0) = 0;
X_cv(X_cv = -999.0) = 0;
X_test(X_test = -999.0) = 0;

% Define the network size and parameters here.
network = [size(X_train,2), 50, 50, 2];
iter = 1500;

% Define the value of the bias factor lambda 'lm'
lm = 1.2;

num_layers = size(network,1);
lamba = ones(num_layers-1,1).*lm;

% Setting initialize network parameters i.e. thetas or weights.
initial_nn_params = [];
for i = 1:num_layers-1
    parm = randInitializeWeights(network(i),network(i+1));
    initial_nn_params = [ initial_nn_params; parm(:)];
end

options = optimset('MaxIter', iter);

% Training NN.
cost_function = @(p) nn_cost_function(p, network, X_train, Y_train, 0, lambda);
[nn_params, cost] = fmincg(cost_function, initial_nn_params, options);

%unrolling theta	
Theta = cell(num_layers-1,1);
read = 0;
for i = 1:num_layers - 1
	Theta{i} = reshape(nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
					network(i+1), network(i)+1);
						
	read = 	read + network(i+1) * (network(i) + 1);
end

