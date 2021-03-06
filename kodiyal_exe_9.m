% Assume that generate data function has been run.
% General Regression Neural Network .

%add paths
addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\optimization');
addpath('functions\reg_functions');

% % Load all data in the form of .mat files.
fprintf('Loading data from .mat files...\n');
load('data\train\train.mat');
load('data\train\cv.mat');
load('data\test\test.mat');

% Replace -999.0 with 0.
fprintf('Replacing -999.0 with 0...\n');
X_train(X_train == -999.0) = 0;
X_cv(X_cv == -999.0) = 0;
X_test(X_test == -999.0) = 0;

% Normalize the data.
fprintf('Normalizing data...\n');
X_train = normalize_range(X_train, -1, 1);
X_cv = normalize_range(X_cv, -1, 1);
X_test = normalize_range(X_test, -1, 1);

% Define the network size and parameters here.
fprintf('Initializing the Network...\n');
network = [size(X_train,2); 50; 50; 50; 1];

% Define the value of the bias factor lambda 'lm'
lm = 1.1;
iter = 2400;

num_layers = size(network,1);
lambda = ones(num_layers-1,1).*lm;

% Setting initialize network parameters i.e. thetas or weights.
initial_nn_params = [];
for i = 1:num_layers-1
    parm = rand_initialize_weights(network(i),network(i+1));
    initial_nn_params = [ initial_nn_params; parm(:)];
end

options = optimset('MaxIter', iter);

% Training NN.
fprintf('Training the Network...\n');
cost_function = @(p) nn_reg_cost_function(p, network, X_train, normalize_range(W_train,-1,1), lambda);
[nn_params, cost] = fmincg(cost_function, initial_nn_params, options);

%unrolling theta	
Theta = cell(num_layers-1,1);
read = 0;
for i = 1:num_layers - 1
	Theta{i} = reshape(nn_params(read + 1: read + network(i+1) * (network(i) + 1)), ...
					network(i+1), network(i)+1);
						
	read = 	read + network(i+1) * (network(i) + 1);
end

fprintf('Training Completed.\n');

%Assuming 1 is signal and 0 is background noise
%running prediction function
fprintf('Network ready to predict...\n');

%using Theta to predict Training output
pred = predict_reg(Theta,X_train);
pred = normalize_range(pred,min(W_train),max(W_train));

train_acc = coeff_of_deter(pred, W_train);		
fprintf('\nTraining Accuracy: %f |\tlambda: %f\n', train_acc, i);	


%using Theta to predict Cross Validation output
pred = predict_reg(Theta,X_cv);
pred = normalize_range(pred,min(W_train),max(W_train));
cv_acc = coeff_of_deter(pred, W_cv);		
fprintf('\nCV Accuracy: %f |\tlambda: %f\n', cv_acc, i);

%using Theta to predict Test output and rank
fprintf('Predicting Test Output...\n');
pred = predict(Theta,X_test);
W_test = normalize_range(pred,min(W_train),max(W_train));

fprintf('Save W_test Output...\n');
save('data\test\test_v2.mat','X_test','W_test');
% % %Preparing Output file
% disp('Writing Test Output... \n');
% save_name = sprintf('output\\%s_result%s.csv','Project_kodiyal',datestr(clock,'HH_MM_DDDD_mmmm_YYYY'));

% % %writing the headers first
% out_id = fopen(save_name,'w+');
% fprintf(out_id,'%s\n','EventId,RankOrder,Class');
% out = [test_id, rank, pred];
% %dlmwrite (save_name, out, '-append','delimiter',',');
% fprintf(out_id,'%d,%d,%d\n',out');
% fclose(out_id);


fprintf('Done!!!\n');

