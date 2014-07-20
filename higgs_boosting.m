%implementation of Boosting with NN

%add paths
addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\optimization');


% Load all data in the form of .mat files.
fprintf('Loading data from .mat files...\n');
load('data\train\train.mat');
load('data\train\cv.mat');
load('data\test\test.mat');
%load('circle.mat');

% Normalize the data.
fprintf('Normalizing data...\n');
X_train = normalize_range(X_train, -1, 1);
X_cv = normalize_range(X_cv, -1, 1);
X_test = normalize_range(X_test, -1, 1);

% Define the network size and parameters here.
fprintf('Initializing the Network...\n');
network = [size(X_train,2);  100; 100; 2];
num_layers = size(network,1);

% Define the value of the bias factor lambda 'lm'
lm = 1.2;
lambda = ones(num_layers-1,1).*lm;

% set number of experts
experts = 50;

ensemble = nn_boosting( experts, network, X_train, Y_train, lambda);

fprintf("Trainig done!!!\n");
% calculate predictions and accuracy
fprintf('Predicting Training Accuracy...\n');
hypothesis = predict_ensemble(ensemble,X_train,experts);
train_acc = mean(double(hypothesis == Y_train)) * 100;
fprintf('\nTraining Accuracy: %f \n', train_acc);	

% calculate predictions and accuracy
fprintf('Predicting CV Accuracy...\n');
hypothesis = predict_ensemble(ensemble,X_cv,experts);
cv_acc = mean(double(hypothesis == Y_cv)) * 100;
fprintf('\nCross Validation Accuracy: %f \n', cv_acc

%using Theta to predict Test output and rank
fprintf('Predicting Test Output...\n');
pred = predict_ensemble(ensemble,X_test,experts);
rank = get_rank_ensemble(ensemble,X_test,experts);

%Preparing Output file
disp('Writing Test Output... \n');
save_name = sprintf('output\\%s_result%s.csv','Project_kodiyal',datestr(clock,'HH_MM_DDDD_mmmm_YYYY'))

%writing the headers first
out_id = fopen(save_name,'w+');
fprintf(out_id,'%s\n','EventId,RankOrder,Class');
out = [test_id, rank, pred];
%dlmwrite (save_name, out, '-append','delimiter',',');
fprintf(out_id,'%d,%d,%d\n',out');
fclose(out_id);

fprintf('Done!!!\n');


