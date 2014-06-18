% Assume that generate data function has been run.
% Major change in this variant is the addition of combinational features.
% Stochastic implementation.

%add paths
addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\optimization');

% Load all data in the form of .mat files.
fprintf('Loading data from .mat files...\n');
load('data\train\train.mat');
load('data\train\cv.mat');
load('data\test\test.mat');

% Normalize the data.
fprintf('Normalizing data...\n');
X_train = normalize_std(X_train);
X_cv = normalize_std(X_cv);
X_test = normalize_std(X_test);

% Initial setting of theta (weights).
m = size(X_train,1);
n = size(X_train,2);
lambda = 1.2;

initial_theta = rand_initialize_weights(n,1)';

% Cost function.
options = optimset('MaxIter', 100);
cost_function = @(p) reg_cost_function(p , [ones(m,1), X_train], W_train, lambda);
[theta, cost] = fmincg(cost_function, initial_theta, options);

pred = [ones(m,1), X_train] * theta;


