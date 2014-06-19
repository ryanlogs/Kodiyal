% Assume that generate data function has been run.
% Major change in this variant is the addition of combinational features.
% Stochastic implementation.

%add paths
addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\reg_functions');
addpath('functions\optimization');

% Load all data in the form of .mat files.
fprintf('Loading data from .mat files...\n');
load('data\train\train.mat');
load('data\train\cv.mat');
load('data\test\test.mat');

% Replace -999.0 with 0.
fprintf('Replacing -999.0 with 0...\n');
X_train(X_train == -999.0) = 0;
X_cv(X_cv == -999.0) = 0;
X_test(X_test == -999.0) = 0;

% Adding square and cubic features.
% X_train = [X_train, X_train.^2];
% X_cv = [X_cv, X_cv.^2];
% X_test = [X_test, X_test.^2];

% Saving square features
X_train_sq = X_train.^2;
X_cv_sq = X_cv.^2;
X_test_sq = X_test.^2;

% Adding combinational features.
fprintf('Adding combinational features...\n');
X_train =  [ add_features(X_train), X_train_sq ];
X_cv = [ add_features(X_cv), X_cv_sq ];
X_test = [ add_features(X_test), X_test_sq ] ;

% Applying pca algorithm.
fprintf('Running pca algorithm...\n');
[X_train, dummy] = apply_pca(X_train, 0, 1);
k = size(X_train,2); % Number of features that X_train has been reduced to.
[X_cv, dummy] = apply_pca(X_cv, 1, k);
[X_test, dummy] = apply_pca(X_test, 1, k);

% Normalize the data.
fprintf('Normalizing data...\n');
X_train = normalize_std(X_train);
X_cv = normalize_std(X_cv);
X_test = normalize_std(X_test);

X_train = [X_train, X_train.^2, X_train.^3];

% Initial setting of theta (weights).
m = size(X_train,1);
n = size(X_train,2);
lambda = 0.2;

initial_theta = rand_initialize_weights(n,1)';

% Cost function.
options = optimset('MaxIter', 5000); %enter iterations here
cost_function = @(p) reg_cost_function(p , [ones(m,1), X_train], W_train, lambda);
[theta, cost] = fmincg(cost_function, initial_theta, options);

pred = [ones(m,1), X_train] * theta;

r = coeff_of_deter(pred,W_train)

