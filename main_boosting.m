%implementation of Boosting with NN

%add paths
addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\optimization');

% Load all data in the form of .mat files.
fprintf('Loading data from .mat files...\n');
%load('data\train\train.mat');
%load('data\train\cv.mat');
%load('data\test\test.mat');
load('circle.mat');

% Normalize the data.
fprintf('Normalizing data...\n');
X_train = normalize_range(X_train, -1, 1);
%X_cv = normalize_range(X_cv, -1, 1);
%X_test = normalize_range(X_test, -1, 1);

% Define the network size and parameters here.
fprintf('Initializing the Network...\n');

network = [size(X_train,2);  100; 100; 2];

num_layers = size(network,1);

% Define the value of the bias factor lambda 'lm'
lm = 1.2;
lambda = ones(num_layers-1,1).*lm;

% set number of experts
experts = 200;

ensemble = nn_boosting( experts, network, X_train, Y_train, lambda);

hold off;