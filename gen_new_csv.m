%script to add new features and save to csv
%add paths
addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\optimization');
addpath('functions\reg_functions');

path = 'training.csv';
M = csvread(path,1,0);
Event_id = M(:,1);
Weigts = M(:,end-1);
Output = M(:,end);

X_train = M(:,2:end-2);

% Replace -999.0 with 1000000000.
fprintf('Replacing -999.0 with 0...\n');
X_train(X_train == -999.0) = 1000000000;

% Adding combinational features.
fprintf('Adding combinational features...\n');
X_train =  add_features(X_train);

%  Replace values with -999
fprintf('Replacing values with -999.0 ...\n');
X_train(X_train < -1000000 || X_train > 1000000) = -999;

% Applying pca algorithm.
fprintf('Running pca algorithm...\n');
[X_train, dummy] = apply_pca(X_train, 0, 1);
k = size(X_train,2); % Number of features that X_train has been reduced to.

% Writing new csv
X_train_new = [Event_idm, X_train, Weigts, Output];
