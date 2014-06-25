%script to add new features and save to csv
%add paths
addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\optimization');
addpath('functions\reg_functions');

path = 'data\train\training.csv';
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
X_train(X_train < -1000000) = -999;
X_train(X_train > 1000000) = -999;

% Applying pca algorithm.
fprintf('Running pca algorithm...\n');
[X_train, dummy] = apply_pca(X_train, 0, 1);
k = size(X_train,2); % Number of features that X_train has been reduced to.

% Writing new csv
X_train_new = [Event_id, X_train, Weigts, Output];
out_id = fopen('data\train\new_training.csv','w+');
fprintf(out_id,'%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n',X_train_new');
fclose(out_id);

clear all;
k=24

path = 'data\test\test.csv';
M = csvread(path,1,0);
Event_id = M(:,1);

X_test = M(:,2:end);

% Replace -999.0 with 1000000000.
fprintf('Replacing -999.0 with 0...\n');
X_test(X_train == -999.0) = 1000000000;

% Adding combinational features.
fprintf('Adding combinational features...\n');
X_test =  add_features(X_test);

%  Replace values with -999
fprintf('Replacing values with -999.0 ...\n');
X_test(X_test < -1000000) = -999;
X_test(X_test > 1000000) = -999;

% Applying pca algorithm.
fprintf('Running pca algorithm...\n');
[X_test, dummy] = apply_pca(X_test, 1, k);

% Writing new csv
X_test_new = [Event_id, X_test];
out_id = fopen('data\test\new_test.csv','w+');
fprintf(out_id,'%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n',X_test_new');
fclose(out_id);


