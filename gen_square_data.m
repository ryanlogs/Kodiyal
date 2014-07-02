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

% Adding combinational features.
fprintf('Adding combinational features...\n');
X_train =  [X_train X_train.^2];

% Writing new csv
X_train_new = [Event_id, X_train, Weigts, Output];
out_id = fopen('data\train\new_training.csv','w+');
fprintf(out_id,'%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n',X_train_new');
fclose(out_id);

clear all;
k=24;

path = 'data\test\test.csv';
M = csvread(path,1,0);
Event_id = M(:,1);

X_test = M(:,2:end);

% Adding combinational features.
fprintf('Adding combinational features...\n');
X_test =  [X_test X_test.^2];

% Writing new csv
X_test_new = [Event_id, X_test];
out_id = fopen('data\test\new_test.csv','w+');
fprintf(out_id,'%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n',X_test_new');
fclose(out_id);



