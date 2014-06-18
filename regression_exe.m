addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\reg_functions');


X = 1:100;
Y = 4.* X.^2;
m = size(X,1);

initial_theta = rand_initialize_weights(1,1);

options = optimset('MaxIter', 100);
cost_function = @(p) reg_cost_function(p, [ones(m,1), X], Y, lambda);
[theta, cost] = fmincg(cost_function, initial_theta, options);

pred = [ones(m,1), X] * theta;

figure();
hold on;
scatter(X,Y);
plot(X,pred,'red');
hold off;