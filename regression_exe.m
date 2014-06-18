addpath('functions\utils');
addpath('functions\nn_functions');
addpath('functions\reg_functions');


x = (1:100)';
Y = sin(x);
m = size(x,1);
X = [ones(m,1), x, x.^2];
n = size(X,2)-1;
lambda = 1.2;

initial_theta = rand_initialize_weights(n,1)';
options = optimset('MaxIter', 100);
cost_function = @(p) reg_cost_function(p, X, Y, lambda);
[theta, cost] = fmincg(cost_function, initial_theta, options);

pred = X * theta;

figure();
hold on;
scatter(x,Y);
plot(x,pred,'red');
hold off;