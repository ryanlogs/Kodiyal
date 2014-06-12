function g = hyperbolic(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = (1.7159 .* tanh((2.*z)./3));
end
