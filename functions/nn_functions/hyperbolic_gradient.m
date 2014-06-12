function g = hyperbolic_gradient(z)

g = zeros(size(z));

g = ((2*1.7159/3) .* (1-tanh((2.*z)./3).^2)) ;

end