function [r2] = coeff_of_deter(X, Y)
	
	m = size(X,1);
	
	X_mean = mean(X);
	Y_mean = mean(Y);
	
	X_std = std(X);
	Y_std = std(Y);
	
	r2 = (sum((X - X_mean) .* (Y - Y_mean)) ./ m ./(X_std*Y_std))^2;
end