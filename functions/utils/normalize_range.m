%scales the features between the given range
function [A_norm] = normalize_range(A, r_min, r_max)

	ratio = (r_max - r_min) ./ (max(A) - min(A));
 
	m = size(A,1);
	% divide by ratio
	A_norm = A - (ones(m,1) * min(A));
    A_norm = A_norm .* (ones(m,1) * ratio);
    
	% addition rmin
	A_norm = A_norm +   r_min;

end
