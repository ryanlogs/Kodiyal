%scales the features between the given range
function [A_norm] = normalise(A, r_min, r_max)
	ratio = (r_max - r_min) ./ (max(A) - min(A));
	
	% divide by ratio
	A_norm = bsxfun(@minus, A, min(A));
	A_norm = bsxfun(@times, A, ratio);
	
	% addition rmin
	A_norm = bsxfun(@plus, A_norm, r_min);

end
 