function [X] = recover(Z, U)

	n = size(U,1);
	k = size(Z,2);
	
	X = Z * U(:,1:k)'; 
end