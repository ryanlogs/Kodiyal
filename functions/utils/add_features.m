% Takes a matrix with primary features
% adds all combinations of x1.x2 and x1/x2 
% and returns a matrix with secondary features
function [X_secondary] = add_features(X_primary)
	
	%first add all features of X_primary
	X_secondary = X_primary;
	
	%get all combinations i,j
	combos = combntns(1:size(X_primary,2),2);
	
	%add product and division
	for i = 1:size(combos,1)
		i = combos(i,1);
		j = combos(i,2);
		%X_secondary = [ X_secondary, X_primary(:,i).*X_primary(:,j), X_primary(:,i)./X_primary(:,j) ];
		X_secondary = [ X_secondary, X_primary(:,i).*X_primary(:,j) ];
	end
end
 