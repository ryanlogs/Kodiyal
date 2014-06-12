%error function for nnThetaCostFunction
function [y] = error_function(output,teacher)
	%output = output;
	%y =  teacher .* (2/3) .* (1.7159 - output.^2) .* (teacher - output);
	y = output.^2 ./ 2 - output .* (teacher) + 0.5;
end 