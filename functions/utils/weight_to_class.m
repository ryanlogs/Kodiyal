function [Y] = weight_to_class(weight,lower_limit, range)
	% assigns class to weights
	m = size(weight,1);
	Y = zeros(m,1);
	for i = 1:m
		if(weight(i) > lower_limit)
			w = weight(i);
			% get the floor value and increment by range
			% class = w - mod(w,range) + range;
			class = ceil(w/range);			
			Y(i) = class;
		end	
	end
end