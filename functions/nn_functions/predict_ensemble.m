function hypothesis=predict_ensemble(ensemlble,X,num_experts)
	hypothesis = zeros(size(X,1),1);
	
	for i = 1:num_experts
		pred = predict(ensemlble{i,1},X);
		pred(pred==0) = -1;
		hypothesis = hypothesis + ensemlble{i,2} .* pred;
	end
	
	hypothesis(hypothesis>=0) = 1;
	hypothesis(hypothesis<0) = 0;
end