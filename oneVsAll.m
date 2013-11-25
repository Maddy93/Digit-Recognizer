function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% ONEVSALL trains multiple logistic regression classifiers and returns all
% the classifiers in a matrix all_theta, where the i-th row of all_theta 
% corresponds to the classifier for label i


m = size(X, 1);
n = size(X, 2);
j=0;

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
initial_theta=zeros(n+1,1);

% For all labels(1 to 10), we call fmincg(used to return the optimim parameters)
for c=1:1:num_labels
	options = optimset('GradObj', 'on', 'MaxIter', 100);
	[all_theta(c,:),j] = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                 initial_theta, options);
end

end
