function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
%	lambda is a parameter for regularization. Regularization is used 
%	to keep the values of theta small so that we do not overshoot the
%	minimum while going down the slope.
	
% Initialize some useful values
m = length(y); % number of training examples

J = 0;%cost
grad = zeros(size(theta));%gradient


J=-((y'*log(sigmoid(X*theta)))+((1-y')*log(1-sigmoid(X*theta))))/m;
J=J+lambda*(sum(theta([2:size(theta)],:).^2))/(2.0*m);

grad=(((sigmoid(X*theta)-y)'*X))'/m;
grad([2:size(grad)],:)=grad([2:size(grad)],:)+theta([2:size(grad)],:)*lambda/m;

end
