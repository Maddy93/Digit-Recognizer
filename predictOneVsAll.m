function p = predictOneVsAll(all_theta, X)
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. 

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Multiplies the each row of input matrix with all the classifiers. It then
% takes the maximum value, as that is the closest prediction, for each row 
% and returns that as the predicted digit
[s,p]=max(X*all_theta',[],2);


end
