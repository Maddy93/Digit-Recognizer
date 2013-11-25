%% Initialization
clear ; close all; clc

%% Setup of parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%  We start by first loading the dataset. 
%  Dataset that contains handwritten digits.

% Load Training and Testing Data
fprintf('Loading Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
Xtest = dlmread('testInput.csv');
ytest = dlmread('testOutput.csv');
m = size(X, 1);

%  logistic regression

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predictOneVsAll(all_theta, Xtest);
dlmwrite('predictedOutput.csv', pred);
%calculation of accuracy of classifier, the predicted 
%outputs(in pred) are compared with ytest, and 
%percentage of data that matched is printed
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

