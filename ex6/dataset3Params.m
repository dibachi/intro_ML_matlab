function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%from a number of options, we want the best combo of C and sigma. Set
%vectors of C and sigma values to choose from
C_v = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_v = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%initialize a matrix to track the prediction error for each guess of C and
%sigma
prediction_error = zeros(length(C_v), length(sigma_v));

%loop through each choice of C and sigma to populate error matrix
for i = 1:length(C_v)
    for j = 1:length(sigma_v)
        C = C_v(i);
        sigma = sigma_v(j);
        %train the model with chosen C and sigma on training data and using
        %the gaussian kernel to evaluate similarity
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        %predict output with validation data and obtain validation error
        predictions = svmPredict(model, Xval);
        prediction_error(i,j) = mean(double(predictions ~= yval));
    end
end

%find the minimum validation error for each choice of sigma
min_val_col = min(prediction_error);
%find the column with the minimum error of all choices C and sigma. the
%index corresponds to the value of sigma from sigma_v that gives the best
%model
[min_error, col_ind] = min(min_val_col);
%use the column index to get the row index of the minimum error,
%corresponding to the best choice of C
[~, row_ind] = min(prediction_error(:,col_ind));

%grab the values of C and sigma that give the minimum validation error
C = C_v(row_ind);
sigma = sigma_v(col_ind);


% =========================================================================

end
