function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta) - 1;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

goz = sigmoid(X*theta);

internal = -y.*log(goz) - (1-y).*log(1-goz);
J = (1/m)*sum(internal) + (lambda/(2*m))*sum(theta(2:n+1).^2);
grad = (1/m)*X'*(goz - y) + (lambda/m)*[0; theta(2:n+1)];

%-----
% J = 0;
% grad = zeros(size(theta));
% n = size(theta,1) - 1;
% 
% z = X*theta;
% insidecost = zeros(m, 1);
% for i = 1:m
%     insidecost(i) = -y(i)*log(sigmoid(z(i))) - (1 - y(i))*log(1-sigmoid(z(i)));
%     for j = 1:n+1
%         insidegrad(i, j) = (sigmoid(z(i)) - y(i)) * X(i, j);
%     end
% end
% J = (1/m)*sum(insidecost);
% for j = 1:n+1
%     grad(j,1) = (1/m)*sum(insidegrad(:,j));
% end
%-----






% =============================================================

grad = grad(:);

end
