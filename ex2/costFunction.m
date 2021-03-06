function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta,1) - 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% initialize_theta = zeros(n+1, 1);
z = X*theta;
insidecost = zeros(m, 1);
for i = 1:m
    insidecost(i) = -y(i)*log(sigmoid(z(i))) - (1 - y(i))*log(1-sigmoid(z(i)));
    for j = 1:n+1
        insidegrad(i, j) = (sigmoid(z(i)) - y(i)) * X(i, j);
    end
end
J = (1/m)*sum(insidecost);
for j = 1:n+1
    grad(j,1) = (1/m)*sum(insidegrad(:,j));
end




% =============================================================

end
