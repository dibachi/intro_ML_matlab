function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

z = X*theta;
insidecost = zeros(m, 1);
for i = 1:m
    insidecost(i) = -y(i)*log(sigmoid(z(i))) - (1 - y(i))*log(1-sigmoid(z(i)));
    for j = 1:n+1
        insidegrad(i, j) = (sigmoid(z(i)) - y(i)) * X(i, j);
    end
end
J = (1/m)*sum(insidecost) + (lambda/(2*m))*sum(theta(2:n+1).^2);
for j = 1:n+1
    if j == 1
        grad(j,1) = (1/m)*sum(insidegrad(:,j));
    else
        grad(j,1) = (1/m)*sum(insidegrad(:,j)) + (lambda/m)*theta(j);
    end
end





% =============================================================

end
