function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%%%
%add bias node to input data
X = [ones(m,1) X];
%determine values for hidden layer nodes
a2 = sigmoid(X*Theta1');
%add bias node to hidden layer
a2 = [ones(m,1) a2];
%determine network output values
a3 = sigmoid(a2*Theta2');
network_output = a3;

%%%%
%binary matirx for results y. rows are training exs, cols are labels
y_bin = zeros(m, num_labels);
for index = 1:m
    %identifies label for each training ex and turns it into a 1 for its
    %corresponding index among possible NN outputs
    %ex: y = 5 turns into y_bin_matrix(index, :) = [0 0 0 0 1 0 0 0 0 0];
    y_bin(index,y(index)) = 1;
end
%forms theta matrices which exclude bias nodes, which are summed to
%regularize cost function
theta1_reg = Theta1(:,2:end);
theta2_reg = Theta2(:,2:end);
%term to calculate regularization term, summing over rows, columns, and
%layers of theta matrices
regularization = (lambda/(2*m))*(sum(sum(theta1_reg.^2)) + sum(sum(theta2_reg.^2)));
%calculates stuff inside nested sum of cost function
to_be_summed = y_bin.*log(network_output) + ((1-y_bin).*log(1-network_output));

J = -(1/m)*sum(sum(to_be_summed)) + regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%initialize Delta matrices, which accumulate gradient values
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t = 1:m
    %set a1 to the first training example, then transpose to make it a
    %column vector. NOTE: the bias node is already included here
    a1 = X(t,:)';
    %feedforward to obtain output for training ex. t
    z2 = Theta1*a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    %calculate delta3. y_bin matrix has training exs. on rows, so transpose
    %to turn into a column vector
    delta3 = a3 - y_bin(t,:)';
    %term1 is 26x1 while z2 is 25x1. the operations below make dimensions
    %agree and throw out delta2(1) simultaneously, then calculates delta2
    term1 = Theta2'*delta3;
    term1 = term1(2:end);
    delta2 = term1.*sigmoidGradient(z2);
    
    Delta1 = Delta1 + delta2*a1';
    Delta2 = Delta2 + delta3*a2';
end
%implements regularization of gradients by replacing bias node in Theta
%matirces with zeros and adding the resultant regularization matrix to each
%gradient matrix, thus ignoring regularization for bias nodes
grad1_reg = Theta1;
grad1_reg(:,1) = 0;
grad2_reg = Theta2;
grad2_reg(:,1) = 0;

Theta1_grad = (1/m)*Delta1 + (lambda/m)*grad1_reg;
Theta2_grad = (1/m)*Delta2 + (lambda/m)*grad2_reg;
    
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
