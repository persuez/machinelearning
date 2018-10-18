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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

temp1 = 0;
for i = 1 : m
  h = sigmoid(theta' * X(i, :)');
  temp1 = temp1 + y(i) * log(h) + (1 - y(i)) * log(1 - h);
endfor

temp1 = -1 * temp1 / m;

temp2 = 0;
for j = 2 : size(theta)
  temp2 = temp2 + theta(j)^2;
endfor

J = temp1 + temp2 * lambda / (2 * m);

% compute gradient
for j = 1 : size(grad)
  for i = 1 : m
    h = sigmoid(theta' * X(i, :)');
    grad(j) = grad(j) + (h - y(i)) * X(i, :)(j);
  endfor
  grad(j) = grad(j) / m;
  if (j > 1)
    grad(j) = grad(j) + lambda / m * theta(j);
  endif
endfor






% =============================================================

end
