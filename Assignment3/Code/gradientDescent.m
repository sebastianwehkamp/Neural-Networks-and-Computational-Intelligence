function [cost, error, weights] = gradientDescent(xi, tau, p, Q, tmax)
    % Initialize learning rate
    learning_rate = 0.05;
    [n,~] = size(xi);
    % Intialize empty list of costs
    cost = zeros(1, tmax);
    error = zeros(1, tmax);
    
    % Generate data and initialize weights
    vectors = xi;
    weights = randn(n,2);
    % Normalize weights vectors
    weights(:,1) = weights(:,1)/norm(weights(:,1));
    weights(:,2) = weights(:,2)/norm(weights(:,2));
    labels = tau';
        
    for i = 1:tmax
        for j = 1:p
            % Select random vector and label from training data set
            idx = randi(size(vectors(1:p), 2));
            vector = vectors(:,idx);
            label = labels(idx);

            % Calculate gradient
            gradient = gradientCalc(weights,vector,label);

            % Update weights
            weights = weights - (learning_rate * gradient);
        end
        
        % Calc cost function
        cost(i) = mean(1/2 * (sum(tanh(weights'*vectors(:, 1:p)))' - labels(1:p)).^2);
        % Calc generalization error
        error(i) = mean(1/2 * (sum(tanh(weights'*vectors(:, p+1:p+Q)))' - labels(p+1:p+Q)).^2);
    end      
end

% Function which calculates the gradient
function grad = gradientCalc(weights,vector, label)
    grad = ((sum(tanh(weights'*vector)) - label) * (1 - tanh(weights'*vector).^2) * vector')';
end
