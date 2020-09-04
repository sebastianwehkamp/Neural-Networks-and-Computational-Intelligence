function [successArray] = perceptron(nDim, mAlpha, nEpochs)
    % Loop to create 50 independent datasets
    successArray = [];
    success = 0;
    
    for a=0:0.25:mAlpha
        successArray = [successArray, success];
        success = 0;
        % Create varying alpha's from 0-3
        for loop = drange(1:50)
            % Create dataset
            nVec = round(a *  nDim);
            [vectors, labels] = generateVector(nDim, nVec);
            [dim, P] = size(vectors);
            % Create a weights vector of 0's
            weights = zeros(1,dim);
            for epoch=1:nEpochs
                % Keep track of succes
                correct = 0;
                for index=1:P 
                    % For every vector calculate the error and adjust
                    % weights
                    vector = vectors(:, index)';
                    label = labels(index);
                    % error = weights' * vector * label;
                    error = dot(weights, vector) * label;
                    if error <= 0
                        weights = weights + 1/dim*vector*label;
                    end
                end
                % Check whether it was succesful. If so stop.
                for index=1:P
                    vector = vectors(:, index)';
                    label = labels(index);
                    error = dot(weights, vector) * label;
                    if error > 0
                        correct = correct + 1;
                    end
                end
                % Keep track of number of successes
                if correct == P
                    success = success + 1;
                    break
                end
            end
        end
    end
    % Remove first element from success array since it's always 0
    successArray(1) = [];
end

% Function which generates vectors and labels
function [vectors, labels] = generateVector(dim, num)
    % Define possible labels
    possibleLabels = [-1, 1];
    % Initialize vectors randomly
    vectors = randn(dim,num);
    % Initialize all labels selected from possibleLabels
    labels = possibleLabels(randi(2,[1,num]));
end
