clear all;
load('data3.mat');

% Run gradient Descent
[costs, error, weights] = gradientDescent(xi, tau, 2000, 2000, 2000);

% Calculate moving average
meanCost = movmean(costs,100);
meanError = movmean(error,100);

% Plot Error values
hold on
plot(costs);
plot(error);
plot(meanCost);
plot(meanError);
title('Plot of error values');
xlabel('Epoch');
ylabel('error value');
legend('Training error', 'Generalization error');
hold off

% Plot weights in single plot
figure;
subplot(1,2,1);
bar(weights(:,1));
title('Weights 1');
subplot(1,2,2);
bar(weights(:,2));
title('Weights 2');


