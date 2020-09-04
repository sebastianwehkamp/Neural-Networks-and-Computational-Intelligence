clear all;
success = perceptronN(20,3,100);

x = linspace(0,3,12);
hold on;
plot(x,success/50);
hold off;
ylabel('Success rate');
xlabel('P/N');
title('P = 200, max epochs = 100');