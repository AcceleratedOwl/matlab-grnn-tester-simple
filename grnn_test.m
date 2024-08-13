clc; clear all; close all;
 
SPREADS = [0.16, 0.02, 0.24];
 
x = 1:0.3:7;
y = [2.8831 2.2478 1.4118 0.4496 -0.5528 -1.5057 -2.3242 -2.9350 -3.2837 -3.3391 -3.0961 -2.5766 -1.8270 -0.9141 0.0804 1.0677 1.9597 2.6766 3.1544 3.3504 3.2472];
x_test = [2.45 3.15 3.85];
y_test = [-1.3540 -3.0125 -3.2541];
 
new_x = [2.85  3.55];
 
MSEs = zeros(1, length(SPREADS));
 
for i = 1:length(SPREADS)
    figure(i);
    hold on;
    
    plot(x, y, '-xg');
    plot(x_test, y_test, 'ob');
 
    net = newgrnn(x, y, SPREADS(i));
 
    pred_y = net(x_test);
    pred_new_y = net(new_x);
 
    plot(x_test, pred_y, 'r+');
    plot(new_x, pred_new_y, 'r+');
    legend('Training data', 'Test data', 'Prediction data', 'Location', 'South')
 
    mse = perform(net, y_test, pred_y);
    MSEs(i) = mse;
    disp('MSE:')
    disp(mse)
    
    title(['SPREAD = ', num2str(SPREADS(i)), '; MSE = ', num2str(mse)])
    hold off;
end

[min_mse, min_mse_index] = min(MSEs);
disp('min_mse_index')
disp(min_mse_index)
net = newgrnn(x, y, SPREADS(min_mse_index));
view(net);
pred_new_y = net(new_x)
