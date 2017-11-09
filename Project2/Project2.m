%EEE 511 FALL 2017 
%PROJECT 2
%TEAM 9 (511NINERS)
%DANIEL F. BOWDEN, MEINRAD A. CHARLES, VIVEK K. SHARMA
%SUBMITTED 10/08/2017
clc
close all;
clear all;
rng(1);
raw_data={5;11;16;23;36;58;29;20;10;8;3;0;0;2;11;27;47;63;60;39;28;26;22;11;21;40;78;122;103;73;47;35;11;5;16;34;70;81;111;101;73;40;20;16;5;11;22;40;60;80.9000000000000;83.4000000000000;47.7000000000000;47.8000000000000;30.7000000000000;12.2000000000000;9.60000000000000;10.2000000000000;32.4000000000000;47.6000000000000;54;62.9000000000000;85.9000000000000;61.2000000000000;45.1000000000000;36.4000000000000;20.9000000000000;11.4000000000000;37.8000000000000;69.8000000000000;106.100000000000;100.800000000000;81.6000000000000;66.5000000000000;34.8000000000000;30.6000000000000;7;19.8000000000000;92.5000000000000;154.400000000000;125.900000000000;84.8000000000000;68.1000000000000;38.5000000000000;22.8000000000000;10.2000000000000;24.1000000000000;82.9000000000000;132;130.900000000000;118.100000000000;89.9000000000000;66.6000000000000;60;46.9000000000000;41;21.3000000000000;16;6.40000000000000;4.10000000000000;6.80000000000000;14.5000000000000;34;45;43.1000000000000;47.5000000000000;42.2000000000000;28.1000000000000;10.1000000000000;8.10000000000000;2.50000000000000;0;1.40000000000000;5;12.2000000000000;13.9000000000000;35.4000000000000;45.8000000000000;41;30.1000000000000;23.9000000000000;15.6000000000000;6.60000000000000;4;1.80000000000000;8.50000000000000;16.6000000000000;36.3000000000000;49.6000000000000;64.2000000000000;67;70.9000000000000;47.8000000000000;27.5000000000000;8.50000000000000;13.2000000000000;56.9000000000000;121.500000000000;138.300000000000;103.200000000000;85.7000000000000;64.6000000000000;36.7000000000000;24.2000000000000;10.7000000000000;15;40.1000000000000;61.5000000000000;98.5000000000000;124.700000000000;96.3000000000000;66.6000000000000;64.5000000000000;54.1000000000000;39;20.6000000000000;6.70000000000000;4.30000000000000;22.7000000000000;54.8000000000000;93.8000000000000;95.8000000000000;77.2000000000000;59.1000000000000;44;47;30.5000000000000;16.3000000000000;7.30000000000000;37.6000000000000;74;139;111.200000000000;101.600000000000;66.2000000000000;44.7000000000000;17;11.3000000000000;12.4000000000000;3.40000000000000;6;32.3000000000000;54.3000000000000;59.7000000000000;63.7000000000000;63.5000000000000;52.2000000000000;25.4000000000000;13.1000000000000;6.80000000000000;6.30000000000000;7.10000000000000;35.6000000000000;73;85.1000000000000;78;64;41.8000000000000;26.2000000000000;26.7000000000000;12.1000000000000;9.50000000000000;2.70000000000000;5;24.4000000000000;42;63.5000000000000;53.8000000000000;62;48.5000000000000;43.9000000000000;18.6000000000000;5.70000000000000;3.60000000000000;1.40000000000000;9.60000000000000;47.4000000000000;57.1000000000000;103.900000000000;80.6000000000000;63.6000000000000;37.6000000000000;26.1000000000000;14.2000000000000;5.80000000000000;16.7000000000000;44.3000000000000;63.9000000000000;69;77.8000000000000;64.9000000000000;35.7000000000000;21.2000000000000;11.1000000000000;5.70000000000000;8.70000000000000;36.1000000000000;79.7000000000000;114.400000000000;109.600000000000;88.8000000000000;67.8000000000000;47.5000000000000;30.6000000000000;16.3000000000000;9.60000000000000;33.2000000000000;92.6000000000000;151.600000000000;136.300000000000;134.700000000000;83.9000000000000;69.4000000000000;31.5000000000000;13.9000000000000;4.40000000000000;38;141.700000000000;190.200000000000;184.800000000000;159;112.300000000000;53.9000000000000;37.6000000000000;27.9000000000000;10.2000000000000;15.1000000000000;47;93.7000000000000;105.900000000000;105.500000000000;104.500000000000;66.6000000000000;68.9000000000000;38;34.5000000000000};
raw_data_train=raw_data(1:255);
min_MSE=10000;
min_seed=0;
min_delay=0;
min_units=0;
ind=1;
for i= 10:1:20
    for j=10:5:70
        MSE_best=10000000;
        for k=1:10
            rng(k);
            delays=1:j;
            hidden_size=[i i];
            temp_net=narnet(delays,hidden_size);
            temp_net=RandomizeWeights(temp_net);
            net=temp_net;
            net.divideParam.trainRatio=0.85;
            net.divideParam.valRatio=0.15;
            net.divideParam.testRatio=0;
            [Xs,Xi,Ai,Ts] = preparets(net,{},{},raw_data_train');
            net = train(net,Xs,Ts,Xi,Ai);
            [Y,Xf,Af] = net(Xs,Xi,Ai);
            perf = perform(net,Ts,Y);
            [netc,Xic,Aic] = closeloop(net,Xf,Af);
            y2 = netc(cell(0,20),Xic,Aic);
            MAE=(1/20)*sum(abs(cell2mat(raw_data(256:275))-cell2mat(y2')));
            MSE=(1/20)*sum((cell2mat(raw_data(256:275))-cell2mat(y2')).^2);
            if MSE<min_MSE
                min_MSE=MSE;
                min_MAE=MAE;
                min_seed=k;
                min_delay=j;
                min_hidden=i;
                min_net=temp_net;
                temp_min_net=net;
                min_Xs=Xs;
                min_Xi=Xi;
                min_Ai=Ai;
                min_Ts=Ts;
                min_perf=perf;
            end
            if MSE<MSE_best
                MSE_best=MSE;
            end
        end
        plot_z(i-9,(j/5)-1)=MSE_best;
        plot_x(i-9)=i;
        plot_y((j/5)-1)=j;
        
    end
end
[Y,Xf,Af] = temp_min_net(min_Xs,min_Xi,min_Ai);
perf(i) = perform(temp_min_net,min_Ts,Y);
[netc,Xic,Aic] = closeloop(temp_min_net,Xf,Af);
y2 = netc(cell(0,20),Xic,Aic);
model_error=cell2mat(raw_data(256:275))-cell2mat(y2');
figure()
bar(256:275,model_error)
title('Model Error')
xlabel('Year')
ylabel('Target-Model Estimate')
figure()
plot(256:275,cell2mat(raw_data(256:275)),'-x')
hold on
plot(256:275,cell2mat(y2),'-o')
title('Target Data vs Estimated Data')
xlabel('Year')
ylabel('Physical Variable')
legend('Target Data', 'Estimated Data')
hold off
figure()
surfl(plot_x,plot_y,1./plot_z')
title('1/MSE vs Hidden Neurons vs Delay')
xlabel('X: Hidden Neurons')
ylabel('Y: Delay')
zlabel('Z: 1/MSE')
figure
plot(1:275,cell2mat(raw_data))
title('Provided data')
xlabel('Year')
ylabel('Physical Variable')
rng(7);
delays=1:50;
hidden_size=[12 12];
min_net=narnet(delays,hidden_size);
min_net=RandomizeWeights(min_net);
min_net.divideParam.trainRatio=0.85;
min_net.divideParam.valRatio=0.15;
min_net.divideParam.testRatio=0;
[Xs,Xi,Ai,Ts] = preparets(min_net,{},{},raw_data');
min_net = train(min_net,Xs,Ts,Xi,Ai);
[Y,Xf,Af] = min_net(Xs,Xi,Ai);
[netc,Xic,Aic] = closeloop(min_net,Xf,Af);
y3 = netc(cell(0,30),Xic,Aic);

figure()
plot(1:275,cell2mat(raw_data),'-x')
hold on
plot(276:305,cell2mat(y3),'-o')
title('Provided Data and Predicted Data')
xlabel('Year')
ylabel('Physical Variable')
legend('Provided Data', 'Predicted Data')
hold off

function [net]=RandomizeWeights(net)
% Randomly initializing the weight vector
net.initFcn = 'initlay';

for j=1:size(net.layers,1)
    net.layers{j}.initFcn = 'initwb';
end

for j=1:size(net.inputWeights,1)
    for k=1:size(net.inputWeights,2)
        if ~isempty(net.inputWeights{j,k})
            net.inputWeights{j,k}.initFcn = 'rands';
        end
    end
end
for j=1:size(net.layerWeights,1)
    for k=1:size(net.layerWeights,2)
        if ~isempty(net.layerWeights{j,k})
            net.layerWeights{j,k}.initFcn = 'rands';
        end
    end
end
%initialize net with required weights
net=init(net);
end
