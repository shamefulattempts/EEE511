%EEE 511 FALL 2017 
%PROJECT 1
%TEAM 9 (511NINERS)
%DANIEL F. BOWDEN, MEINRAD A. CHARLES, VIVEK K. SHARMA
%SUBMITTED 10/15/2017
close all;
clear all;
clc

rng(25)

[x_train1,y_train1,t_train1,x1_nomix,y1_nomix]=arc_generator(1000,2,10,6);
[x_test1,y_test1,t_test1,~,~]=arc_generator(500,2,10,6);
[tr1,bound1,z1]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,1,'traingd',.01,10);
[tr2,bound2,z2]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,2,'traingdm',.01,10);
[tr3,bound3,z3]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,3,'trainlm',.01,10);
[tr4,bound4,z4]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,1,'traingd',.1,10);
[tr5,bound5,z5]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,2,'traingdm',.1,10);
[tr6,bound6,z6]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,3,'trainlm',.1,10);
[tr7,bound7,z7]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,1,'traingd',1,10);
[tr8,bound8,z8]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,2,'traingdm',1,10);
[tr9,bound9,z9]=design_net(x_train1,y_train1,t_train1,x_test1,y_test1,t_test1,3,'trainlm',1,10);
figure(1);
plotconfusion(t_test1,z4);
title('Confusion Matrix Back-propagation with d=2');
figure(2);
plotconfusion(t_test1,z5);
title('Confusion Matrix Back-propagation with Momentum with d=2');
figure(3);
plotconfusion(t_test1,z6);
title('Confusion Matrix Levenberg-Marquardt with d=2');
figure(4);
hold on
scatter(x1_nomix(1:1000),y1_nomix(1:1000))
scatter(x1_nomix(1001:2000),y1_nomix(1001:2000))
plot(bound1(1,:),bound1(2,:),'r')
plot(bound2(1,:),bound2(2,:),'g')
plot(bound3(1,:),bound3(2,:),'k')
plot(bound4(1,:),bound4(2,:),'--r')
plot(bound5(1,:),bound5(2,:),'--g')
plot(bound6(1,:),bound6(2,:),'--k')
plot(bound7(1,:),bound7(2,:),'-.r')
plot(bound8(1,:),bound8(2,:),'-.g')
plot(bound9(1,:),bound9(2,:),'-.k')
title('Scatter plot with decision boundary with d=2');
legend('Cluster A','Cluster B','Back-propagation w/ L.R=.01','Back-propagation with momentum w/ L.R=.01','Levenberg-Marquardt w/ L.R=.01','Back-propagation w/ L.R=.1','Back-propagation with momentum w/ L.R=.1','Levenberg-Marquardt w/ L.R=.1','Back-propagation w/ L.R=1','Back-propagation with momentum w/ L.R=1','Levenberg-Marquardt w/ L.R=1');
xlabel('X')
ylabel('Y')
hold off
figure(5)
hold on
semilogy(tr1.epoch,tr1.perf,'r')
semilogy(tr2.epoch,tr2.perf,'g')
semilogy(tr3.epoch,tr3.perf,'k')
semilogy(tr4.epoch,tr4.perf,'--r')
semilogy(tr5.epoch,tr5.perf,'--g')
semilogy(tr6.epoch,tr6.perf,'--k')
semilogy(tr7.epoch,tr7.perf,'-.r')
semilogy(tr8.epoch,tr8.perf,'-.g')
semilogy(tr9.epoch,tr9.perf,'-.k')
title('Learning curve with d=2');
legend('Back-propagation w/ L.R=.01','Back-propagation with momentum w/ L.R=.01','Levenberg-Marquardt w/ L.R=.01','Back-propagation w/ L.R=.1','Back-propagation with momentum w/ L.R=.1','Levenberg-Marquardt w/ L.R=.1','Back-propagation w/ L.R=1','Back-propagation with momentum w/ L.R=1','Levenberg-Marquardt w/ L.R=1');
xlabel('Epochs')
ylabel('Mean Square Error (MSE)')
hold off

[x_train2,y_train2,t_train2,x2_nomix,y2_nomix]=arc_generator(1000,-4,10,6);
[x_test2,y_test2,t_test2,~,~]=arc_generator(500,-4,10,6);
[tr10,bound10,z10]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,6,'traingd',.01,10);
[tr11,bound11,z11]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,7,'traingdm',.01,10);
[tr12,bound12,z12]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,8,'trainlm',.01,10);
[tr13,bound13,z13]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,6,'traingd',.1,10);
[tr14,bound14,z14]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,7,'traingdm',.1,10);
[tr15,bound15,z15]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,8,'trainlm',.1,10);
[tr16,bound16,z16]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,6,'traingd',1,10);
[tr17,bound17,z17]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,7,'traingdm',1,10);
[tr18,bound18,z18]=design_net(x_train2,y_train2,t_train2,x_test2,y_test2,t_test2,8,'trainlm',1,10);
figure(6);
plotconfusion(t_test2,z13);
title('Confusion Matrix Back-propagation with d=-4');
figure(7);
plotconfusion(t_test2,z14);
title('Confusion Matrix Back-propagation with momentum with d=-4');
figure(8);
plotconfusion(t_test2,z15);
title('Confusion Matrix Levenberg-Marquardt with d=-4');
figure(9);
hold on
scatter(x2_nomix(1:1000),y2_nomix(1:1000))
scatter(x2_nomix(1001:2000),y2_nomix(1001:2000))
plot(bound10(1,:),bound10(2,:),'r')
plot(bound11(1,:),bound11(2,:),'g')
plot(bound12(1,:),bound12(2,:),'k')
plot(bound13(1,:),bound13(2,:),'--r')
plot(bound14(1,:),bound14(2,:),'--g')
plot(bound15(1,:),bound15(2,:),'--k')
plot(bound16(1,:),bound16(2,:),'-.r')
plot(bound17(1,:),bound17(2,:),'-.g')
plot(bound18(1,:),bound18(2,:),'-.k')
title('Scatter plot with decision boundary with d=-4');
legend('Cluster A','Cluster B','Back-propagation w/ L.R=.01','Back-propagation with momentum w/ L.R=.01','Levenberg-Marquardt w/ L.R=.01','Back-propagation w/ L.R=.1','Back-propagation with momentum w/ L.R=.1','Levenberg-Marquardt w/ L.R=.1','Back-propagation w/ L.R=1','Back-propagation with momentum w/ L.R=1','Levenberg-Marquardt w/ L.R=1');
xlabel('X')
ylabel('Y')
hold off
figure(10)
hold on
semilogy(tr10.epoch,tr10.perf,'r')
semilogy(tr11.epoch,tr11.perf,'g')
semilogy(tr12.epoch,tr12.perf,'k')
semilogy(tr13.epoch,tr13.perf,'--r')
semilogy(tr14.epoch,tr14.perf,'--g')
semilogy(tr15.epoch,tr15.perf,'--k')
semilogy(tr16.epoch,tr16.perf,'-.r')
semilogy(tr17.epoch,tr17.perf,'-.g')
semilogy(tr18.epoch,tr18.perf,'-.k')
title('Learning curve with d=-4');
legend('Back-propagation w/ L.R=.01','Back-propagation with momentum w/ L.R=.01','Levenberg-Marquardt w/ L.R=.01','Back-propagation w/ L.R=.1','Back-propagation with momentum w/ L.R=.1','Levenberg-Marquardt w/ L.R=.1','Back-propagation w/ L.R=1','Back-propagation with momentum w/ L.R=1','Levenberg-Marquardt w/ L.R=1');
xlabel('Epochs')
ylabel('Mean Square Error (MSE)')
hold off

[x_train3,y_train3,t_train3,x3_nomix,y3_nomix]=arc_generator(1000,-8,10,6);
[x_test3,y_test3,t_test3,~,~]=arc_generator(500,-8,10,6);
[tr19,bound19,z19]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,11,'traingd',.01,10);
[tr20,bound20,z20]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,12,'traingdm',.01,10);
[tr21,bound21,z21]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,13,'trainlm',.01,10);
[tr22,bound22,z22]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,11,'traingd',.1,10);
[tr23,bound23,z23]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,12,'traingdm',.1,10);
[tr24,bound24,z24]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,13,'trainlm',.1,10);
[tr25,bound25,z25]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,11,'traingd',1,10);
[tr26,bound26,z26]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,12,'traingdm',1,10);
[tr27,bound27,z27]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,13,'trainlm',1,10);
figure(11);
plotconfusion(t_test3,z22);
title('Confusion Matrix Back-propagation with d=-8');
figure(12);
plotconfusion(t_test3,z23);
title('Confusion Matrix Back-propagation with momentum with d=-8');
figure(13);
plotconfusion(t_test3,z24);
title('Confusion Matrix Levenberg-Marquardt with d=-8');
figure(14);
hold on
scatter(x3_nomix(1:1000),y3_nomix(1:1000))
scatter(x3_nomix(1001:2000),y3_nomix(1001:2000))
plot(bound19(1,:),bound19(2,:),'r')
plot(bound20(1,:),bound20(2,:),'g')
plot(bound21(1,:),bound21(2,:),'k')
plot(bound22(1,:),bound22(2,:),'--r')
plot(bound23(1,:),bound23(2,:),'--g')
plot(bound24(1,:),bound24(2,:),'--k')
plot(bound25(1,:),bound25(2,:),'-.r')
plot(bound26(1,:),bound26(2,:),'-.g')
plot(bound27(1,:),bound27(2,:),'-.k')
title('Scatter plot with decision boundary with d=-8');
legend('Cluster A','Cluster B','Back-propagation w/ L.R=.01','Back-propagation with momentum w/ L.R=.01','Levenberg-Marquardt w/ L.R=.01','Back-propagation w/ L.R=.1','Back-propagation with momentum w/ L.R=.1','Levenberg-Marquardt w/ L.R=.1','Back-propagation w/ L.R=1','Back-propagation with momentum w/ L.R=1','Levenberg-Marquardt w/ L.R=1');
xlabel('X')
ylabel('Y')
hold off
figure(15)
hold on
semilogy(tr19.epoch,tr19.perf,'r')
semilogy(tr20.epoch,tr20.perf,'g')
semilogy(tr21.epoch,tr21.perf,'k')
semilogy(tr22.epoch,tr22.perf,'--r')
semilogy(tr23.epoch,tr23.perf,'--g')
semilogy(tr24.epoch,tr24.perf,'--k')
semilogy(tr25.epoch,tr25.perf,'-.r')
semilogy(tr26.epoch,tr26.perf,'-.g')
semilogy(tr27.epoch,tr27.perf,'-.k')
title('Learning curve with d=-8');
legend('Back-propagation w/ L.R=.01','Back-propagation with momentum w/ L.R=.01','Levenberg-Marquardt w/ L.R=.01','Back-propagation w/ L.R=.1','Back-propagation with momentum w/ L.R=.1','Levenberg-Marquardt w/ L.R=.1','Back-propagation w/ L.R=1','Back-propagation with momentum w/ L.R=1','Levenberg-Marquardt w/ L.R=1');
xlabel('Epochs')
ylabel('Mean Square Error (MSE)')
hold off

[tr10,bound10,z10]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,16,'traingd',.1,1);
[tr11,bound11,z11]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,17,'traingd',.1,5);
[tr12,bound12,z12]=design_net(x_train3,y_train3,t_train3,x_test3,y_test3,t_test3,18,'traingd',.1,10);

figure(16);
plotconfusion(t_test3,z10);
title('Confusion Matrix Back-propagation with d=-8 and 1 hidden neuron');
figure(17);
plotconfusion(t_test3,z11);
title('Confusion Matrix Back-propagation with d=-8 and 5 hidden neurons');
figure(18);
plotconfusion(t_test3,z12);
title('Confusion Matrix Back-propagation with d=-8 and 10 hidden neurons');
figure(19);
hold on
scatter(x3_nomix(1:1000),y3_nomix(1:1000))
scatter(x3_nomix(1001:2000),y3_nomix(1001:2000))
plot(bound10(1,:),bound10(2,:),'r')
plot(bound11(1,:),bound11(2,:),'g')
plot(bound12(1,:),bound12(2,:),'c')
title('Scatter plot with decision boundary with d=-8');
legend('Cluster A','Cluster B','1 Hidden neuron','5 Hidden neurons','10 Hidden neurons');
hold off
figure(20)
hold on
semilogy(tr10.epoch,tr10.perf,'r')
semilogy(tr11.epoch,tr11.perf,'g')
semilogy(tr12.epoch,tr12.perf,'c')
title('Learning curve with d=-8');
legend('1 Hidden neuron','5 Hidden neurons','10 Hidden neurons');
hold off


function [ x_data_m,y_data_m,t_data_m,x_data,y_data ] = arc_generator( size,d,r,w )

    x_data=zeros(2*size,1);
    y_data=zeros(2*size,1);
    t_data=[zeros(size,1);ones(size,1)];
    x_center=r;
    y_center=-d;
    inner_radius=r-.5*w;
    outer_radius=r+.5*w;
%     Top Circle
    for i=1:size
        got_point=false;
        while got_point==false
            x=outer_radius*(2*rand(1)-1);
            y=outer_radius*rand(1);
            radius = sqrt(x^2+y^2);
            if (radius<outer_radius && radius>inner_radius)
                got_point=true;
            end
        end
        x_data(i)=x;
        y_data(i)=y;
    end
%     Bottom Circle
    for i=size+1:2*size
        got_point=false;
        while got_point==false
            x=outer_radius*(2*rand(1)-1);
            y=-outer_radius*rand(1);
            radius = sqrt(x^2+y^2);
            if (radius<outer_radius && radius>inner_radius)
                got_point=true;
            end
        end
        x_data(i)=x+x_center;
        y_data(i)=y+y_center;
    end
%     Mix training data
    rp=randperm(2*size);
    x_data_m = x_data(rp);
    y_data_m = y_data(rp);
    t_data_m = t_data(rp);
    x_data_m=x_data_m.';
    y_data_m=y_data_m.';
    t_data_m=t_data_m.';
end

function [bound]=decision_boundary(net)
x_iter=(-15:.5:25);
y_iter=(-10:.25:15);
bound=15*ones(2,length(x_iter));
for i=1:length(x_iter)
    br=false;
    for j=1:length(y_iter)
        output=net([x_iter(i);y_iter(j)]);
        if (output <= .5)
            bound(1:2,i)=[x_iter(i);y_iter(j)];
            br=true;
            break;
        end
    end
    if br==false
        bound(1:2,i)=[x_iter(i);y_iter(length(y_iter))];
    end
end
end

function [tr,bound,z]=design_net(x_train,y_train,t_train,x_test,y_test,t_test,fig_num_start,method,learning_rate,hidden_neurons)
net=feedforwardnet(hidden_neurons,method);

%using tanh for output activation function
net.layers{2}.transferFcn = 'tansig';
%Dividing up training data
net.divideParam.trainRatio = 0.66;
net.divideParam.valRatio = 0.34;
net.divideParam.testRatio = 00;
net.trainParam.epochs=5000;
% Randomly initializing the weight vector
net.initFcn = 'initlay';

for i=1:size(net.layers,1)
  net.layers{i}.initFcn = 'initwb'; 
end

for i=1:size(net.inputWeights,1)
  for j=1:size(net.inputWeights,2)
       if ~isempty(net.inputWeights{i,j})
           net.inputWeights{i,j}.initFcn = 'rands';
       end
  end
end
for i=1:size(net.layerWeights,1)
  for j=1:size(net.layerWeights,2)
       if ~isempty(net.layerWeights{i,j})
           net.layerWeights{i,j}.initFcn = 'rands';
       end
  end
end
%initialize net with required weights
net=init(net);

net.trainParam.lr = learning_rate;
input=[x_train;y_train];
test=[x_test;y_test];
[net,tr] = train(net,input,t_train);
z = net(test);
bound=decision_boundary(net);
end
