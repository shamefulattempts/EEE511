% Daniel Bowden
% EEE 511
% Assignment 1

close all;
clear all;
clc;

rng(1);

dev = 1;
size = 1000;
x1 = dev * randn(2,size);
x2 = 2.5 + dev *randn(1,size);
x3 = dev * randn(1,size);
x4 = [x2;x3];
x = [x1,x4];
t1 =zeros(1,size);
t2 = ones(1,size);
t = [t1,t2];
scatter(x1(1,:),x1(2,:),'r.')
hold on
scatter(x4(1,:),x4(2,:),'g.')
% Randomize Input Data
rp1=randperm(2*size);
mixed_x = x(:,rp1);
mixed_t = t(rp1);
net = perceptron;
net.trainParam.epochs = 20;
[net,tr] = train(net,mixed_x,mixed_t);
plotpc(net.IW{1},net.b{1});
hold off
z1 = dev * randn(2,size);
z2 = 2.5 + dev *randn(1,size);
z3 = dev * randn(1,size);
z = [z1,[z2;z3]];
y = net(z);
figure
plotconfusion(t,y)