% Setting parameters
iT = 1000; % Path length
iRep = 250; % Number of replications
iN = 23400; % Discrete points

% Variables set and to be estimated using GMM
dXi = 0.0225; % Expectation of variance
vLambda = [0.005, 0.01, 0.015, 0.035, 0.07];
vV = [1.25, 0.75, 0.5, 0.3, 0.2];
vH = [0.05, 0.1, 0.3, 0.5, 0.7]; % Hurst exponents
num_repetitions = iRep;

% acf number of lags
vell = [0,1,2,3,5,20,50];

%% fBm plot
rng(123)
f1 = fbm(0.1,iN,1);
rng(123)
f2 = fbm(0.5,iN,1);
rng(123)
f3 = fbm(0.9,iN,1);
t = 0:1/(iN-1):1;
subplot(1,3,1)
plot(t,f1)
xlabel('time')
title('H=0.1')
subplot(1,3,2)
plot(t,f2)
xlabel('time')
title('H=0.5')
subplot(1,3,3)
plot(t,f3)
xlabel('time')
title('H=0.9')

%% Figure 1 panal A Bolko
vY = SimModelvY(iN, 1, 0, dXi, vLambda, vV, vH, 2);
t = 0:1/(iN-1):1;
plot(t,(vY.vY(1,:)))
hold on
plot(t,vY.vY(2,:))
plot(t,vY.vY(3,:))
plot(t,vY.vY(4,:))
plot(t,vY.vY(5,:))
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('log(spot variance)')
xlabel('time')
grid("on")
hold off


%% Figure 1 panal A Bolko
vY = SimModelSimplejump(iN, 10,5.5, 0.0225, 0.005,1.25, 0.7, 2);
%t = 0:1/(iN*10-1):10;
plot(((vY.vX)))
%hold on
%plot(vY.jump*250,'-')
%plot(t,exp(vY.vX(4,:)))
%plot(t,exp(vY.vX(5,:)))
legend('H=0.05')
ylabel('Asset price')
%ylim([230 290])
xlabel('time')
grid("on")
hold off
%%
t = 0:1/(iN):10;
yMark=NaN(1,length(vY.vX));
 vX = vY.vX;
for i=1:1:length(vY.vX)
  
if vY.jump(i) >= 1
 yMark(i)=exp(vY.vX(i));
  yMark(i-1)=exp(vY.vX(i-1));
  vX(i-1:i) =NaN;
end
end

plot(t(1:end-1),yMark,'->')
hold on


plot(t(1:end-1),exp(vX))
legend('Jump','H=0.05')
ylabel('Asset price')
%ylim([210 255])
xlabel('time')
hold off


%% Figure 1 panal A Bolko
for i =1:5
vY = SimModelSigmajump(iN, 1,5.5, 0.0225, vLambda(i),vV(i), vH(i),1);
%t = 0:1/(iN*10-1):10;


t = 0:1/(iN):1;
yMark=NaN(1,length(vY.vSigma));
 vX = log(vY.vSigma);
for i=1:1:length(vY.vSigma)
  
if vY.jump(i) >= 1
 yMark(i)=log(vY.vSigma(i));
  yMark(i-1)=log(vY.vSigma(i-1));
  vX(i-1:i) =NaN;
end
end

hold on


plot(t(1:end-1),vX)

end

plot(t(1:end-1),yMark,'b*')
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70','Jump')
ylabel('log spot-variance')
%ylim([210 255])
xlabel('time')
hold off

%% Figure 1 panal A Bolko
%vY = SimModelSimplejump(iN, 100,5.5, 0.0225, 0.001,1.25, 0.05, 3);
%t = 0:1/(iN*10-1):10;
plot(((vY.RV)))
hold on
plot((vY.BV))
%plot(t,exp(vY.vX(4,:)))
%plot(t,exp(vY.vX(5,:)))
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('Asset price')
xlabel('time')
grid("on")
hold off

%%
tic
% Simulating the model
vY = SimModel(iN, iT, 0, dXi, vLambda, vV, vH, 2);
%%
% Computing vTheta_IV
vTheta_IV = zeros(5, 8);
for j = 1:5
    vTheta_IV(j, 1) = mean(vY.IV(j, 1:500));  
    vTheta_IV(j, 2) = mean(vY.RV(j, 1:500));
     vTheta_IV(j, 3) = mean(vY.BV(j, 1:500));
      vTheta_IV(j, 4) = 0;%mean(vY.QPV(j, 1:1000));
          vTheta_IV(j, 5) = mean(vY.RVs(j, 1:500));
     vTheta_IV(j, 6) = mean(vY.BVs(j, 1:500));
               vTheta_IV(j, 7) = mean(vY.RVfive(j, 1:500));
     vTheta_IV(j, 8) = mean(vY.BVfive(j, 1:500));
end
disp(vTheta_IV);
toc
%%
[initial(vY.IV(5,:)), initial(vY.RV(5,:)), initial(vY.RVs(5,:)), initial(vY.RVfive(5,:))]
    
%%
time = 0:1/23399:1;
plot(time,exp(vY.vZ(1,1:23400)),'.')
hold on
plot(time,exp(vY.vZ(2,1:23400)),'.')
plot(time,exp(vY.vZ(3,1:23400)),'.')
plot(time,exp(vY.vZ(4,1:23400)),'.')
plot(time,exp(vY.vZ(5,1:23400)),'.')
hold off
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
%%

plot(vY.IV(1, 1:100))
hold on
plot(vY.RV(1, 1:100))
plot(vY.BV(1, 1:100))
plot(vY.RVs(1, 1:100))
plot(vY.BVs(1, 1:100))
hold off
legend('IV','RV', 'BV', 'RV^*','BV^*')
%%

%%
for i = 1:250
    for j = 1:5
PJ(i,j) =  max(1-  vY.BVfive(j, i) / vY.RVfive(j, i),0);
    end
end
plot(PJ(:,1))
hold on 
plot(PJ(:,2))
plot(PJ(:,3))
plot(PJ(:,4))
plot(PJ(:,5))
hold off
legend()
%% Figure 1 panel B Bolko
plot(log(vY.IV(1,1:250)))
hold on
plot(log(vY.IV(2,1:250)))
plot(log(vY.IV(3,1:250)))
plot(log(vY.IV(4,1:250)))
plot(log(vY.IV(5,1:250)))
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('log(integrated variance)')
xlabel('time')
grid("on")
hold off

%% Figure 1 panel B Bolko
plot(log(vY.RV(1,1:250)))
hold on
plot(log(vY.RV(2,1:250)))
plot(log(vY.RV(3,1:250)))
plot(log(vY.RV(4,1:250)))
plot(log(vY.RV(5,1:250)))
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('log(integrated variance)')
xlabel('time')
grid("on")
hold off
%% Figure 1 panel B Bolko
plot(log(vY.BV(1,1:250)))
hold on
plot(log(vY.BV(2,1:250)))
plot(log(vY.BV(3,1:250)))
plot(log(vY.BV(4,1:250)))
plot(log(vY.BV(5,1:250)))
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('log(integrated variance)')
xlabel('time')
grid("on")
hold off

%% IV RV BV simulation plot
plot(log(vY.IV(1,1:250)))
hold on
plot(log(vY.RVfive(1,1:250)))
plot(log(vY.BVfive(1,1:250)))
legend('IV', 'RV^{5min}', 'BV^{5min}')
ylabel('Integrated variance')
xlabel('time')
grid("on")
hold off
%%
rng(100);

% Create a cell array to store simulation from each repetition
%vY_list = cell(1, num_repetitions);
%%
%vY_list = cell(1);
rng(125)
for rep = 1:250   
    tic
    % Generate vY for each repetition
    vY = SimModel(23400, 1000, 0, dXi, vLambda, vV, vH, rand() * 10000000);
    
    % Save simulation in the cell array
    vY_list{rep} = vY;
    
    disp(rep);
    toc
end
%%
% Save the cell array to a .mat file
save('SimulationsBolko.mat', 'vY_list');
%%
% Load the cell array from the .mat file
loadedData = load('fname5.mat');
vY_list = loadedData.vY_list;

%%

for i=1:1000

jt(i) = vY_list{1, 1}.RV(1,i) - vY_list{1, 1}.BV(1,i);
jt2(i) = vY_list{1, 1}.RV(2,i) - vY_list{1, 1}.BV(2,i);
jt3(i) = vY_list{1, 1}.RV(3,i) - vY_list{1, 1}.BV(3,i);
jt4(i) = vY_list{1, 1}.RV(4,i) - vY_list{1, 1}.BV(4,i);
jt5(i) = vY_list{1, 1}.RV(5,i) - vY_list{1, 1}.BV(5,i);


end
plot(jt)
hold on
plot(jt2)
plot(jt3)
plot(jt4)
plot(jt5)
hold off

%%

for i=1:100

PJ(i) = 1-  vY_list{1, 1}.BV(1,i) / vY_list{1, 1}.RV(1,i);
PJ2(i) = 1-  vY_list{1, 1}.BV(2,i) / vY_list{1, 1}.RV(2,i);
PJ3(i) =1-  vY_list{1, 1}.BV(3,i) / vY_list{1, 1}.RV(3,i);
PJ4(i) = 1-  vY_list{1, 1}.BV(4,i) / vY_list{1, 1}.RV(4,i);
PJ5(i) = 1-  vY_list{1, 1}.BV(5,i) / vY_list{1, 1}.RV(5,i);


end
plot(PJ)
hold on
plot(PJ2)
plot(PJ3)
plot(PJ4)
plot(PJ5)
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
hold off

%%
for j = 1:250
for i=1:1000

PJ(i,j) = max(1-  vY_list{1, j}.BVfive(1,i) / vY_list{1, j}.RVfive(1,i),0);
PJ2(i,j) =max( 1-  vY_list{1, j}.BVfive(2,i) / vY_list{1, j}.RVfive(2,i),0);
PJ3(i,j) =max( 1-  vY_list{1, j}.BVfive(3,i) / vY_list{1, j}.RVfive(3,i),0);
PJ4(i,j) =max( 1-  vY_list{1, j}.BVfive(4,i) / vY_list{1, j}.RVfive(4,i),0);
PJ5(i,j) =max( 1-  vY_list{1, j}.BVfive(5,i) / vY_list{1, j}.RVfive(5,i),0);


end
PJmean(j) = mean(PJ(:,j));
PJmean2(j) = mean(PJ2(:,j));
PJmean3(j) = mean(PJ3(:,j));
PJmean4(j) = mean(PJ4(:,j));
PJmean5(j) = mean(PJ5(:,j));
end
semilogy(PJmean(1:250),'.','MarkerSize',10)
hold on 
semilogy(PJmean2(1:250),'.','MarkerSize',10)
semilogy(PJmean3(1:250),'.','MarkerSize',10)
semilogy(PJmean4(1:250),'.','MarkerSize',10)
semilogy(PJmean5(1:250),'.','MarkerSize',10)
hold off
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('Jump proportion')
xlabel('Simulation number')
grid("on")

%%
for j = 1:250
for i=1:1000

PJ(i,j) = max(1-  vY_list{1, j}.BVs(1,i) / vY_list{1, j}.RVs(1,i),0);
PJ2(i,j) =max( 1-  vY_list{1, j}.BVs(2,i) / vY_list{1, j}.RVs(2,i),0);
PJ3(i,j) =max( 1-  vY_list{1, j}.BVs(3,i) / vY_list{1, j}.RVs(3,i),0);
PJ4(i,j) =max( 1-  vY_list{1, j}.BVs(4,i) / vY_list{1, j}.RVs(4,i),0);
PJ5(i,j) =max( 1-  vY_list{1, j}.BVs(5,i) / vY_list{1, j}.RVs(5,i),0);


end
PJmean(j) = mean(PJ(:,j));
PJmean2(j) = mean(PJ2(:,j));
PJmean3(j) = mean(PJ3(:,j));
PJmean4(j) = mean(PJ4(:,j));
PJmean5(j) = mean(PJ5(:,j));
end
semilogy(PJmean,'.','MarkerSize',10)
hold on 
semilogy(PJmean2,'.','MarkerSize',10)
semilogy(PJmean3,'.','MarkerSize',10)
semilogy(PJmean4,'.','MarkerSize',10)
semilogy(PJmean5,'.','MarkerSize',10)
hold off
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('Jump proportion')
xlabel('Simulation number')
grid("on")



%%
vTHeta_intialRV = nan(5,4);
for i =1:5
vTHeta_intialRV(i,:) = initial(vY.RV(i,:));
end
vTHeta_intialRV

%%
vTHeta_intialIV = nan(5,4);
for i =1:5
vTHeta_intialIV(i,:) = initial(vY.IV(i,:));
end
vTHeta_intialIV

%%
% Test kappa_ell function
result_kappa = kappa_ell(1, 0.005, 0.05, 0.36744595);
disp(result_kappa);
%%
% Test E_IV_IV_ell function
result_expectation = E_IV_IV_ell(3, 0.005, 0.9999, 0.36744595, 0.0225);
disp(result_expectation);


%%
tic
Estimate = cell(1,5);  
for i = 1:5
Estimate{i} =  GMM_Estimator3(vY.IV(i,:), initial(vY.IV(i,:))', vell,'N');
end
toc

%%
   estimatrix = NaN(5, 4);
    for j = 1:5
        estimatrix(j, 1:4) = Estimate{j};
    end
estimatrix

%% Initial values for IV
for i = 1:250
one(i,:) = initial(vY_list{1,i}.IV(1,:));
two(i,:) = initial(vY_list{1,i}.IV(2,:));  
three(i,:) = initial(vY_list{1,i}.IV(3,:));  
four(i,:) = initial(vY_list{1,i}.IV(4,:));  
five(i,:) = initial(vY_list{1,i}.IV(5,:));  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)
%% Initial values for RV
for i = 1:250
one(i,:) = initial(vY_list{1,i}.RVfive(1,:));
two(i,:) = initial(vY_list{1,i}.RVfive(2,:));  
three(i,:) = initial(vY_list{1,i}.RVfive(3,:));  
four(i,:) = initial(vY_list{1,i}.RVfive(4,:));  
five(i,:) = initial(vY_list{1,i}.RVfive(5,:));  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)
%% Initial values for IV obs
for i = 1:250
one(i,:) = initial(vY_list{1,i}.IV(1,1:500));
two(i,:) = initial(vY_list{1,i}.IV(2,1:500));  
three(i,:) = initial(vY_list{1,i}.IV(3,1:500));  
four(i,:) = initial(vY_list{1,i}.IV(4,1:500));  
five(i,:) = initial(vY_list{1,i}.IV(5,1:500));  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)
%% Initial values for RV obs
for i = 1:250
one(i,:) = initial(vY_list{1,i}.RVfive(1,1:500));
two(i,:) = initial(vY_list{1,i}.RVfive(2,1:500));  
three(i,:) = initial(vY_list{1,i}.RVfive(3,1:500));  
four(i,:) = initial(vY_list{1,i}.RVfive(4,1:500));  
five(i,:) = initial(vY_list{1,i}.RVfive(5,1:500));  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)
%% GMM Estimation for IV

for rep = 1:250
    tic
    for i = 1:5
    [EstimatesIV{rep}{i}, JACOBIBolkoIV{rep}{i}, HACBolkoIV{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.IV(i,:), initial(vY_list{1,rep}.IV(i,:))', vell,'N');
    end
    disp(rep)
    toc
end


%% GMM Estimation for IV mom

for rep = 2:250
    tic
    for i = 1:5
    [EstimatesIVmom{rep}{i}, JACOBIBolkoIVmom{rep}{i}, HACBolkoIVmom{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.IV(i,:), initial(vY_list{1,rep}.IV(i,:))', [0 1 2 5 10],'N');
    end
    disp(rep)
    toc
end

%% GMM Estimation for IV obs

for rep = 1:250
    tic
    for i = 1:5
    [EstimatesIVobs{rep}{i}, JACOBIBolkoIVobs{rep}{i}, HACBolkoIVobs{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.IV(i,1:500), initial(vY_list{1,rep}.IV(i,1:500))', vell,'N');
    end
    disp(rep)
    toc
end

%% GMM Estimation for IV obs

for rep = 1:250
    tic
    for i = 1:5
    [EstimatesIVobs{rep}{i}, JACOBIBolkoIVobs{rep}{i}, HACBolkoIVobs{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.IV(i,1:500), initial(vY_list{1,rep}.IV(i,1:500))', vell,'N');
    end
    disp(rep)
    toc
end

%% GMM Estimation for IV kernel

for rep = 1:250
    tic
    for i = 1:5
    [EstimatesIVk{rep}{i}, JACOBIBolkoIVk{rep}{i}, HACBolkoIVk{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.IV(i,:), initial(vY_list{1,rep}.IV(i,:))', vell,'N');
    end
    disp(rep)
    toc
end

%%
seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIBolkoIV{1, i}{1, j}.'*inv(HACBolkoIV{1, i}{1, j})*JACOBIBolkoIV{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4)*(iT));
density(i) = (EstimatesIV{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-10 10])
%%
seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIBolkoIVmom{1, i}{1, j}.'*inv(HACBolkoIVmom{1, i}{1, j})*JACOBIBolkoIVmom{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4)*(iT));
density(i) = (EstimatesIVmom{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-10 10])
%%
seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIBolkoIVobs{1, i}{1, j}.'*inv(HACBolkoIVobs{1, i}{1, j})*JACOBIBolkoIVobs{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4)*(iT/2));
density(i) = (EstimatesIVobs{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-10 10])
%%
seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIBolkoIVk{1, i}{1, j}.'*inv(HACBolkoIVk{1, i}{1, j})*JACOBIBolkoIVk{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4)*(iT));
density(i) = (EstimatesIVk{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-10 10])
%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesIVmom{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesIVmom{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesIVmom{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesIVmom{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesIV{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesIV{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesIV{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesIV{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesIVk{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesIVk{1, h}{1, j}(2);   
    GMMv(j,h) = EstimatesIVk{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesIVk{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesIVobs{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesIVobs{1, h}{1, j}(2);   
    GMMv(j,h) = EstimatesIVobs{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesIVobs{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]


%% GMM estimation for uncorrected RV
parfor rep = 26:250
    tic
    for i = 1:5
        EstimatesRVu{rep}{i} =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', vell,'N');
    end
   toc
   disp(rep);
end
%% GMM estimation for Uncorrected RV obs
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVuobs{rep}{i}, JACOBIBolkoRVuobs{rep}{i}, HACBolkoRVuobs{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,1:500), initial(vY_list{1,rep}.RVfive(i,1:500))', vell,'N');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Uncorrected RV kernel
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVuk{rep}{i}, JACOBIBolkoRVuk{rep}{i}, HACBolkoRVuk{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', vell,'N');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Uncorrected RV moments
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVumom{rep}{i}, JACOBIBolkoRVumom{rep}{i}, HACBolkoRVumom{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', [0 1 2 5 10],'N');
    end
   toc
   disp(rep);
end 
%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVu{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVu{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVu{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVu{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVumom{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVumom{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVumom{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVumom{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVuk{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVuk{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVuk{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVuk{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%

for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVuobs{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVuobs{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVuobs{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVuobs{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%% GMM estimation for Approximate RV
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVc{rep}{i}, JACOBIBolkoRVc{rep}{i}, HACBolkoRVc{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', vell,'R');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Approximate RV mom
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVcmom{rep}{i}, JACOBIBolkoRVcmom{rep}{i}, HACBolkoRVcmom{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', [0 1 2 5 10],'R');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Approximate RV
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVck{rep}{i}, JACOBIBolkoRVck{rep}{i}, HACBolkoRVck{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', vell,'R');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Approximate RVobs
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVcobs{rep}{i}, JACOBIBolkoRVcobs{rep}{i}, HACBolkoRVcobs{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,1:500), initial(vY_list{1,rep}.RVfive(i,1:500))', vell,'R');
    end
   toc
   disp(rep);
end 

%%
seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIBolkoRVc{1, i}{1, j}.'*inv(HACBolkoRVc{1, i}{1, j})*JACOBIBolkoRVc{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4)*(1000));
density(i) = (EstimatesRVc{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density, 'Bandwidth',1)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-10 10])

%%
seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIBolkoRVcobs{1, i}{1, j}.'*inv(HACBolkoRVcobs{1, i}{1, j})*JACOBIBolkoRVcobs{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4)*(iT/2));
density(i) = (EstimatesRVcobs{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-10 10])
%%
seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIBolkoRVcmom{1, i}{1, j}.'*inv(HACBolkoRVcmom{1, i}{1, j})*JACOBIBolkoRVcmom{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4)*(iT));
density(i) = (EstimatesRVcmom{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-10 10])
%%
seH = 0;
for j = 1:5
for i = 1:100
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = ((jacobianjumpX9{1, i}{1, j}.'*(WjumpX9{1, i}{1, j})*jacobianjumpX9{1, i}{1, j}));
seH2(i) = sqrt(seH(4,4));
density(i) = (EstimatesjumpX9{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density, "bandwidth", 0.5)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[f,xi]= ksdensity(x);
plot(xi,f,'--b','LineWidth',1.5);
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-4 4])
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVc{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVc{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVc{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVc{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVcmom{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVcmom{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVcmom{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVcmom{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVck{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVck{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVck{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVck{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVcobs{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVcobs{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVcobs{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVcobs{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%% GMM estimation for Exact RV
for rep = 58:250
    tic
    for i = 1:5
        EstimatesRVe{rep}{i} =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', vell,'E');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Exact RVmom
for rep = 90:250
    tic
    for i = 1:5
        [EstimatesRVemom{rep}{i}, JACOBIBolkoRVemom{rep}{i}, HACBolkoRVemom{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', [0 1 2 5 10],'E');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Exact RVobs
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVeobs{rep}{i}, JACOBIBolkoRVeobs{rep}{i}, HACBolkoRVeobs{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,1:500), initial(vY_list{1,rep}.RVfive(i,1:500))', vell,'E');
    end
   toc
   disp(rep);
end 
%% GMM estimation for Exact RV kernel
for rep = 1:250
    tic
    for i = 1:5
        [EstimatesRVek{rep}{i}, JACOBIBolkoRVek{rep}{i}, HACBolkoRVek{rep}{i}] =  GMM_Estimator3(vY_list{1,rep}.RVfive(i,:), initial(vY_list{1,rep}.RVfive(i,:))', vell,'E');
    end
   toc
   disp(rep);
end 
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVe{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVe{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVe{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVe{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVemom{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVemom{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVemom{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVemom{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVek{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVek{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVek{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVek{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVeobs{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVeobs{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVeobs{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVeobs{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesRVemom{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVemom{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVemom{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVemom{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%
%estimatesForall = NaN(10, 5 * 4);
% for each repetition
%EstimatesIV = cell(1,250);
parfor rep = 1:20
    % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4);
    for i = 1:5
    EstimatesBVu{rep}{i} =  GMM_Estimator3(vY_list{1,rep}.BV(i,:), initial(vY_list{1,rep}.BV(i,:))', vell,'N');
    EstimatesBVc{rep}{i} =  GMM_Estimator3(vY_list{1,rep}.BV(i,:), initial(vY_list{1,rep}.BV(i,:))', vell,'B');
    end
    %EstimatesIV{rep} = GMM_Estimator2_Parallel(vY_list{1,rep}.IV, reshape(vTheta_initials(rep, :), 5, 4),vell);
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
   % end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
end

%%

%save('GMMestimatesIVBolko.mat', 'EstimatesIV');

%%
save('GMMestimatesRVuBolkoTEST.mat', 'EstimatesRVu');
%%
%save('GMMestimatesRVCBolko.mat', 'EstimatesRVc');
%%
% Load the cell array from the .mat file
%loadedData = load('GMMestimates.mat');

%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0;
for j = 1:1
    for h = 1:50     % Run for each rep
    GMMXi(j,h) = EstimatesRVu{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVu{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVu{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVu{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%
plot(GMMH(1,:))
hold on
plot(0.01:0.01:0.99,'.')
hold off

%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0;
for j = 1:1
    for h = 7:94  % Run for each rep
    GMMXi(j,h) = EstimatesRVc{1, h}(1);
    GMMLambda(j,h) = EstimatesRVc{1, h}(2);
    GMMv(j,h) = EstimatesRVc{1, h}(3);
    GMMH(j,h) = EstimatesRVc{1, h}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]




 %%

Theta_real = ...
   [0.0225, 0.0050, 1.25, 0.050; ...
    0.0225, 0.010, 0.75, 0.1;...
    0.0225, 0.015, 0.5, 0.3;...
    0.0225, 0.035, 0.3, 0.5; ...
    0.0225, 0.070, 0.2, 0.7];

%%


%%
for j = 1:5
for i = 0:50
    kap(i+1,j) = kappa_ell(i,Theta_real(j,2),Theta_real(j,4),Theta_real(j,3));
end
end

plot(kap(:,1),"b")
hold on 
plot(kap(:,2),"r")
plot(kap(:,3),"c")
plot(kap(:,4),"k")
plot(kap(:,5),"g")
hold off
legend()

%%

for j = 1:5
for i = 0:50
    EIVIV(i+1,j) = E_IV_IV_ell(i,Theta_real(j,2),Theta_real(j,4),Theta_real(j,3),Theta_real(j,1));
end
end

plot(EIVIV(:,1),"b")
hold on 
plot(EIVIV(:,2),"r")
plot(EIVIV(:,3),"c")
plot(EIVIV(:,4),"k")
plot(EIVIV(:,5),"g")
hold off
legend()

%%
for j = 1:5
for i = 0:50
    kap(i+1,j) = kappa_ell(i,Theta_real(j,2),Theta_real(j,4),Theta_real(j,3));
end
end

plot(kap(:,1),"b")
hold on 
plot(kap(:,2),"r")
plot(kap(:,3),"c")
plot(kap(:,4),"k")
plot(kap(:,5),"g")
hold off
legend()
%%
for j = 1:5
for i = 0:50
    EIVIV(i+1,j) = E_IV_IV_ell(i,vTHeta_intialRV(j,2),vTHeta_intialRV(j,4),vTHeta_intialRV(j,3),vTHeta_intialRV(j,1));
end
end

plot(EIVIV(:,1),"b")
hold on 
plot(EIVIV(:,2),"r")
plot(EIVIV(:,3),"c")
plot(EIVIV(:,4),"k")
plot(EIVIV(:,5),"g")
hold off
legend()
%%
for j = 1:5
for i = 0:50
    kap(i+1,j) = kappa_ell(i,vTHeta_intialIV(j,2),vTHeta_intialIV(j,4),vTHeta_intialIV(j,3));
end
end

plot(kap(:,1),"b")
hold on 
plot(kap(:,2),"r")
plot(kap(:,3),"c")
plot(kap(:,4),"k")
plot(kap(:,5),"g")
hold off
legend()
%%
for j = 1:5
for i = 0:50
    EIVIV(i+1,j) = E_IV_IV_ell(i,Theta_real(j,2),Theta_real(j,4),Theta_real(j,3),Theta_real(j,1));
end
end

plot(EIVIV(:,1),"b")
hold on 
plot(EIVIV(:,2),"r")
plot(EIVIV(:,3),"c")
plot(EIVIV(:,4),"k")
plot(EIVIV(:,5),"g")
hold off
legend()

%%
for j = 1:5
for i = 0:50
    EIVIViv(i+1,j) = E_IV_IV_ell(i,vTHeta_intialIV(j,2),vTHeta_intialIV(j,4),vTHeta_intialRV(j,3),vTHeta_intialIV(j,1));
end
end
%%
plot(EIVIV(:,1),"b",'LineWidth',2)
hold on 
plot(EIVIV(:,2),"r",'LineWidth',2)
plot(EIVIV(:,3),"c",'LineWidth',2)
plot(EIVIV(:,4),"k",'LineWidth',2)
plot(EIVIV(:,5),"g",'LineWidth',2)

plot(EIVIViv(:,1),"b:",'LineWidth',2)

plot(EIVIViv(:,2),"r:",'LineWidth',2)
plot(EIVIViv(:,3),"c:",'LineWidth',2)
plot(EIVIViv(:,4),"k:",'LineWidth',2)
plot(EIVIViv(:,5),"g:",'LineWidth',2)
hold off
legend()


%%

surf(kap)
legend('H','Lags','Value')
xlabel('Parameter set')
ylabel('Lags')
zlabel('value')
%%

surf(EIVIV)


xlabel('Parameter set')
ylabel('Lags')
zlabel('value')

%% Fig 2a Bolko

for j = 1:iRep
for h = 1:5
    for i = 1:iRep
    stdx(i) =  EstimatesIV{1, i}{1, h}(4);
    end
    statistic(j,h) = (EstimatesIV{1, j}{1, h}(4)   - vH(h))/ (std(stdx));
end
end

[f,xi] = ksdensity(statistic(:,1)); 
figure
plot(xi,f,'LineWidth',1,'LineStyle','--');
hold on 
[f,xi] = ksdensity(statistic(:,2));
plot(xi,f,'LineWidth',1,'LineStyle','--');
[f,xi] = ksdensity(statistic(:,3)); 
plot(xi,f,'LineWidth',1, 'LineStyle','--');
[f,xi] = ksdensity(statistic(:,4)); 
plot(xi,f,'LineWidth',1,'LineStyle','--');
[f,xi] = ksdensity(statistic(:,5)); 
plot(xi,f,'LineWidth',1,'LineStyle','--');
y = normpdf(xi,0,1);
plot(xi,y,'LineWidth',2)
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70','N(0,1)','Location','northwest')
xlim([-4 4])
hold off
%% Fig 2b Bolko

for j = 1:iRep
for h = 1:5
    for i = 1:iRep
    stdx(i) =  EstimatesRVc{1, i}{1, h}(4);
    end
    statistic(j,h) = (EstimatesRVc{1, j}{1, h}(4)   - vH(h))/ (std(stdx));
end
end

[f,xi] = ksdensity(statistic(:,1)); 
figure
plot(xi,f,'LineWidth',1,'LineStyle','--');
hold on 
[f,xi] = ksdensity(statistic(:,2));
plot(xi,f,'LineWidth',1,'LineStyle','--');
[f,xi] = ksdensity(statistic(:,3)); 
plot(xi,f,'LineWidth',1, 'LineStyle','--'); 
[f,xi] = ksdensity(statistic(:,4)); 
plot(xi,f,'LineWidth',1,'LineStyle','--');
[f,xi] = ksdensity(statistic(:,5)); 
plot(xi,f,'LineWidth',1,'LineStyle','--');
y = normpdf(xi,0,1);
plot(xi,y,'LineWidth',2)
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70','N(0,1)','Location','northwest')
xlim([-4 4])
hold off
%%
%% Figure Price miscrostructure noise (REMEMBER to activate vZ in SimModelMS
rng(4)
vYMS = SimModelMS(23400, 1, 1, dXi, vLambda, vV, vH, rand() * 10000000);

plot(0:1/23399:1,exp(vYMS.vZ(1,:)),'.')
hold on
plot(0:1/23399:1,exp(vYMS.vZ(2,:)),'.')
plot(0:1/23399:1,exp(vYMS.vZ(3,:)),'.')
plot(0:1/23399:1,exp(vYMS.vZ(4,:)),'.')
plot(0:1/23399:1,exp(vYMS.vZ(5,:)),'.')

legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('price')
xlabel('time')
grid("on")
hold off
%%
%vY_list = cell(1);
%rng(125)
for rep = 224:250   
    tic
    % Generate vY for each repetition
    vYMS = SimModelMS(23400, 1000, 5.225, dXi, vLambda, vV, vH, rand() * 10000000);
    
    % Save simulation in the cell array
    vY_listMS{rep} = vYMS;
        
    disp(rep);
    toc
end

%% Initial values for IV

for i = 1:250
one(i,:) = initial(vY_listMS{1,i}.IV(1,:));
two(i,:) = initial(vY_listMS{1,i}.IV(2,:));  
three(i,:) = initial(vY_listMS{1,i}.IV(3,:));  
four(i,:) = initial(vY_listMS{1,i}.IV(4,:));  
five(i,:) = initial(vY_listMS{1,i}.IV(5,:));  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)
%% Initial values for RV
for i = 1:250
one(i,:) = initial(vY_listMS{1,i}.RVs(1,:));
two(i,:) = initial(vY_listMS{1,i}.RVs(2,:));  
three(i,:) = initial(vY_listMS{1,i}.RVs(3,:));  
four(i,:) = initial(vY_listMS{1,i}.RVs(4,:));  
five(i,:) = initial(vY_listMS{1,i}.RVs(5,:));  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)

%% Initial values for RV
for i = 1:250
one(i,:) = initial(vY_listMS{1,i}.RVfive(1,:));
two(i,:) = initial(vY_listMS{1,i}.RVfive(2,:));  
three(i,:) = initial(vY_listMS{1,i}.RVfive(3,:));  
four(i,:) = initial(vY_listMS{1,i}.RVfive(4,:));  
five(i,:) = initial(vY_listMS{1,i}.RVfive(5,:));  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)


%% Initial values for RV
one = zeros(1,7); two = zeros(1,7); three = zeros(1,7); four =zeros(1,7); five = zeros(1,7);
for i = 1:100
one(i,:) = initialSVJ(vY_listJumpX9{1,i}{1,1}.BVfive,vY_listJumpX9{1,i}{1,1}.RVfive,vY_listJumpX9{1,i}{1,1}.SPV);
two(i,:) =  initialSVJ(vY_listJumpX9{1,i}{1,2}.BVfive,vY_listJumpX9{1,i}{1,2}.RVfive,vY_listJumpX9{1,i}{1,2}.SPV);
three(i,:) =  initialSVJ(vY_listJumpX9{1,i}{1,3}.BVfive,vY_listJumpX9{1,i}{1,3}.RVfive,vY_listJumpX9{1,i}{1,3}.SPV);
four(i,:) =  initialSVJ(vY_listJumpX9{1,i}{1,4}.BVfive,vY_listJumpX9{1,i}{1,4}.RVfive,vY_listJumpX9{1,i}{1,4}.SPV);  
five(i,:) =  initialSVJ(vY_listJumpX9{1,i}{1,5}.BVfive,vY_listJumpX9{1,i}{1,5}.RVfive,vY_listJumpX9{1,i}{1,5}.SPV);
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)


%% GMM estimation for IV
for rep = 1:250
    tic
    for i = 1:5
        EstimateIVMS{rep}{i} =  GMM_Estimator3(vY_listMS{1,rep}.IV(i,:), initial(vY_listMS{1,rep}.IV(i,:))', vell,'N');
    end
   toc
   disp(rep);
end 

%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0;
for j = 1:5
    for h = 1:250     % Run for each rep
    GMMXi(j,h) = EstimateIVMS{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimateIVMS{1, h}{1, j}(2);
    GMMv(j,h) = EstimateIVMS{1, h}{1, j}(3);
    GMMH(j,h) = EstimateIVMS{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]


%% GMM estimation for ppreaveraged RV
for rep = 1:250
    tic
    for i = 1:5
        EstimatesRVsMS{rep}{i} =  GMM_Estimator3(vY_listMS{1,rep}.RVs(i,:), initial(vY_listMS{1,rep}.RVs(i,:))', vell,'N');
    end
   toc
   disp(rep);
end 
%%
for j = 1:100
for h = 1:5
    for i = 1:100
    stdx(i) =  EstimatesjumpX9{1, i}{1, h}(4);
    end
    statistic(j,h) = (EstimatesjumpX9{1, j}{1, h}(4)   - vH(h))/ (std(stdx));
end
end

[f,xi] = ksdensity(statistic(:,1)); 
figure
plot(xi,f,'LineWidth',1,'LineStyle','--');
hold on 
[f,xi] = ksdensity(statistic(:,2));
plot(xi,f,'LineWidth',1,'LineStyle','--');
[f,xi] = ksdensity(statistic(:,3)); 
plot(xi,f,'LineWidth',1, 'LineStyle','--');
[f,xi] = ksdensity(statistic(:,4)); 
plot(xi,f,'LineWidth',1,'LineStyle','--');
[f,xi] = ksdensity(statistic(:,5)); 
plot(xi,f,'LineWidth',1,'LineStyle','--');
y = normpdf(xi,0,1);
plot(xi,y,'LineWidth',2)
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70','N(0,1)','Location','northwest')
xlim([-4 4])
hold off
%%
initial(vY_listMS{1,18}.RVs(1,:))

plot(vY_listMS{1,18}.RVs(1,:))
min(vY_listMS{1,18}.RVs(1,:))

%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0;
for j = 1:5
    for h = 1:250     % Run for each rep
    GMMXi(j,h) = EstimatesRVsMS{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVsMS{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVsMS{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVsMS{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%% GMM estimation for unCorreted RV 5 min
for rep = 19:250
    tic
    for i = 1:5
        RV = vY_listMS{1,rep}.RVfive(i,:);
        RV = RV( ~any( isnan( RV ) | isinf( RV ), 1 ));

        EstimatesRVuMS{rep}{i} =  GMM_Estimator3(RV, initial(RV)', vell,'N');
    end
   toc
   disp(rep); 
end  
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0;
for j = 1:5
    for h = 1:250     % Run for each rep
    GMMXi(j,h) = EstimatesRVuMS{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVuMS{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVuMS{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVuMS{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%% GMM estimation for approx RV 5 min
for rep = 1:250
    tic
    for i = 1:5
        RV = vY_listMS{1,rep}.RVfive(i,:);
        RV = RV( ~any( isnan( RV ) | isinf( RV ), 1 ));

        EstimatesRVcMS{rep}{i} =  GMM_Estimator3(RV, initial(RV)', vell,'R');
    end
   toc
   disp(rep); 
end  
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0;
for j = 1:5
    for h = 1:250     % Run for each rep
    GMMXi(j,h) = EstimatesRVcMS{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVcMS{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVcMS{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVcMS{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%% GMM estimation for Exact RV 5 min
for rep = 115:250
    tic
    for i = 1:5
        RV = vY_listMS{1,rep}.RVfive(i,:);
        RV = RV( ~any( isnan( RV ) | isinf( RV ), 1 ));

        EstimatesRVeMS{rep}{i} =  GMM_Estimator3(RV, initial(RV)', vell,'E');
    end
   toc
   disp(rep); 
end  
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0;
for j = 1:5
    for h = 1:250    % Run for each rep
    GMMXi(j,h) = EstimatesRVeMS{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesRVeMS{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesRVeMS{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesRVeMS{1, h}{1, j}(4);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]


%%
i = 0;
for h = vH
i = i +1;
%vYBolko = SimModelSimplejump(iN, 250, 0, 0.0225, 0.0704, 0.6048, 0.5, i+10);
vYBolko = SimModelSimplejump(2340, 250, 0, 0.0225, vLambda(i), vV(i), h, 1);
%plot((vYBolko.vY(1,1:iN*10)));
%hold on 
end

%hold off

%%

plot(log(vYBolko.BV(1,:)))
hold on
plot(log(vYBolko.RV(1,:)))
hold off

%%
rng(125);

% Create a cell array to store simulation from each repetition
%vY_listJumpX = cell(1, 1);

parfor rep = 1:50
    % Generate vY for each repetition
    %rng(100);
    tic
    for i = 5:5
    vY_listJumpX9{rep}{i} = SimModelSimplejump(iN/10,  2500, log(40000), dXi,vLambda(i) ,vV(i),vH(i),rand() * 10000000);
    end

    % Save simulation in the cell array
    toc
    disp(rep);
end

%%

vlambdaJ * (vMuJ^4 + 6*(vMuJ^2* vSigmaJ^2)+ 3*vSigmaJ^4)+ vlambdaJ^2 *(vMuJ^2 +vSigmaJ^2)^2 + 2* vXi *vlambdaJ*(vMuJ^2+vSigmaJ^2)

%%
vlambdaJt = 0.1; vMuJt = 0.3; vSigmaJt = 0.6501;
vlambdaJt * (vMuJt^4 + 6*(vMuJt^2* vSigmaJt^2)+ 3*vSigmaJt^4)
vlambdaJt^2 *(vMuJt^2 +vSigmaJt^2)^2 + 2* 0.225 *vlambdaJt*(vMuJt^2+vSigmaJt^2)
vlambdaJt^2 *(vMuJt^2 +vSigmaJt^2)^2+ 2* 0.225 *vlambdaJt*(vMuJt^2+vSigmaJt^2)
%%
6* (0.3^2*0.5^2)
%%
plot(( vY_listJumpX9{1,1}{1,5}.RVfive))
hold on
plot(( vY_listJumpX9{1,1}{1,5}.BVfive))
hold off
%%

for i = 2:250000
    dt= 1/2500;
            dN(i) = normrnd(10,(2))*poissrnd(0.08);
            dN2(i) = dN2(i-1) + dN(i);
end
plot(dN2)
%%
[testtest, stop2] = seqTestPos(vY_listApp{1,1}.SRV,0.9,1-0.5,0.9);
 sum(stop2 >= 1)/length(vY_listApp{1,1}.SRV)
%%
y = abs(normalize((vY_listApp{1,1}.SRV)));
x = 1:length(vY_listApp{1,1}.SRV);
yMark=NaN(1,length(y));
for i=1:1:length(vY_listApp{1,1}.SRV)
   
if stop2(i) >= 1
 yMark(i)=y(i);
end
end
plot(yMark,'->')
hold on
plot(repelem((-log(-log(0.9))),length(vY_listApp{1,1}.SRV)))
plot(y, '.')
hold off

%%
 for h = 1:50
ini{h} = initialSVJ(vY_listApp{1,h}.RVtrunc,vY_listApp{1,h}.RV, vY_listApp{1,h}.RD);
    iniplot(h) = ini{h}(1);
 end
plot(iniplot)
mean(iniplot)
%%
rng(1250);

% Create a cell array to store simulation from each repetition
%vY_listApp = cell(1, 1);

parfor rep = 1:10
    % Generate vY for each repetition
    %rng(100);
    vY = SimModelSigmajump(2340, 1000, 5, 0.0225, 0.0704, 0.6048, 0.5,rand() * 10000000);
    
    % Save simulation in the cell array
    vY_listApp{rep} = vY;
    
    disp(rep);
end

%%

plot((vY_listApp{10}.BV))
hold on 
plot((vY_listApp{10}.RV))
hold off

mean(vY_listApp{10}.RV)
mean(vY_listApp{10}.BV)


%%
plot(vY_listApp{1, 1}.vX(1:iN/10*iT/4),'--')
%%

%%
mean(diff(vY_listApp{1, 1}.vX))

%%

plot(vY_listApp{1, 1}.BV)
hold on
plot(vY_listApp{1, 1}.RV)
hold off
%%
PJ = 0;

for i=1:100
for j = 1:1
PJ(i,j) = max((vY_listJumpX9{1, i}{1,j}.BV /  vY_listJumpX9{1, i}{1,j}.RV),0);
end
end
surf(sort(PJ))


%%
PJ=0;PJmean=0;
for j = 1:5
for i=1:50

PJ(i,j) = max(  vY_listApp{1, j}.BV(1,i) / vY_listApp{1, j}.RV(1,i),0);


end
PJmean(j) = mean(PJ(:,j));

end
x = 0.01:0.01:0.99;
plot(x,PJmean,'-','MarkerSize',10)
ylabel('Jump proportion')
xlabel('Hurst exponent')
grid("on")

%%
tic
EstimatesBolkoApp =    GMMBolkoApp(vYBolko.IV,  [0.0225, 0.0704, 0.6048, 0.25], vell);
toc
EstimatesBolkoApp{1,1}


%%
plot((vY_listApp{1,1}.BV))
hold on 
plot((vY_listApp{1,1}.RV))
hold off
legend()

mean(vY_listApp{1,1}.BV)

%%

plot(max(1-vY_listApp{1,1}.BV./vY_listApp{1,1}.RV,0))

%%

sum(max(1-vY_listApp{1,1}.BV./vY_listApp{1,1}.RV,0))
%%
initials = cell(1);
for rep = 1:35
    for i = 1:5
initials{rep}{i} = initialSVJ(vY_listJumpX9{1,rep}{1,i}.BVfive,vY_listJumpX9{1,rep}{1,i}.RVfive,vY_listJumpX9{1,rep}{1,i}.SPV);
    end
end

mean([initials{rep}{1},initials{rep}{2},initials{rep}{2},initials{rep}{4},initials{rep}{5}])

%%
  
% for each repetition
parfor rep = 1:50
    tic
    for i = 5:5
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesjumpX9{rep}{i}, m_hatjumpX9{rep}{i}, WjumpX9{rep}{i}, jacobianjumpX9{rep}{i}] = GMM_EstimatorSVJ(vY_listJumpX9{1,rep}{1,i}.RVfive,vY_listJumpX9{1,rep}{1,i}.RVfive,0,0,0,initialSVJ(vY_listJumpX9{1,rep}{1,i}.BVfive,vY_listJumpX9{1,rep}{1,i}.RVfive,vY_listJumpX9{1,rep}{1,i}.RD),[0 1 2 3 5 7 10 15 20 25 30 50],'N');
    %estimatrix = NaN(5, 4);
    i
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
    end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
    toc
end

%%
plot(log(vY_listJumpX9{1,1}{1,5}.IV))
%%
initialSVJ(vY_listJumpX9{1,2}{1,5}.BVfive,vY_listJumpX9{1,2}{1,5}.RVfive,vY_listJumpX9{1,2}{1,5}.RD)
%%

betaer = [1.24 0.61 1.15 1.21 1.58 1.09 1.30 0.82 1.08 1.40 1.21 1.35 1 1.03 0.69 1.02 0.52 1.11 0.61 0.72 1 0.40 0.90 1.01 0.41 0.63 0.60 0.96 0.41 0.51];
betatest = fitlm(jumpcount,betaer)
plot(betatest)
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for h = 5:5
    for j = 1:50% Run for each rep
    GMMXi(h,j) = EstimatesjumpX9{1, j}{1,h}(1);
    GMMLambda(h,j) = EstimatesjumpX9{1, j}{1,h}(2);
    GMMv(h,j) = EstimatesjumpX9{1, j}{1,h}(3);
    GMMH(h,j) = EstimatesjumpX9{1, j}{1,h}(4);
    GMMSigma(h,j) = EstimatesjumpX9{1, j}{1,h}(5);
    GMMlambdaj(h,j) = EstimatesjumpX9{1, j}{1,h}(6);
    GMMmu(h,j) = EstimatesjumpX9{1, j}{1,h}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]

%%
plot(GMMH(1,:))
hold on
plot(GMMSigma)
plot(GMMlambdaj)
%%
surf(sort(GMMSigma,2))
%%
  
% for each repetition
parfor rep = 1:250
    tic
    for i = 1:5 
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesBolkoApp2Unew{rep}{i}, m_hatjumpXUnew{rep}{i}, WjumpXUnew{rep}{i}, jacobianjumpXUnew{rep}{i}] = GMM_EstimatorSVJ(vY_listJumpX2{1,rep}{1,i}.RVfive,vY_listJumpX2{1,rep}{1,i}.RVfive,0,0,0,initialSVJ(vY_listJumpX2{1,rep}{1,i}.BVfive,vY_listJumpX2{1,rep}{1,i}.RVfive,vY_listJumpX2{1,rep}{1,i}.SPV),[0 1 2 3 5 10 20 50 100 150],'N');
    %estimatrix = NaN(5, 4);
    i
    %for j = 1:5    
   %     estimatrix(j, 1:4) = Estimates{1,j};
    end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
    toc
end
%%
  
% for each repetition
parfor rep = 11:250
    tic
    for i = 1:5 
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesBolkoApp2{rep}{i}, m_hatjumpX{rep}{i}, WjumpX{rep}{i}, jacobianjumpX{rep}{i}] = GMM_EstimatorSVJ(vY_listJumpX2{1,rep}{1,i}.RVfive,vY_listJumpX2{1,rep}{1,i}.RVfive,0,0,0,initialSVJ(vY_listJumpX2{1,rep}{1,i}.BVfive,vY_listJumpX2{1,rep}{1,i}.RVfive,vY_listJumpX2{1,rep}{1,i}.SPV),[0 1 2 3 5 10 20 50 100 150],'R');
    %estimatrix = NaN(5, 4);
    i
    %for j = 1:5    
   %     estimatrix(j, 1:4) = Estimates{1,j};
    end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
    toc
end
%%
  
% for each repetition
parfor rep = 1:10
    tic
    for i = 1:5 
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesBolkoAppU1{rep}{i}, m_hatjumpXU1{rep}{i}, WjumpXU1{rep}{i}, jacobianjumpXU1{rep}{i}] = GMM_EstimatorSVJ(vY_listJumpX{1,rep}{1,i}.RVfive,vY_listJumpX2{1,rep}{1,i}.RVfive,0,0,0,initialSVJ(vY_listJumpX{1,rep}{1,i}.BVfive,vY_listJumpX{1,rep}{1,i}.RVfive,vY_listJumpX{1,rep}{1,i}.SPV),[0 1 2 3 5 10 20 50 100 150],'N');
    %estimatrix = NaN(5, 4);
    i
    %for j = 1:5    
   %     estimatrix(j, 1:4) = Estimates{1,j};
    end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
    toc
end

%%

% for each repetition
parfor rep = 1:10
    tic 
    for i = 1:5 
          iT = length(vY_listJumpX{1,rep}{1,i}.RVfive);
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesBolkoApp1{rep}{i}, m_hatjumpX1{rep}{i}, WjumpX1{rep}{i}, jacobianjumpX1{rep}{i}] = GMM_EstimatorSVJ(vY_listJumpX{1,rep}{1,i}.RVfive,vY_listJumpX2{1,rep}{1,i}.RVfive,0,0,0,initialSVJ(vY_listJumpX{1,rep}{1,i}.BVfive,vY_listJumpX{1,rep}{1,i}.RVfive,vY_listJumpX{1,rep}{1,i}.SPV),[0 1 2 3 5 10 20 50 100 150],'R');
    %estimatrix = NaN(5, 4);
    i
    %for j = 1:5    
   %     estimatrix(j, 1:4) = Estimates{1,j};
    end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
    toc
end 
%%
test =  Make_S_HACSVJ( [0.1914, 1.037, 0.2916, 0.0176, 0.0101, 0.0069, 0.0217, 0.0610, 0.0502, 0.0354, 0.0433], initialSVJ(vY_listJumpX{1,rep}{1,i}.BVfive,vY_listJumpX{1,rep}{1,i}.RVfive,vY_listJumpX{1,rep}{1,i}.SPV), [0 1 2 3 5 10 20 50 100 150], iT,vY_listJumpX{1,rep}{1,i}.RVfive,0,0,0,0,0,'R');

%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for h = 1:5  
    for j = 2:4 % Run for each rep
    GMMXi(h,j) = EstimatesBolkoApp1{1, j}{1,h}(1);
    GMMLambda(h,j) = EstimatesBolkoApp1{1, j}{1,h}(2);
    GMMv(h,j) = EstimatesBolkoApp1{1, j}{1,h}(3);
    GMMH(h,j) = EstimatesBolkoApp1{1, j}{1,h}(4);
    GMMSigma(h,j) = EstimatesBolkoApp1{1, j}{1,h}(5);
    GMMlambdaj(h,j) = EstimatesBolkoApp1{1, j}{1,h}(6);
    GMMmu(h,j) = EstimatesBolkoApp1{1, j}{1,h}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]


%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for h = 1:5  
    for j = 1:4 % Run for each rep
    GMMXi(h,j) = EstimatesBolkoAppU1{1, j}{1,h}(1);
    GMMLambda(h,j) = EstimatesBolkoAppU1{1, j}{1,h}(2);
    GMMv(h,j) = EstimatesBolkoAppU1{1, j}{1,h}(3);
    GMMH(h,j) = EstimatesBolkoAppU1{1, j}{1,h}(4);
    GMMSigma(h,j) = EstimatesBolkoAppU1{1, j}{1,h}(5);
    GMMlambdaj(h,j) = EstimatesBolkoAppU1{1, j}{1,h}(6);
    GMMmu(h,j) = EstimatesBolkoAppU1{1, j}{1,h}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]


%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for h = 1:5  
    for j = 1:250 % Run for each rep
    GMMXi(h,j) = EstimatesBolkoApp2U{1, j}{1,h}(1);
    GMMLambda(h,j) = EstimatesBolkoApp2U{1, j}{1,h}(2);
    GMMv(h,j) = EstimatesBolkoApp2U{1, j}{1,h}(3);
    GMMH(h,j) = EstimatesBolkoApp2U{1, j}{1,h}(4);
    GMMSigma(h,j) = EstimatesBolkoApp2U{1, j}{1,h}(5);  
    GMMlambdaj(h,j) = EstimatesBolkoApp2U{1, j}{1,h}(6);
    GMMmu(h,j) = EstimatesBolkoApp2U{1, j}{1,h}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for h = 1:5  
    for j = 1:250 % Run for each rep
    GMMXi(h,j) = EstimatesBolkoApp2{1, j}{1,h}(1);
    GMMLambda(h,j) = EstimatesBolkoApp2{1, j}{1,h}(2);
    GMMv(h,j) = EstimatesBolkoApp2{1, j}{1,h}(3);
    GMMH(h,j) = EstimatesBolkoApp2{1, j}{1,h}(4);
    GMMSigma(h,j) = EstimatesBolkoApp2{1, j}{1,h}(5);
    GMMlambdaj(h,j) = EstimatesBolkoApp2{1, j}{1,h}(6);
    GMMmu(h,j) = EstimatesBolkoApp2{1, j}{1,h}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]

%%

surf(sort(GMMH,2))

  %%
% for each repetition
parfor rep = 1:100
    tic
    for i = 5:5 
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesBolkoApp2naiv{rep}{i}, m_hatjumpXnaiv{rep}{i}, WjumpXnaiv{rep}{i}, jacobianjumpXnaiv{rep}{i}] = GMM_Estimator3(vY_listJumpX9{1,rep}{1,i}.RVfive,initial(vY_listJumpX9{1,rep}{1,i}.RVfive),[0 1 2 3 5 10 20 50],'R');
    %estimatrix = NaN(5, 4);
    i
    %for j = 1:5    
   %     estimatrix(j, 1:4) = Estimates{1,j};
    end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
    toc
end
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; 
for h = 5:5  
    for j = 1:100 % Run for each rep
    GMMXi(h,j) = EstimatesBolkoApp2naiv{1, j}{1,h}(1);
    GMMLambda(h,j) = EstimatesBolkoApp2naiv{1, j}{1,h}(2);
    GMMv(h,j) = EstimatesBolkoApp2naiv{1, j}{1,h}(3);
    GMMH(h,j) = EstimatesBolkoApp2naiv{1, j}{1,h}(4);

    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%

Year = [2014;2015;2016;2017;2018;2019;2020;2021;2022;2023;2023];
Month = [01;02;03;04;05;06;07;08;09;10;11;12];
Day = [01;02;03;04;05;06;07;08;09;10;11;12;13;14;15;16;17;18;19;20];
time = datetime(Year,Month,Day,'Format','eeee, MMMM d, y')

%%

T = readtable("AAPLYMD.xlsx");

Tnew = table2array(T{:,3})

%%

T{:,3}
%%
time = datetime(T{:,3},'Convertfrom','excel')
%%


%%
plot(1:2535,T(3,:))
%%
p_value = 0;
Sargan = cell(1);
for rep = 1:100
    for i =1:5
Sargan{1,rep}{1,i} = 2500 * m_hatjumpX9{1,rep}{1,i}'*(inv(WjumpX9{1,rep}{1,i}))*m_hatjumpX9{1,rep}{1,i};  % Sargan–Hansen statistic
df = 10-7+1;

p_value(rep,i) = 1 - chi2cdf(Sargan{1,rep}{1,i}, df);
Sargan2(rep,i) = Sargan{1,rep}{1,i};
%[empdist{1,rep}{1,i}, empdistx{1,rep}{1,i}] =ecdf(vY_listJumpX2{1,rep}{1,i}.RVfive);


    end
end

mean(p_value,1)
%%



%%

for i = 1:5
nSimulations = length(Sargan2(:,i));  
df = 4;  


jhs_simulated = (Sargan2(:,i));


alpha_levels = linspace(0.01, 1, 10000);  
critical_values = chi2inv(1 - alpha_levels, df);  


rr = zeros(size(alpha_levels)); 

for i = 1:length(alpha_levels)
   
    rr(i) = mean(jhs_simulated > critical_values(i));
end

plot(alpha_levels, rr, '--', 'LineWidth', 1);
hold on;


end

plot([0 1], [0 1], 'r', 'LineWidth', 1.5);  % Reference line y = x


xlabel('Nominal Level (\alpha)');
ylabel('Rejection Rate');
title('Rejection Rate vs. Nominal Level');
legend('H=0.05','H=0.1','H=0.3','H=0.5','H=0.7', 'Reference Line', 'Location', 'southeast');
grid on;

hold off

%%



hold on
plot(chi2pdf(Sargan{1,1}{1,1}, df))
hold off



%%




orderedp_values = sort(Sargan2);

normalized = (orderedp_values-min(min(orderedp_values))) ./ (max(max(orderedp_values)) -min(min(orderedp_values)));
%normalized = 1 ./ (1 + exp(orderedp_values));


figure;
plot(normalized, orderedp_values2,'o', 'MarkerSize', 6, 'LineWidth', 1.5); % P-P plot
hold on;


plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5); % 45-degree line


xlabel('');
ylabel('Empirical CDF');
title('P-P Plot with 45-Degree Reference Line');

grid on;
axis square; 
hold off;
%%

initialSVJ(vY_listJumpX{1,1}{1,1}.BV,vY_listJumpX{1,1}{1,1}.RV,0)

%%
  
% for each repetition
for rep = 1:10
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    EstimatesBolkoApp3{rep} = GMM_EstimatorSVJS(vY_listApp{1,rep}{1,1}.RV,vY_listApp{1,rep}{1,1}.RV,vY_listApp{1,rep}.RV,vY_listApp{1,rep}.RV,initialSVJS(vY_listApp{1,rep}.RV,vY_listApp{1,rep}.RV,0),[0,1,2,3,4,5,10,20,50],'N');
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
   % end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
end
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for h = 1:5  
    for j = 1:51 % Run for each rep
    GMMXi(h,j) = EstimatesBolkoApp2{1, j}{1,h}(1);
    GMMLambda(h,j) = EstimatesBolkoApp2{1, j}{1,h}(2);
    GMMv(h,j) = EstimatesBolkoApp2{1, j}{1,h}(3);
    GMMH(h,j) = EstimatesBolkoApp2{1, j}{1,h}(4);
    GMMSigma(h,j) = EstimatesBolkoApp2{1, j}{1,h}(5);
    GMMlambdaj(h,j) = EstimatesBolkoApp2{1, j}{1,h}(6);
    GMMmu(h,j) = EstimatesBolkoApp2{1, j}{1,h}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]

%%

plot(vY_listApp{1, 19}.vX)

%%

plot((vY_listJumpX{1, 1}{1,5}.RVtrunc))
hold on
plot((vY_listJumpX{1, 1}{1,5}.BVfive))
hold off
mean(vY_listApp{1, 1}.BV)
%%
%EstimatesJump1 = cell(1,1);
%EstimatesJump = cell(1,1);
%JACOBI = cell(1,1);
for rep = 1:250
tic
for i = 1:5
   [EstimatesJump1IV{rep}{i}, JACOBIIV{rep}{i}, HAC{rep}{i}] =  GMM_Estimator3(vY_listJumpX{1,rep}{1,i}.IV, initial(vY_listJumpX{1,rep}{1,i}.IV)', [0,1,2,3,5,20,50],'N');
end
toc
    disp(rep);
end
%% Initial values for IV

for i = 1:250
one(i,:) = initial(vY_listJumpX{1,i}{1,1}.IV);
two(i,:) = initial(vY_listJumpX{1,i}{1,2}.IV);  
three(i,:) = initial(vY_listJumpX{1,i}{1,3}.IV);  
four(i,:) = initial(vY_listJumpX{1,i}{1,4}.IV);  
five(i,:) = initial(vY_listJumpX{1,i}{1,5}.IV);  
end
mean(one,1)
mean(two,1)
mean(three,1)
mean(four,1)
mean(five,1)
std(one,1)
std(two,1)
std(three,1)
std(four,1)
std(five,1)

%%

for j = 1:4
for i = 1:100
  % seH =  JACOBI{1, i}{1, j}(8,4);

seH = 0;
seH = ((jacobianjumpX9{1, i}{1, j}.'*inv(WjumpX9{1, i}{1, j})*jacobianjumpX9{1, i}{1, j}));

seH2(i) = sqrt(seH(4,4));
density(i) = (EstimatesjumpX9{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
ksdensity(x)
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-4 4])
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma = 0; GMMLambdaJ = 0; GMMMu = 0;
for j = 1:5
    for h = 1:250    % Run for each rep
    GMMXi(j,h) = EstimatesJump1IV{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesJump1IV{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesJump1IV{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesJump1IV{1, h}{1, j}(4);
  
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]

%%
%EstimatesJump1 = cell(1,1);
%EstimatesJump = cell(1,1);
%JACOBI = cell(1,1);
parfor rep = 1:10
tic
for i = 1:5
  % [EstimatesJump1utest{rep}{i}] =  GMM_Estimator3(vY_listJumpX6{1,rep}{1,i}.RVtrunc, initial(vY_listJumpX6{1,rep}{1,i}.RVtrunc)', [0,1,2,3,5,20,50],'R');
     dXi = EstimatesJump1utest{rep}{i}(1);
    EstimatesJumputest{rep}{i} = GMM_EstimatorSVJ2s(vY_listJumpX6{1,rep}{1,i}.RVtrunc,vY_listJumpX6{1,rep}{1,i}.RVfive,vY_listJumpX6{1,rep}{1,i}.QPV,vY_listJumpX6{1,rep}{1,i}.SPV,vY_listJumpX6{1,rep}{1,i}.RD,dXi,initialSVJ(vY_listJumpX6{1,rep}{1,i}.RVtrunc,vY_listJumpX6{1,rep}{1,i}.RVfive,vY_listJumpX6{1,rep}{1,i}.RD),vell,'N');
end
toc
    disp(rep);
end
%%

seH = 0;
for j = 1:4
for i = 1:250
  % seH =  JACOBI{1, i}{1, j}(8,4);
seH = (inv(JACOBIu{1, i}{1, j}.'*inv(HACjumpu{1, i}{1, j})*JACOBIu{1, i}{1, j}))*iT;
seH2(i) = sqrt(seH(4,4));
density(i) = (EstimatesJump1u{1, i}{1, j}(4)-vH(j))/seH2(i);

end

ksdensity(density,'Bandwidth',0.05)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
ksdensity(x)
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-4 4])
%% Initial values for IV

for rep = 1:250
oneJ(rep,:) = initialSVJ(vY_listJumpX{1,rep}{1,1}.RVtrunc,vY_listJumpX{1,rep}{1,1}.RVfive,vY_listJumpX{1,rep}{1,1}.RD);
twoJ(rep,:) = initialSVJ(vY_listJumpX{1,rep}{1,2}.RVtrunc,vY_listJumpX{1,rep}{1,2}.RVfive,vY_listJumpX{1,rep}{1,2}.RD);  
threeJ(rep,:) = initialSVJ(vY_listJumpX{1,rep}{1,3}.RVtrunc,vY_listJumpX{1,rep}{1,3}.RVfive,vY_listJumpX{1,rep}{1,3}.RD);  
fourJ(rep,:) = initialSVJ(vY_listJumpX{1,rep}{1,4}.RVtrunc,vY_listJumpX{1,rep}{1,4}.RVfive,vY_listJumpX{1,rep}{1,4}.RD);  
fiveJ(rep,:) = initialSVJ(vY_listJumpX{1,rep}{1,5}.RVtrunc,vY_listJumpX{1,rep}{1,5}.RVfive,vY_listJumpX{1,rep}{1,5}.RD);  
end
mean(oneJ,1)
mean(twoJ,1)
mean(threeJ,1)
mean(fourJ,1)
mean(fiveJ,1)
std(oneJ,1)
std(twoJ,1)
std(threeJ,1)
std(fourJ,1)
std(fiveJ,1)
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma = 0; GMMLambdaJ = 0; GMMMu = 0;
for j = 1:5
    for h = 1:10   % Run for each rep
    GMMXi(j,h) = EstimatesJump1utest{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesJump1utest{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesJump1utest{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesJump1utest{1, h}{1, j}(4);
    GMMSigma(j,h) = EstimatesJumputest{1, h}{1, j}(1);
    GMMLambdaJ(j,h) = EstimatesJumputest{1, h}{1, j}(2);
    GMMMu(j,h) = EstimatesJumputest{1, h}{1, j}(3);

    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2), mean(GMMSigma,2), mean(GMMLambdaJ,2), mean(GMMMu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMLambdaJ,1,2), std(GMMMu,1,2)]

%%
%EstimatesJump1 = cell(1,1);
%EstimatesJump = cell(1,1);
%JACOBI = cell(1,1);
for rep = 1:250
tic
for i = 1:5
   [EstimatesJump1{rep}{i}, JACOBI{rep}{i}] =  GMM_Estimator3(vY_listJumpX{1,rep}{1,i}.RVtrunc, initial(vY_listJumpX{1,rep}{1,i}.RVtrunc)', [0,1,2,3,5,20,50],'R');
     dXi = EstimatesJump1{rep}{i}(1);
    EstimatesJump{rep}{i} = GMM_EstimatorSVJ2s(vY_listJumpX{1,rep}{1,i}.RVtrunc,vY_listJumpX{1,rep}{1,i}.RVfive,vY_listJumpX{1,rep}{1,i}.QPV,vY_listJumpX{1,rep}{1,i}.SPV,vY_listJumpX{1,rep}{1,i}.RD,dXi,initialSVJ(vY_listJumpX{1,rep}{1,i}.RVtrunc,vY_listJumpX{1,rep}{1,i}.RVfive,vY_listJumpX{1,rep}{1,i}.RD),vell,'N');
end
toc
    disp(rep);
end

%%
initialSVJ(vY_listApp{1,1}.BV,vY_listApp{1,1}.RV,vY_listApp{1,1}.RD)


%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma = 0; GMMLambdaJ = 0; GMMMu = 0;
for j = 1:5
    for h = 1:250    % Run for each rep
    GMMXi(j,h) = EstimatesJump1{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesJump1{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesJump1{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesJump1{1, h}{1, j}(4);
    GMMSigma(j,h) = EstimatesJump{1, h}{1, j}(1);
    GMMLambdaJ(j,h) = EstimatesJump{1, h}{1, j}(2);
    GMMMu(j,h) = EstimatesJump{1, h}{1, j}(3);

    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2), mean(GMMSigma,2), mean(GMMLambdaJ,2), mean(GMMMu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMLambdaJ,1,2), std(GMMMu,1,2)]

%%
   reps = 1:50;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 for h = reps
    GMMHApp1(h) = EstimatesJump1{h}(1);
    GMMHApp2(h) = EstimatesJump1{h}(2);
    GMMHApp3(h) = EstimatesJump1{h}(3);
    GMMHApp4(h) = EstimatesJump1{h}(4);
    GMMHApp5(h) = EstimatesJump{h}(1);
    GMMHApp6(h) = EstimatesJump{h}(2);
    GMMHApp7(h) = EstimatesJump{h}(3);
    ini{h} = initialSVJ(vY_listJumpX{1,h}.BVfive,vY_listJumpX{1,h}.RVfive, vY_listJumpX{1,h}.RD);
    iniplot(h) = ini{h}(1);
     iniplot2(h) = ini{h}(2);
     iniplot3(h) = ini{h}(3);
     iniplot4(h) = ini{h}(4);
     iniplot5(h) = ini{h}(5);
     iniplot6(h) = ini{h}(6);
     iniplot7(h) = ini{h}(7);
 end 
xi = mean(GMMHApp1(reps))
lambda = mean(GMMHApp2(reps))
v = mean(GMMHApp3(reps))
H = mean(GMMHApp4(reps))
SigmaJ = mean(GMMHApp5(reps))
LambdaJ = mean(GMMHApp6(reps))
MuJ = mean(GMMHApp7(reps))


%%
   reps = 1:250;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 for h = reps
    GMMHApp1(h) = EstimatesRVu{h}{1,5}(1);
  

 end 
xi = mean(GMMHApp1(reps))

plot(GMMHApp1(reps))
%%
plot(GMMHApp1(reps))
hold on
plot(iniplot(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp2(reps))
hold on
plot(iniplot2(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp3(reps))
hold on
plot(iniplot3(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp4(reps))
hold on
plot(iniplot4(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp5(reps))
hold on
plot(iniplot5(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp6(reps))
hold on
plot(iniplot6(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp7(reps))
hold on
plot(iniplot7(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
 for h = 1:99
ini{h} = initialSVJ(vY_listApp{1,h}.RVtrunc,vY_listApp{1,h}.RV, vY_listApp{1,h}.RD);
    iniplot(h) = ini{h}(1);
 end
plot(iniplot)
%%
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
 for h = 1:8
    GMMHApp1(h) = EstimatesRVu{1, h}{1, 1}(1);
    GMMHApp2(h) = EstimatesRVu{1, h}{1, 1}(2);
    GMMHApp3(h) = EstimatesRVu{1, h}{1, 1}(3);
    GMMHApp4(h) = EstimatesRVu{1, h}{1, 1}(4);
    GMMHApp5(h) = EstimatesJump{h}(1);
    GMMHApp6(h) = EstimatesJump{h}(2);
    GMMHApp7(h) = EstimatesJump{h}(3);
        ini{h} = initialSVJ(vY_listApp{1,h}.BV,vY_listApp{1,h}.RV, vY_listApp{1,h}.RD);
    iniplot(h) = ini{h}(1);
     iniplot2(h) = ini{h}(2);
     iniplot3(h) = ini{h}(3);
     iniplot4(h) = ini{h}(4);
     iniplot5(h) = ini{h}(5);
     iniplot6(h) = ini{h}(6);
     iniplot7(h) = ini{h}(7);

    
 end 
xi = mean(GMMHApp1(1:8),"omitmissing")
lambda = mean(GMMHApp2(1:8),"omitmissing")
v = mean(GMMHApp3(1:8),"omitmissing")
H = mean(GMMHApp4(1:8),"omitmissing")
SigmaJ = mean(GMMHApp5(1:10),"omitmissing")
LambdaJ = mean(GMMHApp6(1:10),"omitmissing")
MuJ = mean(GMMHApp7(1:10),"omitmissing")

%%
mean2=0;
for rep = 1:8
ini{rep} = initialSVJ(vY_listApp{1,rep}.BV,vY_listApp{1,rep}.RV);
mean2 = mean2+ ini{rep}(5);
iniplot(rep) = ini{rep}(4);
end
mean2 = mean2/8

%%

plot(iniplot)
hold on
plot(GMMHApp4)
hold off
legend('initials','Estimates')

 %%
        reps = 1:10;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
 for h = reps
    GMMHApp1(h) = EstimatesBolkoApp{1, h}(1);
    GMMHApp2(h) = EstimatesBolkoApp{1, h}(2);
    GMMHApp3(h) = EstimatesBolkoApp{1, h}(3);
    GMMHApp4(h) = EstimatesBolkoApp{1, h}(4);
    GMMHApp5(h) = EstimatesBolkoApp{1, h}(5);
    GMMHApp6(h) = EstimatesBolkoApp{1, h}(6);
    GMMHApp7(h) = EstimatesBolkoApp{1, h}(7);
            ini{h} = initialSVJ(vY_listApp{1,h}.RVtrunc,vY_listApp{1,h}.RV, vY_listApp{1,h}.RD);
    iniplot(h) = ini{h}(1);
     iniplot2(h) = ini{h}(2);
     iniplot3(h) = ini{h}(3);
     iniplot4(h) = ini{h}(4);
     iniplot5(h) = ini{h}(5);
     iniplot6(h) = ini{h}(6);
     iniplot7(h) = ini{h}(7);
    
 end 
xi = mean(GMMHApp1(reps),"omitmissing")
lambda = mean(GMMHApp2(reps),"omitmissing")
v = mean(GMMHApp3(reps),"omitmissing")
H = mean(GMMHApp4(reps),"omitmissing")
SigmaJ = mean(GMMHApp5(reps),"omitmissing")
LambdaJ = mean(GMMHApp6(reps),"omitmissing")
MuJ = mean(GMMHApp7(reps),"omitmissing")
%%
plot(GMMHApp1(reps))
hold on
plot(iniplot(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp2(reps))
hold on
plot(iniplot2(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp3(reps))
hold on
plot(iniplot3(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp4(reps))
hold on
plot(iniplot4(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp5(reps))
hold on
plot(iniplot5(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp6(reps))
hold on
plot(iniplot6(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp7(reps))
hold on
plot(iniplot7(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%


plot(vY.vY(1, 1:(iN*5)), 'Color', 'B');
hold on;
plot(vY.vY(2, 1:(iN*5)), 'Color', 'R');
plot(vY.vY(3, 1:(iN*5)), 'Color', 'Y');
plot(vY.vY(4, 1:(iN*5)), 'Color', 'B');
plot(vY.vY(5, 1:(iN*5)), 'Color', 'G');
hold off;
title('vY');
legend('1', '2', '3', '4', '5');

%%

plot(log(vY.IV(1, 1:250)), 'Color', 'blue');
hold on;
plot(log(vY.IV(2, 1:250)), 'Color', 'red');
plot(log(vY.IV(3, 1:250)), 'Color', 'yellow');
plot(log(vY.IV(4, 1:250)), 'Color', 'black');
plot(log(vY.IV(5, 1:250)), 'Color', 'green');
hold off;
title('log(IV)');





%%

for i = [.1 .2 .3 .4 .5 .6 .7 .8 .9]
rng(1)
X = fractional_levy_motion(i,23400,1,1,1, 0, 0);
% Plot the Lévy motion
t = linspace(0, 1, 23400);
plot(t, X);
hold on
end
hold off
xlabel('Time');
ylabel('Position');
title('Simulated Lévy Motion');

%%

[X Y] = levy_driven_ou(1,1,1,1,1/2000,1);
plot(Y)




%%

i= 0
for h=vH
    i = 1 + i
vY(i) = FLDOU(iN, iT, 0, 0.0225, 0.05, 0.5, h, 1.25, 1, 0.1, 0, 125);
end
%%
plot(vY(1).vY)
hold on
plot(vY(2).vY)
plot(vY(3).vY)
plot(vY(4).vY)
plot(vY(5).vY)
hold off
%%
plot(log(vY(1).IV))
hold on
plot(log(vY(2).IV))
plot(log(vY(3).IV))
plot(log(vY(4).IV))
plot(log(vY(5).IV))

%%
    
GMMtest = GMM_Estimator3(vY(5).IV,[0,0,0,0],vell)

%%  
GMMtest





%%

plot((real(X)))







%%
%%
i = 0;
for h = vH
i = i +1;

X = SimModelLFSMOU(iN, 10, 0, 0.0225, vLambda(i), vV(i), vH(i), 1.8, 125);

plot(X.vY)
hold on
end
hold off
%%
for i = 1:5
%vYBolko = SimModelSimplejump(iN, 250, 0, 0.0225, 0.0704, 0.6048, h, 1);
vYBolko = SimModelSimplejump(iN, 250, 0, 0.0225, vLambda(i), vV(i), vH(i), 12);
plot((vYBolko.vY(1,:)));
hold on 
end
hold off
%%
i= 0
%vY = cell(5)
for h=vH
    i = 1 + i
vY{i} =SimModelLFSMOU(iN/10, 50, 0, 0.0225,0.2 , 1, vH(i), 2, 1);
end

plot(vY{1}.vY)
hold on
plot(vY{2}.vY)
plot(vY{3}.vY)
plot(vY{4}.vY)
plot(vY{5}.vY)
hold off

%%


lambda = 0.12;
pois = poissrnd(lambda,1000,1);
uni = unifrnd(0,1,1000,1);
compound = pois.*uni;
plot(cumsum(compound))

%%




%%

T = readtable('BABV.csv');
T = table2array(T);
T = sqrt(T);
plot(T)
mean((T))

AXPini = initial(T);
%%
x = ["AAPLBV.csv","AMGNBV.csv","AMZNBV.csv",'AXPBV.csv','BABV.csv','CATBV.csv','CRMBV.csv','CSCOBV.csv','CVXBV.csv','DISBV.csv','DOWBV.csv','GSBV.csv','HDBV.csv','HONBV.csv','IBMBV.csv','INTCBV.csv','JNJBV.csv','JPMBV.csv','KOBV.csv','MCDBV.csv','MMMBV.csv','MRKBV.csv','MSFTBV.csv','NKEBV.csv','PGBV.csv','TRVBV.csv','UNHBV.csv','VBV.csv','VZBV.csv','WMTBV.csv'];
y = ["AAPLRV.csv","AMGNRV.csv","AMZNRV.csv",'AXPRV.csv','BARV.csv','CATRV.csv','CRMRV.csv','CSCORV.csv','CVXRV.csv','DISRV.csv','DOWRV.csv','GSRV.csv','HDRV.csv','HONRV.csv','IBMRV.csv','INTCRV.csv','JNJRV.csv','JPMRV.csv','KORV.csv','MCDRV.csv','MMMRV.csv','MRKRV.csv','MSFTRV.csv','NKERV.csv','PGRV.csv','TRVRV.csv','UNHRV.csv','VRV.csv','VZRV.csv','WMTRV.csv'];
z= ["AAPLSRV.csv","AMGNSRV.csv","AMZNSRV.csv",'AXPSRV.csv','BASRV.csv','CATSRV.csv','CRMSRV.csv','CSCOSRV.csv','CVXSRV.csv','DISSRV.csv','DOWSRV.csv','GSSRV.csv','HDSRV.csv','HONSRV.csv','IBMSRV.csv','INTCSRV.csv','JNJSRV.csv','JPMSRV.csv','KOSRV.csv','MCDSRV.csv','MMMSRV.csv','MRKSRV.csv','MSFTSRV.csv','NKESRV.csv','PGSRV.csv','TRVSRV.csv','UNHSRV.csv','VSRV.csv','VZSRV.csv','WMTSRV.csv'];
%x = ["AAPLBVs.csv","AMGNBVs.csv","AMZNBVs.csv",'AXPBVs.csv','BABVs.csv','CATBVs.csv','CRMBVs.csv','CSCOBVs.csv','CVXBVs.csv','DISBVs.csv','DOWBVs.csv','GSBVs.csv','HDBVs.csv','HONBVs.csv','IBMBVs.csv','INTCBVs.csv','JNJBVs.csv','JPMBVs.csv','KOBVs.csv','MCDBVs.csv','MMMBVs.csv','MRKBVs.csv','MSFTBVs.csv','NKEBVs.csv','PGBVs.csv','TRVBVs.csv','UNHBVs.csv','VBVs.csv','VZBVs.csv','WMTBVs.csv'];
%y = ["AAPLRVs.csv","AMGNRVs.csv","AMZNRVs.csv",'AXPRVs.csv','BARVs.csv','CATRVs.csv','CRMRVs.csv','CSCORVs.csv','CVXRVs.csv','DISRVs.csv','DOWRVs.csv','GSRVs.csv','HDRVs.csv','HONRVs.csv','IBMRVs.csv','INTCRVs.csv','JNJRVs.csv','JPMRVs.csv','KORVs.csv','MCDRVs.csv','MMMRVs.csv','MRKRVs.csv','MSFTRVs.csv','NKERVs.csv','PGRVs.csv','TRVRVs.csv','UNHRVs.csv','VRVs.csv','VZRVs.csv','WMTRVs.csv'];
N = numel(x);
T = cell(N,1);
U = cell(N,1);
ini = cell(N,1);
for k = 1:N
    
T{k} = (table2array(readtable(x{k})))';
U{k} = (table2array(readtable(y{k})))';
Z{k} = (table2array(readtable(z{k})))';
meanT(k)=mean(U{k});
stdU(k) = std(U{k});
ini{k} = initial(T{k});
end 
mean(meanT*78)
mean(stdU)
%%
for k = 1:N

meanT(k)=mean(U{k});
ini{k} = initialSVJ(T{k},U{k},U{k});
end 
mean(meanT)


%%
plot(78*(U{1}))
hold on 
plot(sqrt(U{1}))
hold off
%%

Utotal = zeros(length(U{1}),1);
Ttotal = zeros(length(U{1}),1);
for k = 1:30
    Unew = interp1((1:numel(U{k})), U{k}, linspace(1, numel(U{k}), numel(U{1})), 'linear')';
 k
Utotal = Utotal + Unew;
    Tnew = interp1((1:numel(T{k})), T{k}, linspace(1, numel(T{k}), numel(U{1})), 'linear')';
 
Ttotal = Ttotal + Tnew;

end

plot(Utotal)
%%


X = 0;
y = 78/30.*Ttotal';
x = 78/30.*Utotal';
X = [ones(length(x),1) x'];
b1 = X\y';
[beta,Sigma] = mvregress(x',y');
%yCalc1 = b1*x';
yCalc2 = X*b1;
plot(x,y,'r.')
hold on
%plot(y,yCalc1)
plot(x,yCalc2,'b-')
xlabel('BV')
ylabel('RV')
str = {sprintf('y = %.2f+%.2fx',Sigma,beta)};
title(str)
grid on
hline = refline(1);
hline.Color = 'k';
hline.LineWidth = 2;
hline.LineStyle = '--';
%plot(1:max(x),'.');
legend('Data','OLS','Reference line','Location','NW');
xlim([0 0.1])
ylim([0 0.1])
hold off
%%
for k = 1:30

meanT(k)=mean(78*U{k});

end 
mean(meanT)
plot(78*U{1})
meanT

%%
for k= 1:30
ini{k}(1)
end
%%

X = 0;
y = 78.*T{1};
x = 78.*U{1};
X = [ones(length(x),1) x'];
b1 = X\y';
[beta,Sigma] = mvregress(x',y');
%yCalc1 = b1*x';
yCalc2 = X*b1;
plot(x,y,'r.')
hold on
%plot(y,yCalc1)
plot(x,yCalc2,'b-')
xlabel('BV')
ylabel('RV')
str = {sprintf('y = %.2f+%.2fx',Sigma,beta)};
title(str)
grid on
hline = refline(1);
hline.Color = 'k';
hline.LineWidth = 2;
hline.LineStyle = '--';
%plot(1:max(x),'.');
legend('Data','OLS','Reference line','Location','NW');
xlim([0 10e-2])
ylim([0 10e-2])
hold off


%%

X = 0;
y = (betaer);
x = jumpcount;
X = [ones(length(x),1) x'];
b1 = X\y';
[beta,Sigma,E,CovB,logL] = mvregress(x',y');
%yCalc1 = b1*x';
yCalc2 = X*b1;
plot(x,y,'r.')
hold on
%plot(y,yCalc1)
plot(x,yCalc2,'b-')
xlabel('Jumpcount')
ylabel(['Beta'])
str = {sprintf('y = %.2f+%.2fx',Sigma,beta)};
title(str)
grid on
%plot(1:max(x),'.');
legend('Data','OLS','Reference line','Location','NW');

hold off
Errors = sqrt(diag(CovB))


%%

fitlm(jumpcount,betaer)
%%
meanPV=0;
PV = 0
for k = 1:30
PV =max(1- T{k}./U{k},0);
meanPV(k) = mean(PV);
end
plot((meanPV))


%%
%gmmemperical = cell(N,1);
parfor k = 1:30   
[gmmempericaltest{k}, m_hatemp{k}, mS_HACemp{k}, jacobianemp{k}] = GMM_Estimator3(real(78*U{k}),initial(78*U{k})',vell,'R');
k
end

%%

for k = 1:30
SOI{k}= 78*U{k}(3:end) - 2*78*U{k}(2:end-1)+78*U{k}(1:end-2);
end
%%
for k = 1:30
Sarganemp(k) = length(real(U{k})) * m_hatemp{1,k}'*(inv(mS_HACemp{1,k}))*m_hatemp{1,k};  % Sargan–Hansen statistic
df = 10-7+1;


p_valueemp(k) = 1 - chi2cdf(Sarganemp(k), df);

   
end
p_valueemp
mean(p_valueemp)
std(p_valueemp)
%%

%%
probplot(p_valueemp)
%%
%gmmemperical = cell(N,1);
for k = 1:30
%gmmemperical{k} = GMM_Estimator3(real(U{k}(78*U{17})),initial(U{k})',vell,'R');
end

%%

plot(78*U{k}(78*U{17}<=2))
%%
%gmmemperical = cell(N,1);  
for k = 1  :30
[gmmempericaljumpX{k}, m_hatjumpXemp{k}, WjumpXemp{k}, jacobianjumpXemp{k}] = GMM_EstimatorSVJ(78*real(U{k}),78*real(U{k}),0,0,0,initialSVJ(78*T{k},78*U{k},0)',[0,1,2,3,5,10,20,50,100, 150],'R');
k
end

%%

mu_Y = @(mu, sigma) sigma * sqrt(2/pi) * exp(-mu^2 / (2 * sigma^2)) + mu * (1 - 2 * normcdf(-mu/sigma));
test = mu_Y(0,0.1)
%%
%gmmemperical = cell(N,1);
parfor k = 1  :30
[gmmempericalMS{k}] = GMM_Estimator3(78*real(T{k}),initial(78*T{k})',vell,'R');
k
end
%%
%gmmemperical = cell(N,1);
parfor k = 1:30
    tic
[gmmempericaljumpY{k}, m_hatjumpYemp{k}, WjumpYemp{k}, jacobianjumpYemp{k}] = GMM_EstimatorSVJS(78*real(U{k}),78*real(U{k}),0,0,initialSVJS(78*U{k   },78*U{k},Z{k})',[0,1,2,3,4,5,10,20,30,50],'N');
k       
toc
end 

%%
k=12;
initialSVJS(78*T{k   },78*U{k},(Z{k}))
%%
%gmmemperical = cell(N,1);
for k = 1:30
gmmempiricaljumpXold{k} = GMM_EstimatorSVJ2s(x) 
k   
end


%%
tic
E_IV_IV_ell(1,0.1,0.001,3,0.00825)
toc
%%
%gmmemperical = cell(N,1);
parfor k = 1  :30
[gmmempericaljumpX{k}, m_hatjumpXemp{k}, WjumpXemp{k}, jacobianjumpXemp{k}] = GMM_EstimatorSVJ(78*real(U{k}),78*real(U{k}),0,0,0,initialSVJ(78*T{k},78*U{k},0)',[0,1,2,3,5,10,20,50,100],'R');
k
end

%%

mean(JPemp)
mean(jumpcount)
%%
%gmmemperical = cell(N,1);
parfor k = 1  :30
[gmmempericaljumpX2step{k}] = GMM_Estimator3(78*real(U{k}),initial(78*T{k})',vell,'R');
k
end
%%
for k = 1:30
Sarganemp(k) = length(real(U{k})) * m_hatjumpXemp{1,k}'*(inv(WjumpXemp{1,k}))*m_hatjumpXemp{1,k};  % Sargan–Hansen statistic
df = 10-7+1;


p_valueemp(k) = 1 - chi2cdf(Sarganemp(k), df);

   
end
p_valueemp
mean(p_valueemp)
%%

 %%
 ini = ones(30,7);
for k = 1:30
ini(k,:) = initialSVJ(78*T{k},78*U{k},0);

end
mean(ini,1)
%%
for k = 1:30
Sarganemp(k) = length(real(78*U{k})) * m_hatemp{1,k}'*(inv(WjumpXemp{1,k}))*m_hatjumpXemp{1,k};  % Sargan–Hansen statistic
df = 10-7+1;


p_valueemp(k) = 1 - chi2cdf(Sarganemp(k), df);

   
end
p_valueemp
mean(p_valueemp)

%%
probplot(p_valueemp)
 %%
for i = 17:17
plot(78*U{i}(U{i}<= 10*std(U{i})))
hold on
end
hold off
legend()

%%

for k = 1:30
    gmmemperical{k}
end

%%

plot3(averageMu,averagesigma,averageLambda2,'o')

%%
sorted = sort([averageMu; averagesigma],2);
plot3(sorted,averageLambda2)


%%
for j = 1:30

n = length(78*U{j});
mfilter = zeros(n, 1);  
window = 50;  
mad_window = 30;   

for i = 1:n
    start = max(1, i - floor(window/2));
    endIdx = min(n, i + floor(window/2));
    windowData = 78*U{j}(start:endIdx);
    windowData(i - start + 1) = [];  
    X = mean(windowData);
    mad = mean(abs(windowData - X));
    mfilter(i) = mad_window * mad;
end
temp = U{j};
U{j}= U{j}(78*temp<=mfilter');
T{j}= T{j}(78*temp<=mfilter');
filtercount(j) = length(temp)- length(U{j})

end
plot(78*U{17})

%%
mean(filtercount)
%%

%%

test = 78*U{17}<=mfilter;
%%

for k = 1:30

[test2 test,Jumptestemp] = teststat(real(SOI{k}), 0.9, 1-gmmempericaltest{k}(4), 0.9);
pvalueemp = 1- gevinv(Jumptestemp,0,1,0);
Jumptestemp
test
test2
end
pvalueemp

%%
%%

for k = 1:30

[Jumptestemp(k) test, Rn] = teststat(vY_listJumpsigma{1,k}{1,1}.RV, 0.9, gmmempericaltest{k}(4), 0.95);
pvalueemp(k) = 1- gevcdf(Rn,0,1,0);
Rn
end
pvalueemp

%%
-log(-log(0.95))
%%

empaapl(:) = seqTestPos(real(Z{k}), 0.9, gmmempericaltest{k}(4), 0.95)

%%
jumpcount =0;
for k = 1:30
[testtest, stop2] = seqTestPos(SOI{k}, 0.9, 1-gmmempericaltest{k}(4), 0.95);
 sum(stop2 >= 1)/length(SOI{k})
jumpcount(k)= sum(stop2 >= 1)
end
%%
y = cell(30,1); ymark2 = cell(30,1);
for i = 1
y{i} = abs(normalize((SOI{i})));
x = 1:length(SOI{i});
yMark=NaN(1,length(y));
[testtest, stop2] = seqTestPos(SOI{i}, 0.9, 1-gmmempericaljumpY{i}(4), 0.9);
y2 = y{i};
for j=1:1:length(SOI{i})
    if stop2(j) >= 1
    yMark(j)=y{i}(j);
    end
    
end
ymark2{i} = yMark(:);
plot(time(1:length(SOI{i}))',yMark,'->')
hold on
plot(repelem((-log(-log(0.9))),length(SOI{1})))
plot(y2,'.')
axis tight
%hold off
end
hold off
%%
surfing = ones(length(SOI{1}),30);
surfing2 = ones(length(SOI{1}),30);
for i = 1:30

surfing(1:length(y{i}),i) = y{i};
surfing2(1:length(ymark2{i}),i) = ymark2{i};
end

%%
test = repelem(ZMax,length(SOI{1}),30);
%%
figure; h=surf(1:30,time(3:end),surfing,'FaceAlpha',0.5);
 view([45,45]); colormap(summer); cb=colorbar; 

 h.EdgeColor='none'; shading('interp'); 
hold on
ZMax=max(-log(-log(0.95)));
xticks(1:30)
xticklabels(labels)

for j = 1:1
    
    h2 = surf(1:30,time(3:end),repelem(ZMax,length(time(3:end)),30),'SpecularExponent',1,...
    'SpecularStrength',1,...
    'DiffuseStrength',1,...
    'AmbientStrength',0.1,...
    'FaceColor',[0.5 0.5 .5],...
    'FaceAlpha',0.8,...
    'AlignVertexCenters','on',...
    'LineWidth',0.1,...
    'EdgeAlpha',0);
 %   alpha 0.5
    axis tight
    
%plot3([1 30],[1 length(SOI{j})],repelem(ZMax,2))

end
hold off
%%

figure; h=surf(1:30,time(3:end),surfing,'FaceAlpha',0.5);
 view([0,90]); colormap(hot); cb=colorbar; 
 %%

 contourf(1:30,time(3:end),surfing)
%%

image(1:30,time(3:end),100*(surfing))
colormap(autumn)
cb=colorbar
set(gca,'XTick',1:30,'XTickLabel',labels)
%%
ynew = cell(1);
counter = 1;
for i = 1:30
    counter = 1;
for k = 13 : 10 : length(y{i})-13
    k2 = k+10;
    theMeans(counter) = mean(y{i}(k-12 :k+13));
    counter = counter + 1;
    
end
ynew{i} = theMeans(1:counter);
end

%%
surfingnew = ones(252,30);

for i = 1:30

surfingnew(1:length(ynew{i}),i) = ynew{i};

end

%%
image(1:30,1:252,surfingnew)
colormap(hot(8))
set(gca,'XTick',1:30,'XTickLabel',labels)

%%
timenew = time(1:10:length(time)-40)
%%
labels = ["AAPL","AMGN","AMZN",'AXP','BA','CAT','CRM','CSCO','CVX','DIS','DOW','GSB','HDB','HON','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PG','TRV','UNH','V','VZ','WMT'];


%%
for i = 1:30
gmmemp(i,:) = gmmempericaltest{i};
gmmemp(i,2)= gmmemp(i,2)*100
end
gmmemp
%%
y{j}(:)
%%
averageXi=0;averageLambda=0;averagev=0;averageH=0;
for k = 1:N
averageXi(k) = gmmempericaltest{k}(1);
averageLambda(k) =100* gmmempericaltest{k}(2);
averagev(k) = gmmempericaltest{k}(3);
averageH(k) = gmmempericaltest{k}(4);
end
averageXi2 = mean(averageXi)
averageLambda2 = mean(averageLambda)
averagev2 = mean(averagev)
averageH2 = mean(averageH)
%%
averageXi=0;averageLambda=0;averagev=0;averageH=0;
for k = 1:N
averageXi(k) = gmmempericalMS{k}(1);
averageLambda(k) =100* gmmempericalMS{k}(2);
averagev(k) = gmmempericalMS{k}(3);
averageH(k) = gmmempericalMS{k}(4);
end
averageXi2 = mean(averageXi)
averageLambda2 = mean(averageLambda)
averagev2 = mean(averagev)
averageH2 = mean(averageH)
%%

matrixxxx = [averageXi; averageLambda; averagev; averageH]'

std(matrixxxx)

%%
averageXi=0;averageLambda=0;averagev=0;averageHj=0; averagesigma = 0; averageLambda = 0;averageMu=0;
for k = 1:30
averageXi(k) = gmmempericaljumpX{k}(1);
averageLambda(k) =100* gmmempericaljumpX{k}(2);
averagev(k) = gmmempericaljumpX{k}(3);
averageHj(k) = gmmempericaljumpX{k}(4);
averagesigma(k) = gmmempericaljumpX{k}(5);
averageLambda2(k) = gmmempericaljumpX{k}(6);
averageMu(k) = gmmempericaljumpX{k}(7);
end

averageXi2 = mean(averageXi)
averageLambda3 = mean(averageLambda)
averagev2 = mean(averagev)
averageH2 = mean(averageHj)
averagesigma2 = mean(averagesigma)
averageLambda22 = mean(averageLambda2)
averageMu2   = mean(averageMu)

%%
averageXi=0;averageLambda=0;averagev=0;averageHj=0; averagesigma = 0; averageLambda = 0;averageMu=0;
for k = 1:30
averageXi(k) = gmmempericaljumpY{k}(1);
averageLambda(k) =100* gmmempericaljumpY{k}(2);
averagev(k) = gmmempericaljumpY{k}(3);
averageHj(k) = gmmempericaljumpY{k}(4);
averagesigma(k) = gmmempericaljumpY{k}(5);
averageLambda2(k) = gmmempericaljumpY{k}(6);
averageMu(k) = gmmempericaljumpY{k}(7);
end

averageXi2 = mean(averageXi)
averageLambda3 = mean(averageLambda)
averagev2 = mean(averagev)
averageH2 = mean(averageHj)
averagesigma2 = mean(averagesigma)
averageLambda22 = mean(averageLambda2)
averageMu2   = mean(averageMu)



%%

matrixxxx = [averageXi; averageLambda; averagev; averageHj; averagesigma; averageLambda2; averageMu]'

std(matrixxxx)
%%

%%
FOAcorr
mean(FOAcorr)
std(FOAcorr)
%%
autocorr(78*U{1},400)

%%
plot(U{27})

%%
 y = averageHj';
x = log(averageLambda');
X = [ones(length(x),1) x];
b1 = X\y;

yCalc1 = b1*x';
yCalc2 = X*b1;
scatter(x,y)
hold on
%plot(y,yCalc1)
plot(x,yCalc2,'-')
xlabel('H_J')
ylabel('H')
title('')
grid on
%plot(FOAcorr,H,'.')
legend('Data','OLS','Location','northeast');
hold off



%%

plot(78*U{17}(78*U{17}<=1))

%%

plot(Z{1})
mean(Z{1})
%%
FOAcorr=0; H=0;
cleanU = cell(30,1);
empiricalcorr = cell(N,1);

for k= 1:30
      empiricalcorr{k} = autocorr(78*real(U{k}), NumLags = 1);
        FOAcorr(k) = empiricalcorr{k}(2);
%index(k) = FOAcorr(k)>0.45;
index(k) = empiricalcorr{k}(1) > 0.5;

if index(k) == 0
    cleanU{k}= 0;
else
    cleanU{k}= 78*T{k};
end
end 

FOAcorr=0;
empiricalcorr = cell(N,1);
H=0;
for k = 1:30
    if cleanU{k} == 0
    0
    else
        empiricalcorr{k} = autocorr(78*real(cleanU{k}), NumLags = 1);
        FOAcorr(k) = empiricalcorr{k}(2);
H(k) = gmmempericaltest{k}(4);
    end


end

H = H(H ~= 0);
FOAcorr = FOAcorr(FOAcorr ~= 0);
y = H';
x = FOAcorr';
X = [ones(length(x),1) x];
b1 = X\y;

yCalc1 = b1*x';
yCalc2 = X*b1;
scatter(x,y)
hold on
%plot(y,yCalc1)
plot(x,yCalc2,'-')
xlabel('rho_1')
ylabel('H')
title('')
grid on
%plot(FOAcorr,H,'.')
legend('Data','OLS','Location','northeast');
hold off
std(FOAcorr)

%%
for k = 1:30
empiricalcorr{k} = autocorr(real(78*U{k}), NumLags = 1);
FOAcorr(k) = empiricalcorr{k}(2);
H(k) = gmmempericaljumpY{k}(4);

end
y = H';
x = FOAcorr';
X = [ones(length(x),1) x];  
b1 = X\y;

yCalc1 = b1*x';
yCalc2 = X*b1;
scatter(x,y)
hold on
%plot(y,yCalc1)
plot(x,yCalc2,'-')
xlabel('rho_1')
ylabel('H')
title('')
grid on
%plot(FOAcorr,H,'.')
legend('Data','OLS','Location','NE');
hold off
%%
for k = 1:30
empiricalcorr{k} = autocorr(real(78*U{k}), NumLags = 1);
FOAcorr(k) = gmmempericaljumpY{k}(7);
H(k) = gmmempericaljumpY{k}(1);

end
y = H';
x = FOAcorr';
X = [ones(length(x),1) x];  
b1 = X\y;

yCalc1 = b1*x';
yCalc2 = X*b1;
scatter(x,y)
hold on
%plot(y,yCalc1)
plot(x,yCalc2,'-')
xlabel('rho_1')
ylabel('H')
title('')
grid on
%plot(FOAcorr,H,'.')
legend('Data','OLS','Location','NE');
hold off

%%

FOAcorr
%%
mean(FOAcorr)
autocorr(78*U{1},NumLags=400, NumSTD=0)

%%

plot(log(T{5}))
hold on 
plot(log(T{1}))
hold off


%%

ts1 = timeseries(T{1})
%%

ts1.Name = 'Realized variance';
ts1.TimeInfo.Units = 'days';
ts1.TimeInfo.StartDate = '01-Jan-2014';  
ts1.TimeInfo.Format = 'mmm dd, yy';       % Set format for display on x-axis.

plot(ts1)
%%
mean(FOAcorr)

%%

plot(T{5})

%%

%%

parfor rep =1:30
tic
   EstimatesJump1{rep} =  GMM_Estimator3(78*U{k}, initial(78*U{k})', [0,1,2,3,4,5,10,20,50],'N');
     dXi = EstimatesJump1{rep}(1);
    EstimatesJump{rep} = GMM_EstimatorSVJ2s(U{k},U{k},U{k},U{k},U{k},dXi,initialSVJ(T{k},U{k},0),vell,'N');
  
toc
    disp(rep);
end

%%
initialSVJ(vY_listApp{1,1}.BV,vY_listApp{1,1}.RV,vY_listApp{1,1}.RD)
%%
   reps = 1:30;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 for h = reps
    GMMHApp1(h) = EstimatesJump1{h}(1);
    GMMHApp2(h) = EstimatesJump1{h}(2);
    GMMHApp3(h) = EstimatesJump1{h}(3);
    GMMHApp4(h) = EstimatesJump1{h}(4);
    GMMHApp5(h) = EstimatesJump{h}(1);
    GMMHApp6(h) = EstimatesJump{h}(2);
    GMMHApp7(h) = EstimatesJump{h}(3);
%   ini{h} = initialSVJ(vY_listApp{1,h}.BVfive,vY_listApp{1,h}.RVfive, vY_listApp{1,h}.RD);
  %  iniplot(h) = ini{h}(1);
 %    iniplot2(h) = ini{h}(2);
  %   iniplot3(h) = ini{h}(3);
   %  iniplot4(h) = ini{h}(4);
    % iniplot5(h) = ini{h}(5);
     %iniplot6(h) = ini{h}(6);
   %  iniplot7(h) = ini{h}(7);
 end 
xi = mean(GMMHApp1(reps))
lambda = mean(GMMHApp2(reps))
v = mean(GMMHApp3(reps))
H = mean(GMMHApp4(reps))
SigmaJ = mean(GMMHApp5(reps))
LambdaJ = mean(GMMHApp6(reps))
MuJ = mean(GMMHApp7(reps))


%%
   reps = 1:250;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
 for h = reps
    GMMHApp1(h) = EstimatesRVu{h}{1,5}(1);
  

 end 
xi = mean(GMMHApp1(reps))

plot(GMMHApp1(reps))
%%
plot(GMMHApp1(reps))
hold on
plot(iniplot(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp2(reps))
hold on
plot(iniplot2(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp3(reps))
hold on
plot(iniplot3(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp4(reps))
hold on
plot(iniplot4(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp5(reps))
hold on
plot(iniplot5(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp6(reps))
hold on
plot(iniplot6(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')
%%
plot(GMMHApp7(reps))
hold on
plot(iniplot7(reps),'.-')
%plot(0.01:0.01:0.2,'*')
hold off
legend('Estimates', 'Initials')



%%
%vY_list = cell(1);
%rng(125)
for rep = 1:250   
    tic
    % Generate vY for each repetition
    vYMS = imModelSigmajump(23400, 1000, 5.225, dXi, vLambda, vV, vH, rand() * 10000000);
    
    % Save simulation in the cell array
    vY_listMS{rep} = vYMS;
        
    disp(rep);
    toc
end

%%
%rng(125);

% Create a cell array to store simulation from each repetition
%vY_listJumpX = cell(1, 1);

parfor rep = 1:5
    % Generate vY for each repetition
    %rng(100);
    tic
    for i = 5:5
    vY_listJumpsigma{rep}{i} = SimModelSigmajump(iN/10,  2500, 5.225, 0.0225, vLambda(i), vV(i), vH(i),rand() * 10000000);
    end

    % Save simulation in the cell array
    toc
    disp(rep);
end
%%
X = (vY_listJumpsigma{1,1}{1,5}.IV(3:end))-2*(vY_listJumpsigma{1,1}{1,5}.IV(2:end-1))+(vY_listJumpsigma{1,1}{1,5}.IV(1:end-2));
initialSVJS(vY_listJumpsigma{1,4}{1,5}.IV,0,X)
%%
plot(vY_listJumpsigma{1,4}{1,1}.IV)
mean(vY_listJumpsigma{1,4}{1,1}.IV)
%%
  
% for each repetition   
parfor rep = 1:5
  for i = 1:1
    tic
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    EstimatesBolkoApp{rep}{i} = GMM_EstimatorSVJS(vY_listJumpsigma{1,rep}{1,i}.IV,vY_listJumpsigma{1,rep}{1,i}.IV,vY_listJumpsigma{1,i}{1,4}.IV,vY_listJumpsigma{1,i}{1,4}.IV,initialSVJS(vY_listJumpsigma{1,rep}{1,i}.RV,vY_listJumpsigma{1,rep}{1,i}.RV,0),[0,1,2,3,4,5,  10,20,50,100],'N');
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
    end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    disp(rep);
    toc 
end

%%

plot(log(vY_listJumpsigma{1,1}{1,1}.IV))
%%
%rng(1250);

% Create a cell array to store simulation from each repetition
 

parfor rep = 401:500
    tic
    for i = 1:5
    % Generate vY for each repetition
    %rng(100);
    vY = SimModelSigmajump(23400/2, 2500, 5, 0.0225, vLambda(i), vV(i), vH(i),rand() * 10000000);
    
    % Save simulation in the cell array
    vY_listApp2{rep}{i} = vY;
    end 
    disp(rep);
   toc
end
%%
EstimatesBolkoAppIV = cell(1,1)
%%
% for each repetition   
parfor rep =1:500
    tic 
    for i = 2
    
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4);
   X = vY_listApp2{1,rep}{1,i}.IV(3:end)-2*vY_listApp2{1,rep}{1,i}.IV(2:end-1)+vY_listApp2{1,rep}{1,i}.IV(1:end-2);
    [EstimatesBolkoAppIVN{rep}{i}, m_hatjumpY{rep}{i}, mS_HACjumpY{rep}{i}, jacobianjumpY{rep}{i}] = GMM_EstimatorSVJS(vY_listApp2{1,rep}{1,i}.IV,vY_listApp2{1,rep}{1,i}.IV,0,0,initialSVJS(vY_listApp2{1,rep}{1,i}.IV,0,X')',[0,1,2,3,4,5,10,20,30,50],'N');
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
   % end    
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    end
    disp(rep);
    toc
    
end
    

%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for j = 1:5
    for h = 1:500 % Run for each rep
    GMMXi(j,h) = EstimatesBolkoAppIVN{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesBolkoAppIVN{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesBolkoAppIVN{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesBolkoAppIVN{1, h}{1, j}(4);    
    GMMSigma(j,h) = EstimatesBolkoAppIVN{1, h}{1, j}(5);
    GMMlambdaj(j,h) = EstimatesBolkoAppIVN{1, h}{1, j}(6);
    GMMmu(j,h) = EstimatesBolkoAppIVN{1, h}{1, j}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]
%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,1}.IV(3:end))-2*(vY_listApp2{1,i}{1,1}.IV(2:end-1))+(vY_listApp2{1,i}{1,1}.IV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,1}.IV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,2}.IV(3:end))-2*(vY_listApp2{1,i}{1,2}.IV(2:end-1))+(vY_listApp2{1,i}{1,2}.IV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,2}.IV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,3}.IV(3:end))-2*(vY_listApp2{1,i}{1,3}.IV(2:end-1))+(vY_listApp2{1,i}{1,3}.IV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,3}.IV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,4}.IV(3:end))-2*(vY_listApp2{1,i}{1,4}.IV(2:end-1))+(vY_listApp2{1,i}{1,4}.IV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,4}.IV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500
X = (vY_listApp2{1,i}{1,5}.IV(3:end))-2*(vY_listApp2{1,i}{1,5}.IV(2:end-1))+(vY_listApp2{1,i}{1,5}.IV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,5}.IV,0,X');
end
mean(initiallll,1)
std(initiallll,1)

%%
initiallll


%%

plot((vY_listApp2{1,1}{1,1}.IV))
mean(vY_listApp2{1,1}{1,1}.IV)
%%
exp(0.1*(exp(2*(0.5*0.2^2))-1))*exp(0.1*200*(exp(0.5*0.2^2)^2-1))
exp(0.025*(exp((0.5*0.2^2))-1))
4*0.0225*(0.1/78)*(0.2^2)
4*0.0225*78*exp((0.1*(exp(2*(0.5*0.2^2)))-1))

%%
(integraljumpsigma(0.1,0.1,0.1,2500))
%%


plot(log(vY_listApp{1,rep}{1,5}.IV))
hold on

%plot(log(vY_listJumpX2{1,1}{1,1}.IV))
hold off

%%
%%
pts = linspace(-4,4,500);
seH = 0;
for j = 1:4
for i = 1:500
  % seH =  JACOBI{1, i}{1, j}(8,4);
  seH = 0;
seH = ((jacobianjumpY{1, i}{1, j}'*(mS_HACjumpY{1, i}{1, j})*(jacobianjumpY{1, i}{1, j})));
seH2(i) = sqrt(seH(4,4));
density(i) = (EstimatesBolkoAppIVN{1, i}{1, j}(4)-vH(j))/seH2(i);

end

[z, y] =ksdensity(density,pts,'bandwidth', 1);
plot(y,z, '-', 'LineWidth',1)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[z, y] =ksdensity(x);
plot(y,z, '--b', 'LineWidth',2)
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-4 4])


%%
p_value = 0;
Sargan = cell(1);
for rep = 1:500
    for i =1:5
Sargan{1,rep}{1,i} = 2500 * m_hatjumpY{1,rep}{1,i}*(inv(mS_HACjumpY{1,rep}{1,i}))*m_hatjumpY{1,rep}{1,i}';  % Sargan–Hansen statistic
df = 11-7+1;

p_value(rep,i) = 1 - chi2cdf(Sargan{1,rep}{1,i}, df);
Sargan2(rep,i) = Sargan{1,rep}{1,i};
%[empdist{1,rep}{1,i}, empdistx{1,rep}{1,i}] =ecdf(vY_listJumpX2{1,rep}{1,i}.RVfive);


    end
end

mean(p_value,1)
%%



%%

for i = 1:5
nSimulations = length(Sargan2(:,i));  
df = 5;  


jhs_simulated = (Sargan2(:,i));


alpha_levels = linspace(0.01, 1, 10000);  
critical_values = chi2inv(1 - alpha_levels, df);  


rr = zeros(size(alpha_levels)); 

for i = 1:length(alpha_levels)
   
    rr(i) = mean(jhs_simulated > critical_values(i));
end

plot(alpha_levels, rr, '--', 'LineWidth', 1);
hold on;


end

plot([0 1], [0 1], 'r', 'LineWidth', 1.5);  % Reference line y = x


xlabel('Nominal Level (\alpha)');
ylabel('Rejection Rate');
title('Rejection Rate vs. Nominal Level');
legend('H=0.05','H=0.1','H=0.3','H=0.5','H=0.7', 'Reference Line', 'Location', 'southeast');
grid on;

hold off

%%
% for each repetition   
parfor rep =1:500
    tic 
    for i = 1   
    
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4);
   X = vY_listApp2{1,rep}{1,i}.RV(3:end)-2*vY_listApp2{1,rep}{1,i}.RV(2:end-1)+vY_listApp2{1,rep}{1,i}.RV(1:end-2);
    [EstimatesBolkoAppRVu{rep}{i}, m_hatjumpYRVu{rep}{i}, mS_HACjumpYRVu{rep}{i}, jacobianjumpYRVu{rep}{i}] = GMM_EstimatorSVJS(vY_listApp2{1,rep}{1,i}.RV,vY_listApp2{1,rep}{1,i}.RV,0,0,initialSVJS(vY_listApp2{1,rep}{1,i}.RV,0,X')',[0,1,2,3,4,5,10,20,30,50],'N');
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
   % end    
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    end
    disp(rep);
    toc
    
end
    
%%
exp(0.02*(exp(0.5*0.1^2)-1))

%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,1}.RV(3:end))-2*(vY_listApp2{1,i}{1,1}.RV(2:end-1))+(vY_listApp2{1,i}{1,1}.RV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,1}.RV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,2}.RV(3:end))-2*(vY_listApp2{1,i}{1,2}.RV(2:end-1))+(vY_listApp2{1,i}{1,2}.RV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,2}.RV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,3}.RV(3:end))-2*(vY_listApp2{1,i}{1,3}.RV(2:end-1))+(vY_listApp2{1,i}{1,3}.RV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,3}.RV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500 
X = (vY_listApp2{1,i}{1,4}.RV(3:end))-2*(vY_listApp2{1,i}{1,4}.RV(2:end-1))+(vY_listApp2{1,i}{1,4}.RV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,4}.RV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
for i = 1:500
X = (vY_listApp2{1,i}{1,5}.RV(3:end))-2*(vY_listApp2{1,i}{1,5}.RV(2:end-1))+(vY_listApp2{1,i}{1,5}.RV(1:end-2));
initiallll(i,:) = initialSVJS(vY_listApp2{1,i}{1,5}.RV,0,X');
end
mean(initiallll,1)
std(initiallll,1)
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for j = 1:5
    for h = 1:500 % Run for each rep
    GMMXi(j,h) = EstimatesBolkoAppRVu{1, h}{1, j}(1);
    GMMLambda(j,h) = EstimatesBolkoAppRVu{1, h}{1, j}(2);
    GMMv(j,h) = EstimatesBolkoAppRVu{1, h}{1, j}(3);
    GMMH(j,h) = EstimatesBolkoAppRVu{1, h}{1, j}(4);    
    GMMSigma(j,h) = EstimatesBolkoAppRVu{1, h}{1, j}(5);
    GMMlambdaj(j,h) = EstimatesBolkoAppRVu{1, h}{1, j}(6);
    GMMmu(j,h) = EstimatesBolkoAppRVu{1, h}{1, j}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]
%%
pts = linspace(-4,4,500);
seH = 0;
for j = 1:4
for i = 1:500
  % seH =  JACOBI{1, i}{1, j}(8,4);
  seH = 0;
seH = ((jacobianjumpYRVu{1, i}{1, j}'*(mS_HACjumpYRVu{1, i}{1, j})*(jacobianjumpYRVu{1, i}{1, j})));
seH2(i) = sqrt(seH(4,4));
density(i) = (EstimatesBolkoAppRVu{1, i}{1, j}(4)-vH(j))/seH2(i);

end

[z, y] =ksdensity(density,pts,'bandwidth', 1);
plot(y,z, '-', 'LineWidth',1)
hold on
end

pd = makedist('Normal','mu',0,'sigma',1);
x = random(pd,1000000,1);
[z, y] =ksdensity(x);
plot(y,z, '--b', 'LineWidth',2)
hold off
legend('0.05','0.1','0.3','0.5', 'N(0,1)')
xl = xlim;
xlim([-4 4])



%%
p_value = 0;
Sargan2 = 0;
Sargan = cell(1);
for rep = 1:500
    for i =1:5
Sargan{1,rep}{1,i} = 2500 * m_hatjumpYRVu{1,rep}{1,i}*(inv(mS_HACjumpYRVu{1,rep}{1,i}))*m_hatjumpYRVu{1,rep}{1,i}';  % Sargan–Hansen statistic
df = 11-7+1;

p_value(rep,i) = 1 - chi2cdf(Sargan{1,rep}{1,i}, df);

Sargan2(rep,i) = Sargan{1,rep}{1,i};
%[empdist{1,rep}{1,i}, empdistx{1,rep}{1,i}] =ecdf(vY_listJumpX2{1,rep}{1,i}.RVfive);


    end
end

mean(p_value,1)
%%



%%

for i = 1:5
nSimulations = length(Sargan2(:,i));  
df = 5;  


jhs_simulated = (Sargan2(:,i));


alpha_levels = linspace(0.01, 1, 10000);  
critical_values = chi2inv(1 - alpha_levels, df);  


rr = zeros(size(alpha_levels)); 

for i = 1:length(alpha_levels)
   
    rr(i) = mean(jhs_simulated > critical_values(i));
end

plot(alpha_levels, rr, '--', 'LineWidth', 1);
hold on;


end

plot([0 1], [0 1], 'r', 'LineWidth', 1.5);  % Reference line y = x


xlabel('Nominal Level (\alpha)');
ylabel('Rejection Rate');
title('Rejection Rate vs. Nominal Level');
legend('H=0.05','H=0.1','H=0.3','H=0.5','H=0.7', 'Reference Line', 'Location', 'southeast');
grid on;

hold off

%%
% for each repetition   
parfor rep = 1 :250    
    tic 
    for i = 1:5
   
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesJumpXbolko{rep}{i}, m_hatjumpxbolko{rep}{i}, mS_HACjumpxbolko{rep}{i}, jacobianjumpxbolko{rep}{i}] = GMM_EstimatorSVJ(vY_list{1,rep}.RVfive(i,:),vY_list{1,rep}.RVfive(i,:),0,0,0,initialSVJ(vY_list{1,rep}.BVfive(i,:),vY_list{1,rep}.RVfive(i,:),0),[0,1,2,3,5,10,20,50,100, 150],'R');
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
   % end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    end
    disp(rep);
    toc
    
end

%%
% for each repetition   
parfor rep = 1 :250    
    tic 
    for i = 1:5
   
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesJumpXbolko2{rep}{i}, m_hatjumpxbolko2{rep}{i}, mS_HACjumpxbolko2{rep}{i}, jacobianjumpxbolko2{rep}{i}] = GMM_Estimator3(vY_listJumpX2{1,rep}{1,i}.RVfive,initial(vY_listJumpX2{1,rep}{1,i}.RVfive)',vell,'R');
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
   % end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    end
    disp(rep);
    toc
    
end


%%
% for each repetition   
parfor rep = 1 :50    
    tic 
    for i = 1:1
   
   % vTheta_initial = reshape(vTheta_initials(rep, :), 5, 4); 
    [EstimatesJumpYbolko2{rep}{i}, m_hatjumpYbolko2{rep}{i}, mS_HACjumpYbolko2{rep}{i}, jacobianjumpYbolko2{rep}{i}] = GMM_Estimator3(vY_listApp2{1,rep}{1,i}.RV,initial(vY_listApp2{1,rep}{1,i}.RV   )',vell,'R');
    %estimatrix = NaN(5, 4);
    %for j = 1:5
   %     estimatrix(j, 1:4) = Estimates{1,j};
   % end
    %estimatesForall(rep, :) = reshape(estimatrix, 1, []);
    end
    disp(rep);
    toc
    
end

%% Initial values for RV
one = ones(1,7); two =ones(1,7); three=ones(1,7); four=ones(1,7); five =ones(1,7);
for rep = 1:250
one(rep,:) =  initialSVJ(vY_list{1,rep}.BVfive(1,:),vY_list{1,rep}.RVfive(1,:),0);
two(rep,:) =  initialSVJ(vY_list{1,rep}.BVfive(2,:),vY_list{1,rep}.RVfive(2,:),0);  
three(rep,:) =  initialSVJ(vY_list{1,rep}.BVfive(3,:),vY_list{1,rep}.RVfive(3,:),0) ; 
four(rep,:) =  initialSVJ(vY_list{1,rep}.BVfive(4,:),vY_list{1,rep}.RVfive(4,:),0);
five(rep,:) =  initialSVJ(vY_list{1,rep}.BVfive(5,:),vY_list{1,rep}.RVfive(5,:),0);  

end

%% Initial values for RV
one = zeros(1,4); two =zeros(1,4); three=zeros(1,4); four=zeros(1,4); five =zeros(1,4);
for rep = 1:250
one(rep,:) =  initial(vY_listJumpX2{1,rep}{1,1}.RVfive);
two(rep,:) =  initial(vY_listJumpX2{1,rep}{1,2}.RVfive);
three(rep,:) =  initial(vY_listJumpX2{1,rep}{1,3}.RVfive);
four(rep,:) =  initial(vY_listJumpX2{1,rep}{1,4}.RVfive);
five(rep,:) =  initial(vY_listJumpX2{1,rep}{1,5}.RVfive);

end
%% Initial values for RV
one = zeros(1,4); two =zeros(1,4); three=zeros(1,4); four=zeros(1,4); five =zeros(1,4);
for rep = 1:250
one(rep,:) =  initial(vY_listJumpsigma{1,rep}{1,1}.RV);
two(rep,:) =  initial(vY_listJumpsigma{1,rep}{1,2}.RV);
three(rep,:) =  initial(vY_listJumpsigma{1,rep}{1,3}.RV);
four(rep,:) =  initial(vY_listJumpsigma{1,rep}{1,4}.RV);
five(rep,:) =  initial(vY_listJumpsigma{1,rep}{1,5}.RV);

end
%%
mean(one,1, 'omitnan')
mean(two,1,'omitnan')
mean(three,1,'omitnan')
mean(four,1,'omitnan')
mean(five,1,'omitnan')
std(one,1,'omitnan')
std(two,1,'omitnan')
std(three,1,'omitnan')
std(four,1,'omitnan')
std(five,1,'omitnan')
%%

one(one ~= -inf)
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for j = 1:5
    for h = 1:250  % Run for each rep
    GMMXi(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(1);
    GMMLambda(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(2);
    GMMv(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(3);
    GMMH(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(4);
   % GMMSigma(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(5);
   % GMMlambdaj(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(6);
   % GMMmu(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for j = 1:1
    for h = 1:50  % Run for each rep
    GMMXi(j,h) = EstimatesJumpYbolko2{1, h}{1,j}(1);
    GMMLambda(j,h) = EstimatesJumpYbolko2{1, h}{1,j}(2);
    GMMv(j,h) = EstimatesJumpYbolko2{1, h}{1,j}(3);
    GMMH(j,h) = EstimatesJumpYbolko2{1, h}{1,j}(4);
   % GMMSigma(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(5);
   % GMMlambdaj(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(6);
   % GMMmu(j,h) = EstimatesJumpXbolko2{1, h}{1,j}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing")]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2)]
%%
initialSVJS(vY_listJumpsigma{1,1}{1,1}.IV,0,0)

%%
mean(vY_listJumpsigma{1,1}{1,1}.IV)

%%


%%
0.0225*(0.0250*(0.3000+0.5*1.2715))
%%
x = integraljumpsigma(0.025,0.3,1.2715,1000)-integraljumpsigmaext(0.025,0.3,1.2715,1000)
0.0225
0.0225*(x)
(0.0250*(0.3000+0.5*1.2715))
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for j = 5:5
    for h = 1:10    % Run for each rep
    GMMXi(h) = EstimatesBolkoAppIV{1, h}{1, j}(1);
    GMMLambda(h) = EstimatesBolkoAppIV{1, h}{1, j}(2);
    GMMv(h) = EstimatesBolkoAppIV{1, h}{1, j}(3);
    GMMH(h) = EstimatesBolkoAppIV{1, h}{1, j}(4);
    GMMSigma(h) = EstimatesBolkoAppIV{1, h}{1, j}(5);
    GMMlambdaj(h) = EstimatesBolkoAppIV{1, h}{1, j}(6);
    GMMmu(h) = EstimatesBolkoAppIV{1, h}{1, j}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]
%%
GMMXi = 0; GMMLambda = 0; GMMv = 0; GMMH = 0; GMMSigma=0; GMMlambdaj=0;  GMMmu=0;
for j = 1:1
    for h = 1:10  % Run for each rep
    GMMXi(h) = EstimatesBolkoAppIV{1, h}{1,j}(1);
    GMMLambda(h) = EstimatesBolkoAppIV{1, h}{1,j}(2);
    GMMv(h) = EstimatesBolkoAppIV{1, h}{1,j}(3);
    GMMH(h) = EstimatesBolkoAppIV{1, h}{1,j}(4);
    GMMSigma(h) = EstimatesBolkoAppIV{1, h}{1,j}(5);
    GMMlambdaj(h) = EstimatesBolkoAppIV{1, h}{1,j}(6);
    GMMmu(h) = EstimatesBolkoAppIV{1, h}{1,j}(7);
    end 
end
[mean(GMMXi,2), mean(GMMLambda,2), mean(GMMv,2), mean(GMMH,2, "omitmissing"),mean(GMMSigma,2),mean(GMMlambdaj,2),mean(GMMmu,2)]
[std(GMMXi,1,2),std(GMMLambda,1,2),std(GMMv,1,2),std(GMMH,1,2),std(GMMSigma,1,2),std(GMMlambdaj,1,2),std(GMMmu,1,2)]
a 
%%

test = integraljumpsigma(0.0279,0.3498,1.2575)

%%
test = {};
test2 = {};
for i=1:250
test{i}= initialSVJS(vY_listJumpsigma{1,i}{1,1}.RV,vY_listJumpsigma{1,i}{1,1}.RV,0);
test2{i}= initialSVJS(vY_list{1,i}.RV,vY_list{1,i}.RV(1,:),0);
Htest(i) = test{1,i}(4);
Htest2(i) = test2{1,i}(4);

end

mean(Htest)
mean(Htest2)
%% 

mean(vY_listJumpsigma{1,1}{1,4}.RV)
%%
plot(log(vY_listJumpsigma{1,1}{1,1}.IV))
%% Figure 1 panel B Bolko
plot(log(vY_listJumpsigma{1,2}{1,1}.IV))
hold on
plot(log(vY_listJumpsigma{1,2}{1,2}.IV))
plot(log(vY_listJumpsigma{1,2}{1,3}.IV))
plot(log(vY_listJumpsigma{1,2}{1,4}.IV))
plot(log(vY_listJumpsigma{1,2}{1,5}.IV))
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('log(integrated variance)')
xlabel('time')
grid("on")
hold off


%% Figure 1 panel B Bolko
plot(log(vY_listJumpX{1,2}{1,1}.SPV))
hold on
plot(log(vY_listJumpX{1,2}{1,2}.SPV))
plot(log(vY_listJumpX{1,2}{1,3}.SPV))
plot(log(vY_listJumpX{1,2}{1,4}.SPV))
plot(log(vY_listJumpX{1,2}{1,5}.SPV))
legend('H=0.05','H=0.10','H=0.30','H=0.50','H=0.70')
ylabel('log(integrated variance)')
xlabel('time')
grid("on")
hold off
%%

[testing testing2 Rn] = teststat(SRV, 0.9, 1-gmmempericaljumpX{1,1}(4), 0.95);

p_value = 1 - evcdf(Rn, 0, 1)
testing 
testing2
Rn
craticallevel = evinv(0.05,0,1)

Rn >= -log(-log(1-0.05))

%%
gmmempericaljumpX{1,1}(4)
%%
for i = 1:30
[tStatistiktest, stop2] = seqTestPos(Z{i}, 1, 1-gmmempericaljumpX{1,i}(4), 0.95);
jumpcount(i) = sum(stop2(:) >= 1)
JPemp(i) = sum(stop2(:) >= 1) /length(SRV)
end
%%

mean(JPemp)
mean(jumpcount)
%%
for i =1:30
gmmempH(i) = gmmempericaljumpX{1,i}(4)

end

%%

plot(JPemp)

%%

surfing = JPemp'*gmmempH
%%

surf(surfing)
%%
plot(stop2)
hold on 
plot(abs(normalize(SRV)))
hold off



%%
[tStatistiktest, stop2] = seqTestPos(Z{1}, 1, 1-gmmemperical{1,1}(4), 0.95);


%%
figure;
y = abs(normalize((Z{1})));
x = 1:length(Z{1});
yMark=NaN(1,length(y));
for i=1:1:length(Z{1})
   
if stop2(i) >= 1
 yMark(i)=y(i);
end
end
plot(time,yMark,'->')

hold on
plot(time,repelem((-log(-log(0.9))),length(time)))
plot(time,y, '.')

hold off
%xlabel(time,'Date')
ylabel('Value')

%%
rep((-log(-log(0.95))))
%%



p_valuegum = 1- evcdf(tStatistiktest)
%%
hold on
plot(SRV)
hold off
%% 
plot(U{1})

%%
for rep = 1:250
    for i = 1:5
SOIS{rep}{i} =  vY_listJumpsigma{1,rep}{1,i}.RV(3:end)-2*vY_listJumpsigma{1,rep}{1,i}.RV(2:end-1)+vY_listJumpsigma{1,rep}{1,i}.RV(1:end-2);
    end
end

%%
PJ = 0;
parfor j= 1:5
for i = 1:250
    i
[tStatistiktest, stop2] = seqTestPos(SOIS{1,i}{1,j}, 0.9, 1-vH(j), 0.95);
PJ(i,j) = sum(stop2 >= 1) /length(SOIS{1,i}{1,j});
end
end
%%

plot(PJ)
hold on
%%

Zhat1 = conv2(PJ,K,'same');

%%
[d1, d2] = meshgrid(time);
timeplot = d1-d2;
%%

timecorrect = time()
%%
figure; h=surf(PJ,'FaceLighting','phong');
 view([0,90]); colormap(hot); cb=colorbar; 
 h.EdgeColor='none'; shading('interp'); 
 xlabel('Parameter Set'), ylabel('Replication'), zlabel('Jump proportion')
 xticks(0:1:5)
 %%
hold on
ZMax=max(-log(-log(0.95)));






