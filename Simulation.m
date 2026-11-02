%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functional Linear Regression with Functional Response
% Benatia, Carrasco and Florens (2017)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Last Update: February 20, 2017
% Author: David Benatia
% Affiliation: Université de Montréal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes:
%   - The script presents a simple version of the program used to perform
%   the simulation detailed in the paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0-Set the working directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
dir = fileparts(mfilename('fullpath'));
cd(dir)
tic

ENDOG = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1-Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% General
t0=100; %number of observations in [0,1] - used to discretize the continuous functions
t1=1000; %number of steps in the pseudocontinuous grid for the 'actual' trajectories
alpha=0.01; %arbitrary alpha: used if REGULARIZATION==0
n=100; %number of prediction functions (cross-sectional unit)
sigma=1; %std associated with the Wiener process
theta=1; %mean-reversion speed of the O-U process

T0=(0:1:(t0-1))'/(t0-1); %interval [0,1] with t0 equally-spaced steps
T1=(0:1:(t1-1))'/(t1-1); %interval [0,1] with t1 equally-spaced steps
incr0=1/(t0-1); %increment given t0 steps
incr1=1/(t1-1); %increment given t1 steps
PI=(1-abs(repmat(T1,1,length(T1))-repmat(T1',length(T1),1)).^2); %true kernel

if ENDOG==1
    a_z=1;
    b_z = 1;
    c_z=sqrt(1+(1-b_z)*5.7201);
    sigma_e=1;
end

% Simulations
ITER = 100; %number of iterations

for p=1:ITER
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 2-Prediction functions zi(t) for i in {1,...,n} and t in [0,1]
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    G1=randn(1,n); %Vector of standard-normally distributed random numbers
    A=2 + (5-2).*rand(1,n);
    B=2 + (5-2).*rand(1,n);
    parfor i=1:n
        if binornd(1,0.5)==1
            R1(1,i)=1;
        else
            R1(1,i)=-1;
        end
    end
    Z1=zeros(t1,n); %initiate TxN matrix (pseudo-continuous functions)
    parfor i=1:n
        Z1(:,i)= betapdf(T1,A(i),B(i));
    end
    parfor t=1:length(T1)
        Z1(t,:)=Z1(t,:)+G1;
    end
    
    if ENDOG==1
        G1=randn(1,n); %Vector of standard-normally distributed random numbers
        A=2 + (5-2).*rand(1,n);
        B=2 + (5-2).*rand(1,n);
        W1=zeros(t1,n);
        parfor i=1:n
            W1(:,i)= betapdf(T1,A(i),B(i));
        end
        parfor i=1:length(T1)
            W1(i,:)=W1(i,:)+G1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 3-Error functions ui(s) for i in {1,...,n} and s in [0,1]
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %O-U PROCESS (Euler-Maruyama method for approximating the numerical solution of a stochastic differential equation)
    d=sigma*randn(t1-1,n); %generate t1 (standard) normally distributed draws - they are the infinitesimal displacement of the process
    U1=zeros(t1,n); %cumulative sum of the displacement produces each brownian motion
    D=cumsum(d,1);
    Wiener=zeros(t1,n);
    Wiener(2:t1,:)=D.*sqrt(T1(t1))/sqrt(t1);
    for t=2:t1
        U1(t,:)=U1(t-1,:)+theta.*(zeros(1,n)-U1(t-1,:)).*incr1+(Wiener(t,:)-Wiener(t-1,:));
    end
    
    % %Wiener PROCESS
    % d=sigma*randn(t1-1,n); %generate t1 (standard) normally distributed draws - they are the infinitesimal displacement of the process
    % U1_0=zeros(t1,n); %cumulative sum of the displacement produces each brownian motion
    % D=cumsum(d,1);
    % U1_0(2:t1,:)=D.*sqrt(T1(t1))/sqrt(t1);
    
    if ENDOG==1;
        d=sigma_e*randn(t1-1,n); %generate t1 (standard) normally distributed draws - they are the infinitesimal displacement of the process
        E1_0=zeros(t1,n); %cumulative sum of the displacement produces each brownian motion
        D=cumsum(d,1);
        Wiener=zeros(t1,n);
        Wiener(2:t1,:)=D.*sqrt(T1(t1))/sqrt(t1);
        for t=2:t1
            E1_0(t,:)=E1_0(t-1,:)+theta.*(zeros(1,n)-E1_0(t-1,:)).*incr1+(Wiener(t,:)-Wiener(t-1,:));
        end
        Z1=a_z*U1+b_z*W1+c_z*E1_0;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 4-Response functions
    % Model: Y = PI * Z + U
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    PIZ1 = innerproduct_trapz(T1,PI,Z1);
    Y1= PIZ1 + U1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 5-Discretization: Select t0 discretized points (whether they're equally spaced or sparse
    %depends on the construction of T0)
    %Note: We do not observe continuous trajectories. We must hence infer the
    %entire trajectories using non-parametric smoothing techniques or kernel
    %functions, or other quadrature rules... (here we use trapezoidal
    %integration, although it is not the fastest method)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [k,d] = dsearchn(T1,T0); %k is the vector of indices for the corresponding values of T0 in T1, d is the distance between the two values
    U0=U1(k,:);
    Z0=Z1(k,:);
    meanZ0=repmat(mean(Z0,2),1,n);
    %Z0=Z0-repmat(mean(Z0,2),1,n);
    if ENDOG==1
        W0=W1(k,:);
        meanW0=repmat(mean(W0,2),1,n);
        %W0=W0-meanW0;
        W=W0;
    end
    
    Y0=Y1(k,:);
    
    meanY0=repmat(mean(Y0,2),1,n);
    %Y0=Y0-repmat(mean(Y0,2),1,n);
    PI0=PI(k,k);
    PIZ0=PIZ1(k,:);
    
    Y=Y0;
    Z=Z0;
    T=T0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 6-Solving for an arbitrary alpha (regularization parameter)
    %Note: We take an arbitrary alpha and solve for the kernel (inverse pb)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [PIhat,Yhat,R2]=FLREG(alpha,Z,Y,T,T);
    SE=trapz(T,trapz(T,((PIhat-PI0).^2)')); %Hilbert-Schmidt Norm
    PIhat_all(:,:,p) = PIhat;
    SE_all(:,:,p) = SE;
    
    if ENDOG==1
        [PIhat_IV,Yhat_IV,R2_IV,Zhat,bhat,R2_FS] = FLREG_IV(alpha,W,Z,Y,T,T,T);
        SE_IV=trapz(T,trapz(T,((PIhat_IV-PI0).^2)'));
        PIhat_IV_all(:,:,p) = PIhat_IV;
        SE_IV_all(:,:,p) = SE_IV;
    end
    
end

PIhat = mean(PIhat_all,3);
SE = mean(SE_all,3);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 8-Simulate PI0_alpha (biased kernel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    n=100;
    iter=100;
    VZ_=zeros(t0,t0,iter);
    PI0_alpha_=zeros(t0,t0,iter);
    for p=1:iter
    %Generate prediction functions
    G1=randn(1,n); %Vector of standard-normally distributed random numbers 
    A=2 + (5-2).*rand(1,n);
    B=2 + (5-2).*rand(1,n);
    Z1=zeros(t1,n);
    parfor i=1:n
        Z1(:,i)= betapdf(T1,A(i),B(i));
    end
    parfor i=1:length(T1)
        
        Z1(i,:)=Z1(i,:)+G1;
        
    end
    [k,d] = dsearchn(T1,T0);
    Z0=Z1(k,:);
    VZ1=(1/n)*(Z1*Z1');
    VZ_(:,:,p)=VZ1(k,k);
    PIZ1=zeros(t1,n);
    parfor s=1:t1
        PIZ1(s,:)=trapz(T1,repmat(PI(s,:)',1,n).*Z1);
    end
    ZZ=zeros(n,n);
    parfor i=1:n
        ZZ(i,:) = trapz(T1,repmat(Z1(:,i),1,n).*Z1);
    end
    A=alpha*n.*eye(n,n)+ZZ;
    PI1_alpha=PIZ1*(Z1/A)';  
    PI0_alpha_(:,:,p)=PI1_alpha(k,k,:);
    end
    PI0_alpha=mean(PI0_alpha_,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 9-Final plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
coord1=159;
coord2=14;
fontsize=15;
    
figure(1)
plot(T1,Z1(:,1:10))
%axis([0 2*pi min(min(Z1))-1 max(max(Z1))+1])
title('Zi (continuous)','FontSize',15)
set(gca,'FontSize',15)
figure(2)
plot(T1,U1(:,1:10))
%axis([0 2*pi min(min(U1))-1 max(max(U1))+1])
title('Ui (continuous)','FontSize',15)
set(gca,'FontSize',15)
figure(3)
plot(T1,Y1(:,1:10))
%axis([0 2*pi min(min(Y1))-1 max(max(Y1))+1])
title('Zi (continuous)','FontSize',15)
set(gca,'FontSize',15)
figure(4)
plot(T0,Z0(:,1:10)+meanZ0(:,1:10))
%axis([0 2*pi min(min(Z1))-1 max(max(Z1))+1])
title('z_i (discrete)','FontSize',15)
set(gca,'FontSize',15)
figure(5)
plot(T0,U0(:,1:10))
%axis([0 2*pi min(min(U1))-1 max(max(U1))+1])
title('u_i (discrete)','FontSize',15)
set(gca,'FontSize',15)
figure(6)
plot(T0,Y0(:,1:10)+meanY0(:,1:10))
%axis([0 2*pi min(min(Y1))-1 max(max(Y1))+1])
title('y_i (discrete)','FontSize',15)
set(gca,'FontSize',15)

%True kernel
figure(11)
surf(T,T,PI0);
ylabel('[0,1] (zi(t))')
xlabel('[0,1] (yi(s))')
view(coord1 ,coord2)
title('True','FontSize',fontsize)
%shading interp
colormap winter %gray for black and white
%freezeColors
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -0.3 1.3])
shading flat

%Biased kernel
figure(12)
surf(T,T,PI0_alpha);
colormap gray
%freezeColors
ylabel('[0,1] (zi(t))')
xlabel('[0,1] (yi(s))')
view(coord1 ,coord2)
title('Regularized','FontSize',fontsize)
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -0.3 1.3])

%Biased vs. True
figure(14)
surf(T,T,PI0)
colormap winter
shading flat
%freezeColors
hold on

surf(T,T,PI0_alpha)
colormap gray

hold off
ylabel('[0,1] (zi(t))')
xlabel('[0,1] (yi(s))')
view(coord1 ,coord2)
title('True vs. regularized','FontSize',fontsize)
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -0.3 1.3])

%Estimated kernel
figure(15)
surf(T,T,PIhat);
colormap summer
%freezeColors
ylabel('[0,1] (zi(t))')
xlabel('[0,1] (yi(s))')
view(coord1 ,coord2)
title('Mean non-IV estimate','FontSize',fontsize)
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -6 6])
shading flat

%Estimated against true
figure(16)
surf(T,T,PIhat);
colormap default
%freezeColors
hold on
surf(T,T,PI0);
colormap winter
%freezeColors
hold off
shading flat
ylabel('[0,1] (zi(t))')
xlabel('[0,1] (yi(s))')
view(coord1 ,coord2)
title('True vs. mean IV estimate','FontSize',fontsize)
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -0.3 1.3])

%Estimated against biased
figure(17)
surf(T,T,PIhat);
colormap default
shading flat
%freezeColors
hold on
surf(T,T,PI0_alpha);
colormap gray
%freezeColors
hold off
ylabel('[0,1] (zi(t))')
xlabel('[0,1] (yi(s))')
view(coord1 ,coord2)
title('Regularized vs. mean estimate','FontSize',fontsize)
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -0.3 1.3])

%Errors relative to true
figure(18)
surf(T,T,PI0-PIhat);
colormap default
%freezeColors
ylabel('T = [0,1] (interval for Zi(t))')
xlabel('S = [0,1] (interval for Yi(s))')
view(coord1 ,coord2)
title('Errors relative to true','FontSize',fontsize)
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -0.7 0.7])
shading flat
hold off

%Bias
figure(18)
surf(T,T,PI0-PI0_alpha);
colormap default
%freezeColors
ylabel('T = [0,1] (interval for Zi(t))')
xlabel('S = [0,1] (interval for Yi(s))')
view(coord1 ,coord2)
title('Bias','FontSize',fontsize)
set(gca,'FontSize',fontsize)
axis([0 1 0 1 -0.7 0.7])
shading flat
hold off

