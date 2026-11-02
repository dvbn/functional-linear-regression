%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Application script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functional Linear Regression with Functional Response
% Benatia, Carrasco and Florens (2017)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Last Update: February 20, 2017
% Author: David Benatia
% Affiliation: Université de Montréal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes:
%   - The script performs the following steps:
%       0) Load DATA.mat
%       1) Linear projection of functionals onto seasonal dummies
%       2) Data-driven selection of the smoothing parameter
%       3) Kernel estimates
%       4) Figures
%
%   - DATA.mat contains:
%       * DATA_Y (25x368x35): Functional response is DATA_Y(:,:,1),  the
%       remaining arrays contain the seasonal dummies, the last array
%       contains a variable for year to separate the samples.
%       * DATA_Z (73x368x35): Functional predictor is DATA_Z(:,:,1),  the
%       remaining arrays contain the seasonal dummies, the last array
%       contains a variable for year to separate the samples.
%       * TIME_Y (25x368): Time variable for functional data
%       * DEMAND (41616x1): Raw electricity demand (aggregate Ontario)
%       * TEMP  (41616x1): Raw temperature (aggregate Ontario)
%       * TIME  (41616x1): Time variable for raw data
%       * period (1x5 struct): decompose time into seasons to draw figure 2
%       * SEASON  (41616x1): decompose time into seasons to draw figure 1
%
%   - Required functions to run the script:
%       * FLREG.m: estimate the kernel
%       * FLREG_Yhat_os.m: out-of-sample prediction using a kernel estimate
%       * FLREG_MSPE.m: choose alpha that min MSPE for out-of
%       sample prediction
%       * FLREG_CV.m: choose alpha that min leave-one-out CV
%       criterion
%       * FLREG_KCV.m: choose alpha that min k-fold CV
%       criterion where k groups are defined by the user
%       * imputelinearvec.m: interpolate elements linearly (optional)
%       * innerproduct_trapz.m: construct the nxn M matrix of L2 inner
%       products using the trapezoidal rule of numerical integration
%       * rotateXLabels.m: Mathworks function to rotate x-axis labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0-Set the working directory and load the data set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
dir = fileparts(mfilename('fullpath'));
cd(dir)

load('DATA.mat')

% Separate out the samples

%Estimation Sample: 2010-2013
[~,b_s] =  find(DATA_Y(2,:,35)~=2014);

%Prediction Sample: 2014
[~,b_os] =  find(DATA_Y(2,:,35)==2014);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1-Deseasonalization of the functional data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We partial out the dummies through linear projection of Y and Z
% This approach is motivated by the Frisch-Waugh-Lowell theorem

% Deseasonalize Y

%Demand curves
Y = DATA_Y(:,b_s,1);
YL = Y(:);
TIME_s = TIME_Y(:,b_s);

Y_os = DATA_Y(:,b_os,1);
YL_os = Y_os(:);
TIME_os = TIME_Y(:,b_os);

%Dummies [1,12:15,70:72,75:97,117:119]
X0 = ones(size(DATA_Y(:,b_s,1))); %Constant
X1 = reshape(repmat(DATA_Y(23,b_s,2:5),25,1,1),[size(DATA_Y(:,b_s,1),1)*size(DATA_Y(:,b_s,1),2),size(DATA_Y(:,b_s,2:5),3)]); %Weekdays
X2 = reshape(repmat(DATA_Y(23,b_s,6:8),25,1,1),[size(DATA_Y(:,b_s,1),1)*size(DATA_Y(:,b_s,1),2),size(DATA_Y(:,b_s,6:8),3)]); %Years
X3 = reshape(DATA_Y(:,b_s,9:31),[size(DATA_Y(:,b_s,1),1)*size(DATA_Y(:,b_s,1),2),size(DATA_Y(:,b_s,9:31),3)]); %d_HOUR
X4 = reshape(repmat(DATA_Y(23,b_s,32:34),25,1,1),[size(DATA_Y(:,b_s,1),1)*size(DATA_Y(:,b_s,1),2),size(DATA_Y(:,b_s,32:34),3)]); %Months

X0_os = ones(size(DATA_Y(:,b_os,1)));
X1_os = reshape(repmat(DATA_Y(23,b_os,2:5),25,1,1),[size(DATA_Y(:,b_os,1),1)*size(DATA_Y(:,b_os,1),2),size(DATA_Y(:,b_os,2:5),3)]); %Weekdays (11-17)
X2_os = reshape(repmat(DATA_Y(23,b_os,6:8),25,1,1),[size(DATA_Y(:,b_os,1),1)*size(DATA_Y(:,b_os,1),2),size(DATA_Y(:,b_os,6:8),3)]); %Years (70-74)
X3_os = reshape(DATA_Y(:,b_os,9:31),[size(DATA_Y(:,b_os,1),1)*size(DATA_Y(:,b_os,1),2),size(DATA_Y(:,b_os,9:31),3)]); %d_HOUR (75-98)
X4_os = reshape(repmat(DATA_Y(23,b_os,32:34),25,1,1),[size(DATA_Y(:,b_os,1),1)*size(DATA_Y(:,b_os,1),2),size(DATA_Y(:,b_os,32:34),3)]); %Months (112-123)

%Linear Projection
Y_REGS = [X0(:),X1,X2,X3,X4];
Y_REGS_os = [X0_os(:),X1_os,X2_os,X3_os,X4_os];
[Y_BETAS,~,R,~,~] = regress(YL,Y_REGS);
Y = reshape(R,size(Y));
Y_os = reshape(YL_os - Y_REGS_os*Y_BETAS,size(Y_os)); % Residuals to be used in the functional regression!

% Deseasonalize Z

%Temperature curves
Z = DATA_Z(:,b_s,1);
ZL = Z(:);

Z_os = DATA_Z(:,b_os,1);
ZL_os = Z_os(:);

%Dummies
X0 = ones(size(DATA_Z(:,b_s,1)));
X1 = reshape(repmat(DATA_Z(30,b_s,2:5),73,1,1),[size(DATA_Z(:,b_s,1),1)*size(DATA_Z(:,b_s,1),2),size(DATA_Z(:,b_s,2:5),3)]); %Weekdays (11-17)
X2 = reshape(repmat(DATA_Z(30,b_s,6:8),73,1,1),[size(DATA_Z(:,b_s,1),1)*size(DATA_Z(:,b_s,1),2),size(DATA_Z(:,b_s,6:8),3)]); %Years (70-74)
X3 = reshape(DATA_Z(:,b_s,9:31),[size(DATA_Z(:,b_s,1),1)*size(DATA_Z(:,b_s,1),2),size(DATA_Z(:,b_s,9:31),3)]); %d_HOUR (75-98)
X4 = reshape(repmat(DATA_Z(30,b_s,32:34),73,1,1),[size(DATA_Z(:,b_s,1),1)*size(DATA_Z(:,b_s,1),2),size(DATA_Z(:,b_s,32:34),3)]); %Months (112-123)

X0_os = ones(size(DATA_Z(:,b_os,1)));
X1_os = reshape(repmat(DATA_Z(30,b_os,2:5),73,1,1),[size(DATA_Z(:,b_os,1),1)*size(DATA_Z(:,b_os,1),2),size(DATA_Z(:,b_os,2:5),3)]); %Weekdays (11-17)
X2_os = reshape(repmat(DATA_Z(30,b_os,6:8),73,1,1),[size(DATA_Z(:,b_os,1),1)*size(DATA_Z(:,b_os,1),2),size(DATA_Z(:,b_os,6:8),3)]); %Years (70-74)
X3_os = reshape(DATA_Z(:,b_os,9:31),[size(DATA_Z(:,b_os,1),1)*size(DATA_Z(:,b_os,1),2),size(DATA_Z(:,b_os,9:31),3)]); %d_HOUR (75-98)
X4_os = reshape(repmat(DATA_Z(30,b_os,32:34),73,1,1),[size(DATA_Z(:,b_os,1),1)*size(DATA_Z(:,b_os,1),2),size(DATA_Z(:,b_os,32:34),3)]); %Months (112-123)


%Linear Projection
Z_REGS = [X0(:),X1,X2,X3,X4];
Z_REGS_os = [X0_os(:),X1_os,X2_os,X3_os,X4_os];
[Z_BETAS,~,R,~,~] = regress(ZL,Z_REGS);
Z = reshape(R,size(Z));
Z_os = reshape(ZL_os - Z_REGS_os*Z_BETAS,size(Z_os));

% Define the additional parameters
S=(0:1:24)'; %domain of Y
T=(0:1:72)'; %domain of Z
n=length(Z); %sample size
nk=1; %number of functional regressors

% Seasonal component
Y_SEAS = Y_REGS*Y_BETAS;
Y_SEAS_os = Y_REGS_os*Y_BETAS;
Y_SEAS = reshape(Y_SEAS,[25,length(Y_SEAS)/25]);
Y_SEAS_os = reshape(Y_SEAS_os,[25,length(Y_SEAS_os)/25]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2-DATA-DRIVEN SELECTION OF ALPHA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Interpolate linearly between discrete observations
% kk = 10; %number of extra points between discrete observations
% [Y] = imputelinearvec(Y,kk); %generate new vectors
% [Z] = imputelinearvec(Z,kk);
% [Y_os] = imputelinearvec(Y_os,kk);
% [Z_os] = imputelinearvec(Z_os,kk);
% [Sy] = imputelinearvec(Sy,kk);
% [Sz] = imputelinearvec(Sz,kk);

% Optimization options
options = optimoptions('fmincon');
options = optimoptions(options, 'TolFun', 1e-6, 'TolX', 1e-6);
options.Display = 'iter';

% LEAVE ONE OUT CV
%Alpha selection using leave-one-out CV
[ALPHA_NFOLD_FLREG,FVAL_NFOLD_FLREG,~,~]=fmincon(@(x)FLREG_CV(x,Z,Y,T,S,nk),1,[],[],[],[],0,20,[],options);

% K-FOLD CV
%Define groups over which cross-validation is performed
[~,GROUPS(1).IDX]=find(DATA_Y(2,:,35)==2010);
[~,GROUPS(2).IDX]=find(DATA_Y(2,:,35)==2011);
[~,GROUPS(3).IDX]=find(DATA_Y(2,:,35)==2012);
[~,GROUPS(4).IDX]=find(DATA_Y(2,:,35)==2013);

%Alpha selection using k-fold CV
[ALPHA_KFOLD_FLREG,FVAL_KFOLD_FLREG,~,~]=fmincon(@(x)FLREG_KCV(x,Z,Y,T,S,nk,GROUPS),1,[],[],[],[],0,20,[],options);

% BEST PREDICTION (min MSPE)
%Alpha selection using argmin MPSE
[ALPHA_BPRED_FLREG,FVAL_BPRED_FLREG,~,~]=fmincon(@(x)FLREG_MSPE(x,Z,Y,T,S,nk,Z_os,Y_os),1,[],[],[],[],0,20,[],options);

% Store results
ALPHA_FLREG = [ALPHA_NFOLD_FLREG,ALPHA_BPRED_FLREG];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3-Kernel estimates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% OLS with hourly predictors
Z_OLS = Z - repmat(mean(Z,2),1,size(Z,2));
Y_OLS = Y - repmat(mean(Y,2),1,size(Y,2));
PI_hat_OLS = ((Z_OLS*Z_OLS')\(Z_OLS*Y_OLS'))';
Yhat_OLS_os = repmat(mean(Y,2),1,size(Y_os,2)) + PI_hat_OLS*(Z_os - repmat(mean(Z,2),1,size(Z_os,2)));

% OLS with 3 hourly predictors
Z_OLS2 = Z((1:3:73),:) - repmat(mean(Z((1:3:73),:),2),1,size(Z((1:3:73),:),2));
PI_hat_OLS2 = ((Z_OLS2*Z_OLS2')\(Z_OLS2*Y_OLS'))';
Yhat_OLS2_os = repmat(mean(Y,2),1,size(Y_os,2)) + PI_hat_OLS2*(Z_os((1:3:73),:) - repmat(mean(Z((1:3:73),:),2),1,size(Z_os((1:3:73),:),2)));

% FLREG with alpha_CV
[PI_hat1] = FLREG(ALPHA_FLREG(1),Z,Y,T,S);
[Yhat_os1] = FLREG_Yhat_os(PI_hat1,Z,Y,Z_os,T,S,nk);

% FLREG with alpha_BP
[PI_hat2] = FLREG(ALPHA_FLREG(2),Z,Y,T,S);
[Yhat_os2] = FLREG_Yhat_os(PI_hat2,Z,Y,Z_os,T,S,nk);

%% %%


Zmean=zeros(length(T),1);
    for kk=1:1
        Zmean(:,kk)=mean(Z((kk-1)*length(T)+1:(kk)*(length(T)),:),2);
        Zi((kk-1)*length(T)+1:(kk)*(length(T)),:)=Z((kk-1)*length(T)+1:(kk)*(length(T)),:)-repmat(Zmean(:,kk),1,n);
    end
   
    
    %Store the mean of Y and normalize
    Ymean=mean(Y(1:length(S),:),2);
    Yi=(Y-repmat(Ymean,1,n));
VZ = 1/n * Zi*Zi';
[V,D] = eig(VZ);
% VZ*V = V*D
D = diag(D);
[a,b] = sort(D,'descend');
V = V(:,b);
M = innerproduct_trapz(T,Zi,V(:,1:5));
 %surf(1/n*Yi*M*(repmat(D(1:5).^(-1),1,length(T)).*V(:,1:5)'))
%surf(VZ*V-V*diag(D))

figure(1)
subplot(1,2,1)
colormap hot(8)
 surf(1/n*Yi*M*(repmat(D(1:5).^(-1),1,length(T)).*V(:,1:5)'))
subplot(1,2,2)
surf(PI_hat2)
colormap hot(8)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4-TABLE RESULTS MSPE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Smoothing?
% [Y_SEAS_os] = imputelinearvec(Y_SEAS_os,kk);
% [D_C_os] = imputelinearvec(D_C_os,kk);

display('FLREG CV Leave-One-Out')
ALPHA_FLREG(1)
mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_os1).^2))
display('FLREG BP')
ALPHA_FLREG(2)
mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_os2).^2))
display('OLS hourly')
mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_OLS_os).^2))
display('OLS 3 hourly')
mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_OLS2_os).^2))

TABLE = ['& $\alpha_{CV}$ & $\alpha_{BP}$  & hourly & 3 hourly  \\ \hline $\alpha$ &', ...
    num2str(ALPHA_FLREG(1),'%2.2f'),' & ',num2str(ALPHA_FLREG(2),'%2.2f'),'  &   &     \\  MSPE & ', ...
    num2str(mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_os1).^2)),'%2.2f'),' & ', ...
    num2str(mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_os2).^2)),'%2.2f'),' & ', ...
    num2str(mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_OLS_os).^2)),'%2.2f'),' & ', ...
    num2str(mean(trapz(S,(DATA_Y(:,b_os,1)-Y_SEAS_os-Yhat_OLS2_os).^2)),'%2.2f'),'\\ ', ...
    'cond(ZZ/n) & & &', num2str(cond(Z_OLS*Z_OLS'/n),'%1.2e\n') ,' & ', num2str(cond(Z_OLS2*Z_OLS2'/n),'%1.2e\n') ,'\\ \hline']
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5-FIGURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% FIGURE 1 & FIGURE 2 %%%%%%%%%%%%%%%

FigHandle = figure(1)
set(FigHandle, 'Position', [100 100 1153 436]);
subplot(1,2,1)
for i=1:5
    plot(TIME(period(i).winter),DEMAND(period(i).winter),'-b')
    hold on
    plot(TIME(period(i).spring),DEMAND(period(i).spring),'-k')
    plot(TIME(period(i).summer),DEMAND(period(i).summer),'-r')
    plot(TIME(period(i).autumn),DEMAND(period(i).autumn),'-k')
end
xlim([datetime(TIME(1)),datetime(TIME(end))])
hold off
set(gca,'FontSize',13)
xlabel('Time')
ylabel('Electricity consumption (GWh)')

subplot(1,2,2)
for i=1:5
    plot(TIME(period(i).winter),TEMP(period(i).winter),'-b')
    hold on
    plot(TIME(period(i).spring),TEMP(period(i).spring),'-k')
    plot(TIME(period(i).summer),TEMP(period(i).summer),'-r')
    plot(TIME(period(i).autumn),TEMP(period(i).autumn),'-k')
end
hold off
set(gca,'FontSize',13)
xlabel('Time')
ylabel('Temperature (C)')
xlim([datetime(TIME(1)) datetime(TIME(end))])

FigHandle = figure(2)
set(FigHandle, 'Position', [100 100 1153 436]);
subplot(1,3,1)
plot(TEMP(SEASON==0),DEMAND(SEASON==0),'xb')
set(gca,'FontSize',13)
xlim([min(TEMP(SEASON==0)) max(TEMP(SEASON==0))])
ylim([10 27])
ylabel('Electricity consumption (GWh)')
xlabel('Temperature (C)')
title('Nov1-Apr30 (Winter)','Fontsize',13)
subplot(1,3,2)
plot(TEMP(SEASON==1 | SEASON==3),DEMAND(SEASON==1 | SEASON==3),'xk')
set(gca,'FontSize',13)
xlim([min(TEMP(SEASON==1)) max(TEMP(SEASON==1 | SEASON==3))])
ylim([10 27])
ylabel('Electricity consumption (GWh)')
xlabel('Temperature (C)')
title('May1-May31 & Sep16-Oct31','Fontsize',13)
subplot(1,3,3)
plot(TEMP(SEASON==2),DEMAND(SEASON==2),'xr')
set(gca,'FontSize',13)
xlim([min(TEMP(SEASON==2)) max(TEMP(SEASON==2))])
ylim([10 27])
ylabel('Electricity consumption (GWh)')
xlabel('Temperature (C)')
title('Jun1-Sept15 (Summer)','Fontsize',13)

%%%%%%%%%%%%%%% FIGURE 3 & FIGURE 4 %%%%%%%%%%%%%%%

FigHandle = figure(3)
set(FigHandle, 'Position', [100 100 1153 436]);
subplot(1,2,1)
plot((0:1:24),DATA_Y(:,b_s,1))
set(gca,'XLim',[0 24])
set(gca,'XTick',[0:3:24])
labels= {'12AM';'03AM';'06AM';'09AM';'12PM';'03PM';'06PM';'09PM';'12AM'};
set( gca, 'XTickLabel', labels )
rotateXLabels(gca(), 45 )
set(gca,'FontSize',13)
ylabel('Electricity Consumption (GW)')
xlabel('Time')
hold on
plot((0:1:24),mean(DATA_Y(:,b_s,1),2),'-r','LineWidth',5)
ylim([10 26])
subplot(1,2,2)
plot((0:1:72)-24,DATA_Z(:,b_s,1))
hold on
plot((0:1:72)-24,mean(DATA_Z(:,b_s,1),2),'-r','LineWidth',5)
plot([24 24], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot([0 0], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
set(gca,'XLim',[-24 48])
set(gca,'XTick',[-24:3:48])
empty=['    '];
labels= {'12AM';repmat(empty,1,1);'06AM';repmat(empty,1,1);'12PM';repmat(empty,1,1);'06PM';repmat(empty,1,1)};
set( gca, 'XTickLabel', labels )
rotateXLabels(gca(), 45 )
ylabel('Cooling Degrees (C)')
xlabel('Time')
plot((-24:1:48),mean(DATA_Z(:,b_s,1),2),'-r','LineWidth',5)
hold off
space=repmat([' '],1,20);
set(gca,'FontSize',13)
title([space,'Day-1',space,'Day',space,'Day+1',space],'Fontsize',13)
ylim([0 26])

FigHandle = figure(4)
set(FigHandle, 'Position', [100 100 1153 436]);
subplot(1,2,1)
plot((0:1:24),DATA_Y(:,b_os,1))
set(gca,'XLim',[0 24])
set(gca,'XTick',[0:3:24])
labels= {'12AM';'03AM';'06AM';'09AM';'12PM';'03PM';'06PM';'09PM';'12AM'};
set( gca, 'XTickLabel', labels )
rotateXLabels(gca(), 45 )
set(gca,'FontSize',13)
ylabel('Electricity Consumption (GW)')
xlabel('Time')
hold on
plot((0:1:24),mean(DATA_Y(:,b_os,1),2),'-r','LineWidth',5)
ylim([10 26])
subplot(1,2,2)
plot((0:1:72)-24,DATA_Z(:,b_os,1))
hold on
plot((0:1:72)-24,mean(DATA_Z(:,b_os,1),2),'-r','LineWidth',5)
plot([24 24], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot([0 0], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
set(gca,'XLim',[-24 48])
set(gca,'XTick',[-24:3:48])
empty=['    '];
labels= {'12AM';repmat(empty,1,1);'06AM';repmat(empty,1,1);'12PM';repmat(empty,1,1);'06PM';repmat(empty,1,1)};
set( gca, 'XTickLabel', labels )
rotateXLabels(gca(), 45 )
ylabel('Cooling Degrees (C)')
xlabel('Time')
plot((-24:1:48),mean(DATA_Z(:,b_os,1),2),'-r','LineWidth',5)
hold off
space=repmat([' '],1,20);
set(gca,'FontSize',13)
title([space,'Day-1',space,'Day',space,'Day+1',space],'Fontsize',13)
ylim([0 26])

%%%%%%%%%%%%%%% FIGURE 5 %%%%%%%%%%%%%%%

ALPHA = linspace(0,5,200);
CV = zeros(size(ALPHA));
BP = zeros(size(ALPHA));
for i=1:length(ALPHA)
    CV(i) = FLREG_CV(ALPHA(i),Z,Y,T,S,nk);
    BP(i) = FLREG_MSPE(ALPHA(i),Z,Y,T,S,nk,Z_os,Y_os);
end

figure(5)
[AX1,H1,H2] = plotyy(ALPHA,CV,ALPHA,BP,'plot')
set(gca,'fontsize',13)
set(AX1(1),'FontSize',13);
set(AX1(2),'FontSize',13);
axes(AX1(1));
ylabel('Cross-Validation Criterion');
xlabel('$\alpha$','interpreter','latex')
set(H2,'linestyle','-','color','red')
axes(AX1(2));
set(AX1,{'ycolor'},{'k';'k'})
ylabel('MSPE');
axes(AX1(1));
H = legend('$CV(\alpha)$', '$MSPE(\alpha)$')
set(H,'interpreter','latex')

%%%%%%%%%%%%%%% FIGURE 6 & FIGURE 7 %%%%%%%%%%%%%%%

FigHandle = figure(6)
set(FigHandle, 'Position', [100 100 1153 436]);
subplot_tight(1,2,1,.11)
[C, H] = contourf((1:3:73)'-24,S,PI_hat_OLS2)
colorbar, axis square
colormap hot(15)
hold on
plot([24 24], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot([0 0], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot(0:24,0:24,'.-k','LineWidth',2) %diag
grid on
caxis([min(min(PI_hat_OLS2)), max(max(PI_hat_OLS2))])
set(gca,'XLim',[-23 49])
set(gca,'XTick',[-23:3:49])
empty=['    '];
labels= {'12AM';repmat(empty,1,1);'06AM';repmat(empty,1,1);'12PM';repmat(empty,1,1);'06PM';repmat(empty,1,1)};
set( gca, 'XTickLabel', labels )
rotateXLabels(gca(), 45 )
set(gca,'YLim',[0 24])
set(gca,'YTick',[0:3:24])
set(gca,'YTickLabel',['12AM';'03AM';'06AM';'09AM';'12PM';'03PM';'06PM';'09PM';'12AM']);
hold off
space=repmat([' '],1,20);
title([space,'Day-1',space,'Day',space,'Day+1',space])
xlabel('Time (Temperature patterns)')
ylabel('Time (Electricity usage patterns)')
set(gca,'fontsize',13)
subplot_tight(1,2,2,.11)
[C, H] = contourf(T-24,S,PI_hat1)
colorbar, axis square
colormap hot(15)
hold on
plot([24 24], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot([0 0], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot(0:24,0:24,'.-k','LineWidth',2) %diag
grid on
caxis([min(min(PI_hat1)), max(max(PI_hat1))])
set(gca,'XLim',[-24 48])
set(gca,'XTick',[-24:3:48])
empty=['    '];
labels= {'12AM';repmat(empty,1,1);'06AM';repmat(empty,1,1);'12PM';repmat(empty,1,1);'06PM';repmat(empty,1,1)};
set( gca, 'XTickLabel', labels )
rotateXLabels(gca(), 45 )
set(gca,'YLim',[0 24])
set(gca,'YTick',[0:3:24])
set(gca,'YTickLabel',['12AM';'03AM';'06AM';'09AM';'12PM';'03PM';'06PM';'09PM';'12AM']);
hold off
space=repmat([' '],1,20);
title([space,'Day-1',space,'Day',space,'Day+1',space])
xlabel('Time (Temperature patterns)')
ylabel('Time (Electricity usage patterns)')
set(gca,'fontsize',13)

FigHandle = figure(7)
set(FigHandle, 'Position', [100 100 1153 436]);
[C, H] = contourf(T-24,S,PI_hat2)
colorbar, axis equal
colormap hot(15)
hold on
plot([24 24], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot([0 0], ylim,'LineStyle','--','color','k','LineWidth',2); %vert
plot(0:24,0:24,'.-k','LineWidth',2) %diag
grid on
caxis([min(min(PI_hat2)), max(max(PI_hat2))])
set(gca,'XLim',[-24 48])
set(gca,'XTick',[-24:3:48])
empty=['    '];
labels= {'12AM';repmat(empty,1,1);'06AM';repmat(empty,1,1);'12PM';repmat(empty,1,1);'06PM';repmat(empty,1,1)};
set( gca, 'XTickLabel', labels )
rotateXLabels(gca(), 45 )
set(gca,'YLim',[0 24])
set(gca,'YTick',[0:3:24])
set(gca,'YTickLabel',['12AM';'03AM';'06AM';'09AM';'12PM';'03PM';'06PM';'09PM';'12AM']);
hold off
space=repmat([' '],1,61);
title([space,'Day-1',space,'Day',space,'Day+1',space])
xlabel('Time (Temperature patterns)')
ylabel('Time (Electricity usage patterns)')
set(gca,'fontsize',13)

%%%%%%%%%%%%%%% FIGURE 8 %%%%%%%%%%%%%%%%

Y_LS_os = Y_os(1:25,:);
Y_LS_os = Y_LS_os(:);
Y_SEAS = Y_REGS*Y_BETAS;
Y_SEAS_os = Y_REGS_os*Y_BETAS;
Y_SEAS = reshape(Y_SEAS,[25,length(Y_SEAS)/25]);
Y_SEAS_os = reshape(Y_SEAS_os,[25,length(Y_SEAS_os)/25]);

% Absolute errors
yyy= (abs(Yhat_os2(2:end) - Y_os(2:end))./(Y_os(2:end)+Y_SEAS_os(2:end)));

FigHandle = figure(8)
subplot(1,2,1)
TIME_id = TIME_os(1:end-1,:);
tick_indx = round(linspace(1,length(TIME_id(:)),10));
tick_label = datestr(TIME_id(tick_indx), 'dd-mmm-yyyy');
plot(yyy)
hold on
smoothy = smooth(yyy,0.2,'loess'); %smooth!
plot(smoothy,'-r','LineWidth',2)
plot([1 length(Yhat_os2(2:end))],[mean(yyy), mean(yyy)],'-k','LineWidth',3)
set(gca,'XTick',tick_indx)
set(gca,'XTickLabel',tick_label)
rotateXLabels(gca(), 45 )
set(gca, 'fontsize', 13)
ylabel('$|\hat{u}_i(s)|/Y_i(s)$','interpreter','latex')
xlim([tick_indx(1) length(Yhat_os2(2:end))])
H = legend('Actual','Smoothed','Mean','location','best')
set(H,'interpreter','latex')

Y_SEAS = Y_SEAS(1:end,:);
Y_SEAS_os = Y_SEAS_os(1:end,:);
Yhat = reshape(Yhat_os2(1:end,:),size(Y_LS_os));

TIME_id = TIME_os(1:end,:);
TIME_id = TIME_id(:);

clear check_i;

%Find consecutive same dates... and then drop! (25 vectors to 24
%vectors)
for i=1:length(TIME_id)-1
    if isempty(find(TIME_id(i) == TIME_id(i+1)))==1
    else
        check_i(i) =i;
    end
end
check_i(check_i==0) = [];

[a1] = find(TIME_id == '16-Jun-2014 00:00:00');
[a2] = find(TIME_id == '21-Jun-2014 00:00:00');

set(FigHandle, 'Position', [100 100 1153 436]);
subplot(1,2,2)
plot(TIME_id(a1:a2),Y_SEAS_os(a1:a2)'+Y_LS_os(a1:a2),'DatetimeTickFormat','dd-MMMM-yyyy','LineWidth',2)
Yhat(check_i(check_i>= a1 & check_i<=a2)+1) = [];
TIME_id(check_i(check_i>= a1 & check_i<=a2)+1) = [];
Y_SEAS_os(check_i(check_i>= a1 & check_i<=a2)+1) = [];
[a1] = find(TIME_id == '16-Jun-2014 00:00:00');
[a2] = find(TIME_id == '21-Jun-2014 00:00:00');
hold on
plot(TIME_id(a1:a2),Y_SEAS_os(a1:a2)'+Yhat(a1:a2),'-r','DatetimeTickFormat','dd-MMMM-yyyy','LineWidth',1)
set(gca,'FontSize',13)
ylabel('Electricity Consumption (GW)')
set(gca,'XLim',[datetime(TIME_id(a1)) datetime(TIME_id(a2))])
set(gca,'XTick',[datetime(TIME_id(a1)):1:datetime(TIME_id(a2))])
labels= {'16-Jun-2014';'17-Jun-2014';'18-Jun-2014';'19-Jun-2014';'20-Jun-2014';'21-Jun-2014'};
set( gca, 'XTickLabel', labels)
rotateXLabels(gca(), 45 )
H = legend('Actual ($Y$)','Prediction ($\hat{Y}$)','Location','best')
set(H,'interpreter','latex')
ylim([11 21])