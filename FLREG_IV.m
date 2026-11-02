function [PIhat_IV,Yhat,R2_IV,Zhat,bhat,R2_FS] = FLREG_IV(alpha,W,Z,Y,Sw,Sz,Sy)
% FLREG_IV: estimate the operator PI using the functional linear regression
%     method for endogenous predictor function developed in Benatia, 
%     Carrasco and Florens (2017). This program is for L^2 functions.
%     
%     [PI_hat,Yhat,R2hat,R2tilde,R2tilde_local] = FLREG(alpha,Z,Y,Sz,Sy,nk)
%     performs a functional linear regression of the functions Y (Syxn) on
%     the functions Z ((Szxnk)xn) using the Tikhonov regularization parameter
%     alpha. Sz is the domain of integration of Z, Sy is that of Y, n is
%     the sample size and nk is the number of functional predictor.
%
%     Yhat is the predicted functional response using the estimated
%     operator PI_hat. R2hat, R2tilde and R2tilde_local are goodness-of-fit
%     criteria proposed in Yao, Muller and Wang (2005).
%
%     Author: David Benatia
%     Affiliation: University of Montreal
%     Last Update: February 20, 2017

%Initialize
    n=size(Z,2); %number of observations
    nk=1; %still not coded for multiple endogenous regressors - to be done.
    MZ=zeros(nk*n,n); %matrix to be inverted
    MW=zeros(nk*n,n); %matrix to be inverted
    Zi=Z; %initial Z
    Wi=Z; %initial W
    Yi=Y; %initial Y

%Store the mean of Z and normalize
    Zmean=zeros(length(Sz),nk);
    for kk=1:nk
        Zmean(:,kk)=mean(Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:),2);
        Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:)=Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:)-repmat(Zmean(:,kk),1,n);
    end

%Store the mean of W and normalize
    Wmean=zeros(length(Sw),nk);
    for kk=1:nk
        Wmean(:,kk)=mean(Z((kk-1)*length(Sw)+1:(kk)*(length(Sw)),:),2);
        Z((kk-1)*length(Sw)+1:(kk)*(length(Sw)),:)=Z((kk-1)*length(Sw)+1:(kk)*(length(Sw)),:)-repmat(Wmean(:,kk),1,n);
    end
    
%Store the mean of Y and normalize
    Ymean=mean(Y(1:length(Sy),:),2);
    Y=(Y-repmat(Ymean,1,n));

for x=1:nk
        MZ((x-1)*n+1:x*n,:) = innerproduct_trapz(Sz,Z((x-1)*length(Sz)+1:x*length(Sz),:),Z((x-1)*length(Sz)+1:x*length(Sz),:));
        MW((x-1)*n+1:x*n,:) = innerproduct_trapz(Sw,W((x-1)*length(Sw)+1:x*length(Sw),:),W((x-1)*length(Sw)+1:x*length(Sw),:));
end

%FLREG IV
A=alpha*n^2.*eye(n,n)+MZ*MW;
PIhat_IV=Y*(Z*MW/A)'; %estimated kernel

% Compute Yhat
    Yhat=zeros(length(Sy),n);
    for kk=1:nk
        Yhat = Yhat + innerproduct_trapz(Sz,PIhat_IV(:,1+length(Sz)*(kk-1):length(Sz)*kk)',Zi(1+length(Sz)*(kk-1):length(Sz)*kk,:)-repmat(Zmean(:,kk),1,n));
    end
    Yhat = repmat(Ymean(:,1),1,n)+Yhat;

%Compute R2 criteria
    R2_IV{1}=trapz(Sy,mean((Yhat-repmat(mean(Yhat,2),1,n)).^2,2))/trapz(Sy,mean((Yi-repmat(mean(Yi,2),1,n)).^2,2));
    R2_IV{2}=trapz(Sy,(mean((Yhat-repmat(mean(Yhat,2),1,n)).^2,2)./mean((Yi-repmat(mean(Yi,2),1,n)).^2,2)));
    R2_IV{3}=mean((Yhat-repmat(mean(Yhat,2),1,n)).^2,2)./mean((Yi-repmat(mean(Yi,2),1,n)).^2,2);
    
%First-stage
A=alpha*n.*eye(n,n)+MW;
bhat=Z*(W/A)'; %estimated kernel

% Compute Yhat
    Zhat=zeros(length(Sz),n);
    for kk=1:nk
        Zhat = Zhat + innerproduct_trapz(Sw,bhat(:,1+length(Sw)*(kk-1):length(Sw)*kk)',Wi(1+length(Sw)*(kk-1):length(Sw)*kk,:)-repmat(Wmean(:,kk),1,n));
    end
    Zhat = repmat(Zmean(:,1),1,n)+Zhat;

%Compute R2 criteria
    R2_FS{1}=trapz(Sz,mean((Zhat-repmat(mean(Zhat,2),1,n)).^2,2))/trapz(Sz,mean((Zi-repmat(mean(Zi,2),1,n)).^2,2));
    R2_FS{2}=trapz(Sz,(mean((Zhat-repmat(mean(Zhat,2),1,n)).^2,2)./mean((Zi-repmat(mean(Zi,2),1,n)).^2,2)));
    R2_FS{3}=mean((Zhat-repmat(mean(Zhat,2),1,n)).^2,2)./mean((Zi-repmat(mean(Zi,2),1,n)).^2,2);

    
end