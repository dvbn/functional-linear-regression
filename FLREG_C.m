function [PIhat_roll] = FLREG_C(alpha,Z,Y,Sz,Sy,rhoL,rhoF)
% FLREG_C: estimate the operator PI using the functional linear regression
%     method with constraints on the kernel's domain. 
%     This program is for L^2 functions.
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

nk=1;
n=size(Z,2);

as = max(min(Sy),min(Sz)+rhoL);
bs = min(max(Sy),max(Sz)-rhoF);
am = find(min((Sy-as).^2)==(Sy-as).^2);
bm = find(min((Sy-bs).^2)==(Sy-bs).^2);

Sy_ = Sy(am:bm);

a = zeros(size(Sy_));
bL = zeros(size(Sy_));
bU = zeros(size(Sy_));

for i=1:length(Sy_)
a(i) = min(find(min((Sy_(i)-Sz).^2) == (Sy_(i)-Sz).^2));
bL(i) = min(find(min(((Sy_(i)-rhoL)-Sz).^2)==((Sy_(i)-rhoL)-Sz).^2));
bU(i) = max(find(min(((Sy_(i)+rhoF)-Sz).^2)==((Sy_(i)+rhoF)-Sz).^2));
end

bef=bL-a; %number of obs before s
aft=bU-a; %number of obs after s

%Normalization
Zi=Z; %initial Z
Yi=Y; %initial Y
Zmean=zeros(length(Sz),nk);
for kk=1:nk
    Zmean(:,kk)=mean(Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:),2);
    Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:)=Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:)-repmat(Zmean(:,kk),1,n);
end
Z=Z/std(Zi(:));
Ymean=mean(Y(1:length(Sy),:),2);
Y=(Y-repmat(Ymean,1,n))/std(Yi(:));

%Rolling Z matrix
parfor s=1:length(Sy_)
    ZZZ(s).Z=zeros(size(Z));
    ZZZ(s).Z(bL(s):bU(s),:)=Z(bL(s):bU(s),:);
end

parfor s=1:length(Sy_)
    ZZ(s).M=zeros(n,n);
end

parfor s=1:length(Sy_)
    for i=1:n
        ZZ(s).M = innerproduct_trapz(Sz,ZZZ(s).Z,ZZZ(s).Z);
    end
    A(s).A=alpha*n.*eye(n,n)+ZZ(s).M;
end

PIhat_roll = zeros(length(Sy),length(Sz));

a = am-1;
parfor s=1:length(Sy_)
    PIhat_roll(s+a,:)=Y(s,:)*(ZZZ(s).Z/A(s).A)'; %estimated kernel
end

PIhat_roll=(PIhat_roll).*std(Yi(:))/std(Zi(:));
end