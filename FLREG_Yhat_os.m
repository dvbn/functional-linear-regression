function [Yhat] = FLREG_Yhat_os(PIhat,Z,Y,Z_os,Sz,Sy,nk)
% FLREG_Yhat_os: out-of-sample prediction using the kernel estimate PIhat
%   
%     PIhat must be estimated using (Z,Y) with function FLREG.m, Z_os is
%     used to predict Y_os based on PIhat. See help FLREG for more
%     information. If nk is not specified, nk is set to 1.
%
%     Author: David Benatia
%     Affiliation: University of Montreal
%     Last Update: February 20, 2017

%Input arguments
if(nargin > 7 || nargin < 6)
    disp('Error: Wrong number of input arguments');
    return;
end
if(nargin == 6)
    nk = 1;
    if size(Z,1)~=length(Sz)
        disp('Error: Z must be ((nk*Sz)*n)');
        return;
    end
else
    if size(Z,1)~=nk*length(Sz)
        disp('Error: Z must be ((nk*Sz)*n)');
        return;
    end
end

%sample size
n=size(Z_os,2); 

%Z mean 
Zmean=zeros(length(Sz),nk);
for kk=1:nk
    Zmean(:,kk)=mean(Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:),2);
end

%Y mean
Ymean=mean(Y(1:length(Sy),:),2);

%Yhat
Yhat=zeros(length(Sy),n);
for kk=1:nk
    Yhat = Yhat + innerproduct_trapz(Sz,PIhat(:,1+length(Sz)*(kk-1):length(Sz)*kk)',Z_os(1+length(Sz)*(kk-1):length(Sz)*kk,:)-repmat(Zmean(:,kk),1,n));
end
Yhat = repmat(Ymean(:,1),1,n)+Yhat;

end