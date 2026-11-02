function [PI_hat,Yhat,R2] = FLREG(alpha,Z,Y,Sz,Sy,nk)
% FLREG: estimate the operator PI using the functional linear regression
%     method developed in Benatia, Carrasco and Florens (2017), in the
%     case of L^2 functions.
%
%     [PI_hat,Yhat,R2hat,R2tilde,R2tilde_local] = FLREG(alpha,Z,Y,Sz,Sy,nk)
%     performs a functional linear regression of the functions Y (Syxn) on
%     the functions Z ((Szxnk)xn) using the Tikhonov regularization parameter
%     alpha. Sz is the domain of integration of Z, Sy is that of Y, n is
%     the sample size and nk is the number of functional predictor.
%
%     Yhat is the predicted functional response using the estimated
%     operator PI_hat. R2 includes 3 alternative goodness-of-fit
%     criteria proposed by Yao, Muller and Wang (2005).
%
%     Author: David Benatia
%     Affiliation: University of Montreal
% Last Update: February 20, 2017

%Input arguments
if(nargin > 6 || nargin < 5)
    disp('Error: Wrong number of input arguments');
    return;
end
if(nargin == 5)
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

%Output arguments
if(nargout > 3 || nargout < 1)
    disp('Error: Wrong number of output arguments');
    return;
end

if(nargout >= 1)
    %Initialize
    n=size(Z,2); %number of observations
    M=zeros(nk*n,n); %matrix to be inverted
    Zi=Z; %initial Z
    Yi=Y; %initial Y
    
    %Store the mean of Z and normalize
    Zmean=zeros(length(Sz),nk);
    for kk=1:nk
        Zmean(:,kk)=mean(Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:),2);
        Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:)=Z((kk-1)*length(Sz)+1:(kk)*(length(Sz)),:)-repmat(Zmean(:,kk),1,n);
    end
    
    Z=Z/std(Zi(:));
    
    %Store the mean of Y and normalize
    Ymean=mean(Y(1:length(Sy),:),2);
    Y=(Y-repmat(Ymean,1,n))/std(Yi(:));
    
    %Construct matrix M
    for x=1:nk
        M((x-1)*n+1:x*n,:) = innerproduct_trapz(Sz,Z((x-1)*length(Sz)+1:x*length(Sz),:),Z((x-1)*length(Sz)+1:x*length(Sz),:));
    end
    
    %Required for multiple functional predictors
    EYEZ=zeros(nk*length(Sz),n);
    EYEM=zeros(nk*n,nk*n);
    
    for x=1:nk
        EYEZ((x-1)*length(Sz)+1:x*length(Sz),(x-1)*n+1:x*n)=Z((x-1)*length(Sz)+1:x*length(Sz),:);
        EYEM((x-1)*n+1:x*n,(x-1)*n+1:x*n)=M((x-1)*n+1:x*n,:);
    end
    
    A=alpha*n.*eye(nk*n,nk*n)+repmat(M,1,nk); %Matrix to invert
    K=(n*alpha)\(EYEZ-repmat(Z,1,nk)*(A\EYEM));
    PI_hat=K*repmat(Y',nk,1);
    
    %Kernel estimate
    PI_hat=(PI_hat')*std(Yi(:))/std(Zi(:));
end

if(nargout >= 2)
    % Compute Yhat
    Yhat=zeros(length(Sy),n);
    for kk=1:nk
        Yhat = Yhat + innerproduct_trapz(Sz,PI_hat(:,1+length(Sz)*(kk-1):length(Sz)*kk)',Zi(1+length(Sz)*(kk-1):length(Sz)*kk,:)-repmat(Zmean(:,kk),1,n));
    end
    Yhat = repmat(Ymean(:,1),1,n)+Yhat;
end

if(nargout == 3)
    % Compute R2 criteria
    R2{1}=trapz(Sy,mean((Yhat-repmat(mean(Yhat,2),1,n)).^2,2))/trapz(Sy,mean((Yi-repmat(mean(Yi,2),1,n)).^2,2));
    R2{2}=trapz(Sy,(mean((Yhat-repmat(mean(Yhat,2),1,n)).^2,2)./mean((Yi-repmat(mean(Yi,2),1,n)).^2,2)));
    R2{3}=mean((Yhat-repmat(mean(Yhat,2),1,n)).^2,2)./mean((Yi-repmat(mean(Yi,2),1,n)).^2,2);
end

end