function [CV,dist,Yhat] = FLREG_CV(alpha,Z,Y,Sz,Sy,nk)
% FLREG_CV: compute the leave-one-out cross-validation criterion using
% sample (Z,Y)
%   
%     PIhat(-i) is estimated using (Z(-i),Y(-i)) with function FLREG.m, Z is
%     used to predict Y based on PIhat(-i) with function FLREG_Yhat_os.m.
%     See help FLREG for more information. The procedure is described
%     in Benatia, Carrasco and Florens (2017).
%     If nk is not specified, nk is set to 1.
%     If nargout==1, CV is reported
%     If nargout==2, CV,dist is reported
%     If nargout==3, CV,dist,Yhat is reported
%
%     Author: David Benatia
%     Affiliation: University of Montreal
%     Last Update: February 20, 2017

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

if(nargout > 3 || nargout < 1)
    disp('Error: Wrong number of output arguments');
    return;
end

n=size(Z,2); %number of observations
Zi=Z; %initial Z
Yi=Y; %initial Y
Yhat = zeros(size(Y));
dist=zeros(n,1);

parfor i=1:n; 
    Ytemp=Yi;
    Ytemp(:,i)=[];
    Ztemp=Zi;
    Ztemp(:,i)=[];
    [PIhattemp] = FLREG(alpha,Ztemp,Ytemp,Sz,Sy,nk);
    [Yhattemp] = FLREG_Yhat_os(PIhattemp,Ztemp,Ytemp,Zi,Sz,Sy,nk);
    dist(i,1)=trapz(Sy,(Yi(:,i)-Yhattemp(:,i)).^2);  
    Yhat(:,i) = Yhattemp(:,i);
end

CV =mean(dist);

end
