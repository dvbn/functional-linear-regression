function [CV,dist,Yhat] = FLREG_KCV(alpha,Z,Y,Sz,Sy,nk,GROUPS)
% FLREG_KCV: compute the k-fold cross-validation criterion using
% user-defined K GROUPS
%   
%     PIhat(-i) is estimated using (Z(-i),Y(-i)) with function FLREG.m, Z is
%     used to predict Y based on PIhat(-i) with function FLREG_Yhat_os.m.
%     See help FLREG for more information. The leave-one-out procedure is 
%     described in Benatia, Carrasco and Florens (2017). This procedure
%     leaves one group of observations out at a time.
%
%     GROUPS is a struct where GROUPS(k).IDX is a vector of indices for 
%     group k that allows to match the observation in the sample (Z,Y), for
%     all K groups.
%     
%     If nk is not specified, nk is set to 1.
%     If nargout==1, CV is reported
%     If nargout==2, CV,dist is reported
%     If nargout==3, CV,dist,Yhat is reported
%
%     Example: for n=100         
%     [~,GROUPS(1).IDX]=(1:25);
%     [~,GROUPS(2).IDX]=(26:50);
%     [~,GROUPS(3).IDX]=(51:75);
%     [~,GROUPS(4).IDX]=(76:100);
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

if(nargout > 3 || nargout < 1)
    disp('Error: Wrong number of output arguments');
    return;
end

k = size(GROUPS,2);
Zi=Z; %initial Z
Yi=Y; %initial Y
Yhat = zeros(size(Y));
dist=zeros(k,1);
for i=1:k; 
    Ytemp=Yi;
    Ytemp(:,GROUPS(i).IDX)=[];
    Ztemp=Zi;
    Ztemp(:,GROUPS(i).IDX)=[];
    [PIhattemp] = FLREG(alpha,Ztemp,Ytemp,Sz,Sy,nk);
    [Yhattemp] = FLREG_Yhat_os(PIhattemp,Ztemp,Ytemp,Zi,Sz,Sy,nk);
    dist(i,1)=mean(trapz(Sy,(Yi(:,GROUPS(i).IDX)-Yhattemp(:,GROUPS(i).IDX)).^2));  
    Yhat(:,GROUPS(i).IDX) = Yhattemp(:,GROUPS(i).IDX);
end
CV =mean(dist);
end
