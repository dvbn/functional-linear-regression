function [MSPE] = FLREG_MSPE(alpha,Z,Y,Sz,Sy,nk,Z_os,Y_os)
% FLREG_MSPE: compute Mean Square Prediction Error on Y_os using
%     predictor Z_os and the estimated operator obtained with sample (Z,Y)
%   
%     PIhat is estimated using (Z,Y) with function FLREG.m, Z_os is
%     used to predict Y_os based on PIhat with function FLREG_Yhat_os.m.
%     See help FLREG for more information. 
%     If nk is not specified, nk is set to 1.
%     
%     Author: David Benatia
%     Affiliation: University of Montreal
%     Last Update: February 20, 2017

%Input arguments
if(nargin > 8 || nargin < 7)
    disp('Error: Wrong number of input arguments');
    return;
end
if(nargin == 7)
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

%Estimate PI from (Z,Y)
[PIhat] = FLREG(alpha,Z,Y,Sz,Sy,nk);

%Predict Y_os from (Z_os,PIhat)
[Yhat_os] = FLREG_Yhat_os(PIhat,Z,Y,Z_os,Sz,Sy,nk);

%Compute MSPE
MSPE = mean(trapz(Sy,(Y_os - Yhat_os).^2));

end


