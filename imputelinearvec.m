function [newvec] = imputelinearvec(oldvec,k)
%imputelinearvec: create a new vector with k linearly interpolated values 
%   between each element of an initial vector (with equal spacing).
%
%   [newvec] = imputelinearvec(oldvec,k) creates a matrix newvec of 
%   dimension (p+(p-1)k)xn based on the pxn matrix oldvec, by adding
%   k linearly interpolated elements between each element of oldvec.
%
%   The interpolation is performed through columns.
%
%   Example: 
%   n = 1;
%   p = 10;
%   k = 3;
%   oldvec = randn(p,n);
%   x = linspace(0,1,p)';
%   [newvec] = imputelinearvec(oldvec,k);
%   [newx] = imputelinearvec(x,k);
%   plot(newx,newvec,'xr') 
%   hold on
%   plot(x,oldvec,'xb')
%   legend('New points', 'Initial points')
%   hold off
%
%   Author: David Benatia
%   Affiliation: University of Montreal
% Last Update: February 20, 2017

if size(oldvec,1)<2
    display('This is not a column vector')
end    

p = size(oldvec,1);
n = size(oldvec,2);
stepsize = (oldvec(2:end,:)-oldvec(1:end-1,:))/(k+1);

kspace = linspace(1,k,k)';

for i=1:size(oldvec,1)-1
 newvec = [newvec;oldvec(i,:);repmat(oldvec(i,:),k,1)+repmat(kspace,1,n).*repmat(stepsize(i,:),k,1)];
end

end
