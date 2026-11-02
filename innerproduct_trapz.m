function [M] = innerproduct_trapz(T,X,Z)
%innerproduct_trapz: compute the nxk M matrix of L2 inner products <x_i,z_j>
%   using the trapezoidal rule.
%
%   [M] = innerproduct_trapz(T,X,Z) computes the nxk matrix M with (i,j)
%   element <x_i,z_j>, where x_i and z_j are T dimensional vectors.
%
%   X is a Txn matrix, Z is a Txk matrix, where T is the vector over which
%   the trapezoidal integration is performed.
%
%   This function is essentially used to calculate the estimator for 
%   functional regression developed in Benatia, Carrasco and Florens
%   (2017).
%
%   Example: 
%   n = 100;
%   k = 200;
%   T = sort(randn(100,1));
%   X = randn(length(T),n);
%   Z = randn(length(T),k);
%   [M] = innerproduct_trapz(T,X,Z);
%   surf(M)
%
%   Author: David Benatia
%   Affiliation: University of Montreal
%   Last Update: February 20, 2017

k = size(Z,2);
weights = (T(2:end)-T(1:end-1));
weights = repmat(([0;weights]+[weights;weights(end)])/2,1,k);
M = X'*(Z.*weights);

end

