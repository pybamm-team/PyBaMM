function Y = cod_qmult (Q, X, method)
%COD_QMULT computes Q'*X, Q*X, X*Q', or X*Q with Q from COD_SPARSE.
% Q is a matrix or a struct representing the SPQR Householder form.  An
% additional column permutation matrix Q.Pc may be present in the Q struct.
%
% Usage: Y = cod_qmult (Q,X,method)
%
%   method = 0: Y = Q'*X    default if 3rd input argument is not present.
%   method = 1: Y = Q*X 
%   method = 2: Y = X*Q'
%   method = 3: Y = X*Q
%
% Example:
%
%   [U, R, V, r] = cod_sparse (A) ;
%   Y = cod_qmult (U, X, 0) ;                   % Y = U' * X
%   U = cod_qmult (U, speye (size (U.H,1)), 1)  % convert U to matrix form
%
% See also cod_sparse, spqr, spqr_qmult

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 3)
    method = 0 ;
end

% multiply Q and X
if (isstruct (Q))
    if (~isfield (Q, 'Pc'))
        Y = spqr_qmult (Q, X, method) ;
    else
        switch method
            case 0, Y = Q.Pc' * spqr_qmult (Q, X, method) ;
            case 1, Y = spqr_qmult (Q, Q.Pc * X, method) ;
            case 2, Y = spqr_qmult (Q, X * Q.Pc', method) ;
            case 3, Y = spqr_qmult (Q, X, method) * Q.Pc ;
        end
    end
else
    switch method
        case 0, Y = Q'*X ;
        case 1, Y = Q*X ;
        case 2, Y = X*Q' ;
        case 3, Y = X*Q ;
    end
end
