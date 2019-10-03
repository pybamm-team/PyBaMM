function r = rank_est (R, m, n, tol)
%RANK_EST computes a cheap estimate of the rank of a triangular matrix
d = abs (get_diag (R)) ;
if (nargin < 4 || tol < 0)
    tol = 20 * (m+n) * eps (max (d)) ;
end
r = sum (d > tol) ;

