function [out1, out2, out3, out4, out5] = umfpack (in1, in2, in3, in4, in5) %#ok
%UMFPACK2 computes x=A\b, x=A/b, or lu (A) for a sparse matrix A
% It is also a built-in function in MATLAB, used in x=A\b.
%
% Example:
%
% UMFPACK:                            |  MATLAB approximate equivalent:
% ---------------------------------------------------------------------
% x = umfpack (A, '\', b) ;           |  x = A \ b
%                                     |
% x = umfpack (b, '/', A) ;           |  x = b / A
%                                     |
% [L,U,P,Q] = umfpack (A) ;           |  [m,n] = size (A) ;
%                                     |  I = speye (n) ;
%                                     |  Q = I (:, colamd (A)) ;
%                                     |  [L,U,P] = lu (A*Q) ;
%                                     |
% [L,U,P,Q,R] = umfpack (A) ;         |  [m,n] = size (A) ;
%                                     |  I = speye (n) ;
%                                     |  Q = I (:, colamd (A)) ;
%                                     |  r = full (sum (abs (A), 2)) ;
%                                     |  r (find (r == 0)) = 1 ;
%                                     |  R = spdiags (r, 0, m, m) ;
%                                     |  [L,U,P] = lu ((R\A)*Q) ;
%                                     |
% [P,Q,F,C] = umfpack (A, 'symbolic') |  [m,n] = size (A) ;
%                                     |  I = speye (n) ;
%                                     |  Q = I (:, colamd (A)) ;
%                                     |  [count,h,parent,post] = ...
%                                     |  symbfact (A*Q, 'col') ;
%
% A must be sparse.  It can be complex, singular, and/or rectangular.  A must be
% square for '/' or '\'.  b must be a full real or complex vector.  For
% [L,U,P,Q,R] = umfpack (A), the factorization is L*U = P*(R\A)*Q.  If A has a
% mostly symmetric nonzero pattern, then replace "colamd" with "amd" in the
% MATLAB-equivalent column in the table above.  Type umfpack_details for more
% information.
%
% An optional final input argument provides control parameters:
%
%   opts.prl        >= 0, default 1 (errors only)
%   opts.strategy   'auto', 'unsymmetric', 'symmetric', default auto
%   opts.ordering   'amd'       AMD for A+A', COLAMD for A'A
%                   'default'   use CHOLMOD (AMD then METIS; take best fount)
%                   'metis'     use METIS
%                   'none'      no fill-reducing ordering
%                   'given'     use Qinit (this is default if Qinit present)
%                   'best'      try AMD/COLAMD, METIS, and NESDIS; take best
%   opts.tol        default 0.1
%   opts.symtol     default 0.001
%   opts.scale      row scaling: 'none', 'sum', 'max'
%   opts.irstep     max # of steps of iterative refinement, default 2
%   opts.singletons 'enable','disable' default 'enable'
%
% An optional final output argument provides informational output, in a
% struct whos contents are mostly self-explanatory.
%
% See also: lu_normest, colamd, amd, umfpack.
% To use UMFPACK for an arbitrary b, see umfpack_solve.

% Copyright 1995-2009 by Timothy A. Davis.

help umfpack 
error ('umfpack mexFunction not found') ;

