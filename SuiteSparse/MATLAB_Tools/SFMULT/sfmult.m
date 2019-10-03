function y = sfmult (A,x, at,ac, xt,xc, yt,yc)
% y = sfmult (A,x, at,ac, xt,xc, yt,yc) where A is sparse and x is full
% y = sfmult (x,A, at,ac, xt,xc, yt,yc) where A is sparse and x is full
%
% Computes y = A*x, x*A, or other variants.
%
% at and ac control how the sparse matrix A is accessed:
%
%   y=A*x           at = 0, ac = 0
%   y=A.'*x         at = 1, ac = 0
%   y=conj(A)*x     at = 0, ac = 1
%   y=A'*x          at = 1, ac = 1
%
% xt and xc modify x in the same way.
% yt and yc modify the result y.  Thus, to compute y = (A.' *x)' use:
%
%   y = sfmult (A, x, 1,0, 0,0, 1,1) ;
%
% To compute y = (x *A.')' do the following:
%
%   y = sfmult (x, A, 1,0, 0,0, 1,1) ;
%
% The transpose of A is never computed.  Thus function requires workspace of
% size up to 4*size(A,1) if x is a matrix.  No workspace is required if x is
% a row or column vector.  At most 2*size(A,1) workspace is required if
% min(size(x)) is 2.

error ('sfmult mexFunction not found') ;
