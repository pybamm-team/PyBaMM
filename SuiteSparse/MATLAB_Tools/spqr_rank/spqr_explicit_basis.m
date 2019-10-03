function Nexp = spqr_explicit_basis (N, type)
%SPQR_EXPLICIT_BASIS converts a null space basis to an explicit matrix
%
% Convert a orthonormal null space bases stored implicitly and created
% by spqr_basic, spqr_null, spqr_pinv, or spqr_cod to an an explicit
% sparse, or optionally full, matrix.  If the input is not a implicit null
% space bases the input is returned unchanged.
%
% Examples:
%    A = sparse(gallery('kahan',100));
%    N = spqr_null(A);                  % creates an implicit null space basis
%    Nexp = spqr_explicit_basis (N) ;         % converts to a sparse matrix
%    Nexp = spqr_explicit_basis (N,'full') ;  % converts to a dense matrix
%
% Note that the dense matrix basis will require less memory than the implicit
% basis if whos_N.bytes > ( prod(size(N.X)) * 8 ) where whos_N = whos('N').
%
% See also spqr_basic, spqr_null, spqr_cod, spqr_pinv, spqr_null_mult.

% Copyright 2012, Leslie Foster and Timothy A. Davis

is_implicit_basis = ...
    isstruct(N) && isfield(N,'Q') && isfield(N,'X') ;

if is_implicit_basis && nargin == 1
    Nexp = spqr_null_mult(N,speye(size(N.X,2)),1) ;
elseif is_implicit_basis && nargin == 2 && strcmp(type,'full')
    % Nexp = spqr_null_mult(N,eye(size(N.X,2)),1) ; % slow for large nullity
    Nexp = spqr_null_mult(N,speye(size(N.X,2)),1) ;
    Nexp = full(Nexp) ;
else
    Nexp = N ;
end

