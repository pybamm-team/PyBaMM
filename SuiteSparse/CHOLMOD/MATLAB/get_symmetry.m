function result = get_symmetry (A,quick)
%GET_SYMMETRY: does the same thing as the spsym mexFunction.
% It's just a lot slower and uses much more memory.  This function
% is meant for testing and documentation only.
[m n] = size (A) ;
if (m ~= n)
    result = 1 ;            % rectangular
    return
end
if (nargin < 2)
    quick = 0 ;
end
d = diag (A) ;
posdiag = all (real (d) > 0) & all (imag (d) == 0) ;
if (quick & ~posdiag)
    result = 2 ;            % Not a candidate for sparse Cholesky.
elseif (~isreal (A) & nnz (A-A') == 0)
    if (posdiag)
        result = 7 ;        % complex Hermitian, with positive diagonal
    else
        result = 4 ;        % complex Hermitian, nonpositive diagonal
    end
elseif (nnz (A-A.') == 0)
    if (posdiag)
        result = 6 ;        % symmetric with positive diagonal
    else
        result = 3 ;        % symmetric, nonpositive diagonal
    end
elseif (nnz (A+A.') == 0)
    result = 5 ;            % skew symmetric
else
    result = 2 ;            % unsymmetric
end

