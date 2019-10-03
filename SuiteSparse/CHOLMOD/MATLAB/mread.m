function [A, Z] = mread (filename,prefer_binary)			    %#ok
%MREAD read a sparse matrix from a file in Matrix Market format.
%
%   Example:
%   A = mread (filename)
%   [A Z] = mread (filename, prefer_binary)
%
%   Unlike MMREAD, only the matrix is returned; the file format is not
%   returned.  Explicit zero entries can be present in the file; these are not
%   included in A.  They appear as the nonzero pattern of the binary matrix Z. 
%
%   If prefer_binary is not present, or zero, a symmetric pattern-only matrix
%   is returned with A(i,i) = 1+length(find(A(:,i))) if it is present in the
%   pattern, and A(i,j) = -1 for off-diagonal entries.  If you want the original
%   Matrix Market matrix in this case, simply use A = mread (filename,1).
%
%   Compare with mmread.m at http://math.nist.gov/MatrixMarket
%
%   See also load

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('mread mexFunction not found') ;
