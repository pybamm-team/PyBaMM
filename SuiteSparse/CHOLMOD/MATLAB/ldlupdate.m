function LD = ldlupdate (LD,C,updown)					    %#ok
%LDLUPDATE multiple-rank update or downdate of a sparse LDL' factorization.
%
%   On input, LD contains the LDL' factorization of A (L*D*L'=A or A(q,q)).
%   The unit-diagonal of L is not stored.  In its place is the diagonal matrix
%   D.  LD can be computed using the CHOLMOD mexFunctions:
%
%       LD = ldlchol (A) ;
%   or
%       [LD,p,q] = ldlchol (A) ;
%
%   With this LD, either of the following MATLAB statements,
%
%   Example:
%       LD = ldlupdate (LD,C)
%       LD = ldlupdate (LD,C,'+')
%
%   return the LDL' factorization of A+C*C' or A(q,q)-C*C' if LD holds the LDL'
%   factorization of A(q,q) on input.  For a downdate:
%
%       LD = ldlupdate (LD,C,'-')
%
%   returns the LDL' factorization of A-C*C' or A(q,q)-C*C'.
%
%   LD and C must be sparse and real.  LD must be square, and C must have the
%   same number of rows as LD.  You must not modify LD in MATLAB (see the
%   WARNING below).
%
%   Note that if C is sparse with few columns, most of the time spent in this
%   routine is taken by copying the input LD to the output LD.  If MATLAB
%   allowed mexFunctions to safely modify its inputs, this mexFunction would
%   be much faster, since not all of LD changes.
%
%   See also LDLCHOL, LDLSPLIT, LDLSOLVE, CHOLUPDATE
%
%   ===========================================================================
%   =============================== WARNING ===================================
%   ===========================================================================
%   MATLAB drops zero entries from its sparse matrices.  LD can contain
%   numerically zero entries that are symbolically present in the sparse matrix
%   data structure.  These are essential for ldlupdate and ldlsolve to work
%   properly, since they exploit the graph-theoretic structure of a sparse
%   Cholesky factorization. If you modify LD in MATLAB, those zero entries may
%   get dropped and the required graph property will be destroyed.  In this
%   case, ldlupdate and ldlsolve will fail catastrophically (possibly with a
%   segmentation fault, terminating MATLAB).  It takes much more time to ensure
%   this property holds than the time it takes to do the update/downdate or the
%   solve, so ldlupdate and ldlsolve simply assume the propertly holds.
%   ===========================================================================

%   Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

error ('ldlupdate mexFunction not found') ;

