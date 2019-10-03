function x = gee_its_sweet (A, b)
% GEE_ITS_SWEET solves Ax=b with just x=A\b; it doesn't get sweeter than this
%
% Example:
%
%   x = gee_its_sweet (A,b) ;       % just does x=A\b, using backslash itself
%
% See also: mldivide

% I include the copyright below ... just to silence the Help Report ...  Yes,
% it's kind of silly to copyright a one-line code :-)  ... but then I do hold
% the copyright to perhaps a third to a half of the code used internally in
% MATLAB itself, to do x=A\b (UMFPACK, CHOLMOD, AMD, COLAMD, SPQR) along with
% co-authors Iain Duff, Patrick Amestoy, John Gilbert, Esmond Ng, and Stefan
% Larimore (CHOLMOD includes other code modules co-authored by Bill Hager,
% Morris Chen, and Siva Rajamanickam, but these do not appear in x=A\b).  The
% other part of x=A\b includes LAPACK and the BLAS (Dongarra et al), a sparse
% QR by John Gilbert, many other specialized solvers by Penny Anderson and Pat
% Quillen, and MA57 by Iain Duff.  Cleve Moler and Rob Schrieber have also
% worked on backslash.  The sparse case includes iterative refinement with
% sparse backward error (algorithm by Mario Arioli, Jim Demmel, and Iain Duff)
% but code by T. Davis.  I've probably left someone out of this cast of
% thousands because I haven't seen the code for mldivide (aka backslash), just
% LAPACK, the BLAS, and my codes.

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

x = A\b ;   % I hearby call upon 250k lines of code (or so) to solve Ax=b
