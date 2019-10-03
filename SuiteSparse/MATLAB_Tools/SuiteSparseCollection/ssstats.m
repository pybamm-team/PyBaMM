function stats = ssstats (A, kind, skip_chol, skip_dmperm, Z)
%SSSTATS compute matrix statistics for the SuiteSparse Matrix Collection
% Example:
%   stats = ssstats (A, kind, skip_chol, skip_dmperm, Z)
%
% A: a sparse matrix
% kind: a string with the Problem.kind
% Z: empty, or a sparse matrix the same size as A.  Only used for
%   pattern_symmetry, nzero, and bandwidth statistics, described below.
%
% Requires amd, cholmod, RBio, and CSparse.  Computes the following
% statistics, returning them as fields in the stats struct:
%
%   nrows               number of rows
%   ncols               number of columns
%   nnz                 number of entries in A
%   RBtype              Rutherford/Boeing type
%   isBinary            1 if binary, 0 otherwise
%   isReal              1 if real, 0 if complex
%   cholcand            1 if a candidate for sparse Cholesky, 0 otherwise
%   numerical_symmetry  numeric symmetry (0 to 1, where 1=symmetric)
%   pattern_symmetry    pattern symmetry (0 to 1, where 1=symmetric)
%   nnzdiag             nnz (diag (A)) if A is square, 0 otherwise
%   nzero               nnz (Z)
%   amd_lnz             nnz(L) for chol(C(p,p)) where, C=A+A', p=amd(C)
%   amd_flops           flop count for chol(C(p,p)) where, C=A+A', p=amd(C)
%   amd_vnz             nnz in Householder vectors for qr(A(:,colamd(A)))
%   amd_rnz             nnz in R for qr(A(:,colamd(A)))
%   nblocks             # of blocks from dmperm
%   sprank              sprank(A)
%   ncc                 # of strongly connected components
%   posdef              1 if positive definite, 0 otherwise
%   isND                1 if a 2D/3D problem, 0 otherwise
%   isGraph             1 if a graph, 0 otherwise
%   lowerbandwidth      lower bandwidth, [i j]=find(A), max(0,max(i-j))
%   upperbandwidth      upper bandwidth, [i j]=find(A), max(0,max(j-i))
%   rcm_lowerbandwidth  lower bandwidth after symrcm
%   rcm_upperbandwidth  upper bandwidth after symrcm
%   xmin            smallest nonzero value
%   xmax            largest nonzero value
%
% amd_lnz and amd_flops are not computed for rectangular matrices.
%
% Ordering statistics are not computed for graphs (amd_*), since they are not
% linear systems.  For directed or undirected graphs (square matrices that
% represent graph problems), the diagonal is typically not present, but it is
% implicitly present.  Thus, sprank(A) is always equal to the # of rows, and
% nblocks is the same as ncc, for these problems.  stats.sprank and
% stats.nblocks are left as -2.
%
% The bandwidth statistics include the Z matrix.  For rectangular matrices,
% symrcm is not applicable, and the rcm_lowerbandwidth and rcm_upperbandwidth
% statistics are the same as the unpermuted versions, lowerbandwidth and
% upperbandwidth, respectively.
%
% If a statistic is not computed, it is set to -2.  If an attempt to compute
% the statistic was made but failed, it is set to -1.
%
% See also ssget, ssindex, RBtype, amd, colamd, cs_scc, cs_sqr, dmperm,
% cholmod2, symrcm

% Copyright 2006-2014, Timothy A. Davis

% Requires the SuiteSparse set of packages: CHOLMOD, RBio, CSparse

%-------------------------------------------------------------------------------
% ensure the matrix is sparse
%-------------------------------------------------------------------------------

if (~issparse (A))
    A = sparse (A) ;
end
[m n] = size (A) ;

uncomputed = -2 ;
failure = -1 ;

if (nargin < 5)
    Z = sparse (m,n) ;
    AZ = A ;
else
    AZ = A + Z ;
    if (nnz (AZ) ~= nnz (A) + nnz (Z))
        error ('A and Z overlap!')
    end
end

if (nargin < 4)
    skip_dmperm = 0 ;
end
if (nargin < 3)
    skip_chol = 0 ;
end
if (nargin < 2)
    kind = '' ;
end

%-------------------------------------------------------------------------------
% basic stats
%-------------------------------------------------------------------------------

tic ;
stats.nrows = m ;
stats.ncols = n ;
stats.nnz = nnz (A) ;
stats.RBtype = RBtype (AZ) ;                    % Rutherford/Boeing type
stats.isBinary = (stats.RBtype (1) == 'p') ;
stats.isReal = (stats.RBtype (1) ~= 'c') ;

fprintf ('RBtype: %s time: %g\n', stats.RBtype, toc) ;

%-------------------------------------------------------------------------------
% symmetry and Cholesky candidacy
%-------------------------------------------------------------------------------

% get the symmetry
tic ;
[s xmatched pmatched nzoffdiag nnzdiag] = spsym (A) ;
if (m ~= n)
    stats.numerical_symmetry = 0 ;
    stats.pattern_symmetry = 0 ;
elseif (nzoffdiag > 0)
    stats.numerical_symmetry = xmatched / nzoffdiag ;
    stats.pattern_symmetry = pmatched / nzoffdiag ;
else
    stats.numerical_symmetry = 1 ;
    stats.pattern_symmetry = 1 ;
end
psym_A = stats.pattern_symmetry ;   % symmetry of the pattern of A (excluding Z)
stats.nnzdiag = nnzdiag ;
stats.cholcand = (s >= 6) ;         % check if Cholesky candidate
stats.nzero = nnz (Z) ;

fprintf ('cholcand: %d\n', stats.cholcand) ;
fprintf ('numerical_symmetry: %g pattern_symmetry: %g time: %g\n', ...
    stats.numerical_symmetry, stats.pattern_symmetry, toc) ;

% recompute the pattern symmetry with Z included
tic ;
if (m == n && stats.nzero > 0)
    [s xmatched pmatched nzoffdiag] = spsym (AZ) ;
    if (nzoffdiag > 0)
        stats.pattern_symmetry = pmatched / nzoffdiag ;
    else
        stats.pattern_symmetry = 1 ;
    end
end
fprintf ('stats with A+Zeros:\n') ;
fprintf ('numerical_symmetry: %g pattern_symmetry: %g time: %g\n', ...
    stats.numerical_symmetry, stats.pattern_symmetry, toc) ;

%-------------------------------------------------------------------------------
% bandwidth (includes explicit zeros)
%-------------------------------------------------------------------------------

[i j] = find (AZ) ;
stats.lowerbandwidth = max (0, max (i-j)) ;
stats.upperbandwidth = max (0, max (j-i)) ;
clear i j
fprintf ('lo %d up %d ', ...
    stats.lowerbandwidth, stats.upperbandwidth) ;
% now with symrcm, if the matrix is square
stats.rcm_lowerbandwidth = stats.lowerbandwidth ;
stats.rcm_upperbandwidth = stats.upperbandwidth ;
if (m == n)
    try
        p = symrcm (AZ) ;
        [i j] = find (AZ (p,p)) ;
        stats.rcm_lowerbandwidth = max (0, max (i-j)) ;
        stats.rcm_upperbandwidth = max (0, max (j-i)) ;
    catch
        fprintf ('================ symrcm failed ') ;
        stats.rcm_lowerbandwidth = failure ;
        stats.rcm_upperbandwidth = failure ;
    end
    fprintf ('rcm: lo %d up %d', ...
        stats.rcm_lowerbandwidth, stats.rcm_upperbandwidth) ;
end
fprintf ('\n') ;
clear AZ i j p

%-------------------------------------------------------------------------------
% isND
%-------------------------------------------------------------------------------

s = 0 ;
if (~isempty (strfind (kind, 'structural')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'fluid')))
    s = 1 ;
elseif (~isempty (strfind (kind, '2D')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'reduction')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'electromagnetics')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'semiconductor')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'thermal')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'materials')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'acoustics')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'vision')))
    s = 1 ;
elseif (~isempty (strfind (kind, 'robotics')))
    s = 1 ;
end
stats.isND = s ;

fprintf ('isND %d\n', stats.isND) ;

%-------------------------------------------------------------------------------
% determine if this is a graph (directed, undirected, or bipartite)
%-------------------------------------------------------------------------------

if (~isempty (strfind (kind, 'graph')) && isempty (strfind (kind, 'graphics')))
    % this is a directed, undirected, or bipartite graph.
    % it might also be a multigraph, and weighted or unweighted.
    stats.isGraph = 1 ;
else
    stats.isGraph = 0 ;
end

fprintf ('isGraph %d\n', stats.isGraph) ;

%-------------------------------------------------------------------------------
% determine if positive definite
%-------------------------------------------------------------------------------

fprintf ('start Cholesky\n') ;
tic ;
if (~stats.cholcand)

    % not a candidate for Cholesky, so it cannot be positive definite
    fprintf ('not a Cholesky candidate\n') ;
    stats.posdef = 0 ;

elseif (stats.isBinary)

    % For all symmetric binary matrices:  only identity matrices are positive
    % definite.  All others are indefinite.  Since at this point, A is a
    % Cholesky candidate, and thus we know that A is symmetric with a zero-free
    % diagonal.  So just a quick check of nnz(A) is needed.
    % See: McKay et al, "Acyclic Digraphs and Eigenvalues of (0,1)-Matrices",
    % Journal of Integer Sequences, Vol. 7 (2004), Article 04.3.3.
    % http://www.cs.uwaterloo.ca/journals/JIS/VOL7/Sloane/sloane15.html

    stats.posdef = (stats.nnz == stats.nrows) ;

elseif (skip_chol)

    % Cholesky was skipped
    fprintf ('skip Cholesky\n') ;
    stats.posdef = uncomputed ;

else

    % try chol
    try
        [x, cstats] = cholmod2 (A, ones (stats.ncols,1)) ;
        rcond = cstats (1) ;
        fprintf ('rcond: %g\n', rcond) ;
        stats.posdef = (rcond > 0) ;
    catch
        % chol failed
        disp (lasterr) ;
        fprintf ('sparse Cholesky failed\n') ;
        stats.posdef = failure ;
    end
    clear x cstats
end

fprintf ('posdef: %d time: %g\n', stats.posdef, toc) ;

%-------------------------------------------------------------------------------
% transpose A if m < n, for ordering methods
%-------------------------------------------------------------------------------

tic ;
if (m < n)
    try
        A = A' ;            % A is now tall and thin, or square
    catch
        disp (lasterr) ;
        fprintf ('transpose failed...\n') ;
        return ;
    end
    [m n] = size (A) ;
end
if (~isreal (A))
    try
        A = spones (A) ;
    catch
        disp (lasterr) ;
        fprintf ('conversion from complex failed...\n') ;
        return ;
    end
end
fprintf ('computed A transpose if needed, time: %g\n', toc) ;

%-------------------------------------------------------------------------------
% order entire matrix with AMD, if square
%-------------------------------------------------------------------------------

if (m == n && ~stats.isGraph)

    tic ;
    try
        if (psym_A < 1)
            C = A|A' ;      % A has unsymmetric pattern, so symmetrize it
        else
            C = A ;         % A already has symmetric pattern
        end
    catch
        disp (lasterr) ;
        fprintf ('A+A'' failed\n') ;
    end
    fprintf ('computed A+A'', time: %g\nstart AMD\n', toc) ;

    tic ;
    try
        p = amd (C) ;
        c = symbfact (C (p,p)) ;
        stats.amd_lnz = sum (c) ;           % nnz (chol (C))
        stats.amd_flops = sum (c.^2) ;      % flop counts for chol (C)
    catch
        disp (lasterr) ;
        fprintf ('amd failed\n') ;
        stats.amd_lnz = failure ;
        stats.amd_flops = failure ;
    end
    clear p c C
    fprintf ('AMD lnz %d flops %g time: %g\n', ...
        stats.amd_lnz, stats.amd_flops, toc) ;

else

    % not computed if rectangular, or for graph problems
    stats.amd_lnz = uncomputed ;
    stats.amd_flops = uncomputed ;
    fprintf ('AMD skipped\n') ;

end

%-------------------------------------------------------------------------------
% order entire matrix with COLAMD, for LU bounds
%-------------------------------------------------------------------------------

if (~stats.isGraph)

    fprintf ('start colamd:\n') ;
    tic ;
    try
        q = colamd (A) ;
        [vnz,rnz] = cs_sqr (A (:,q)) ;
        stats.amd_rnz = rnz ;   % nnz (V), upper bound on L, for A(:,colamd(A))
        stats.amd_vnz = vnz ;   % nnz (R), upper bound on U, for A(:,colamd(A))
    catch
        disp (lasterr) ;
        fprintf ('colamd2 and cs_sqr failed\n') ;
        stats.amd_vnz = failure ;
        stats.amd_rnz = failure ;
    end
    clear q
    fprintf ('COLAMD rnz %d vnz %d time: %g\n', ...
        stats.amd_rnz, stats.amd_vnz, toc) ;

else

    % not computed for graph problems
    stats.amd_rnz = uncomputed ;
    stats.amd_vnz = uncomputed ;
    fprintf ('COLAMD skipped\n') ;

end

%-------------------------------------------------------------------------------
% strongly connected components
%-------------------------------------------------------------------------------

tic ;
fprintf ('start scc:\n') ;
try
    % find the # of strongly connected components of the graph of a square A,
    % or # of connected components of the bipartite graph of a rectangular A.
    if (m == n)
        [p r] = cs_scc (A) ;
    else
        [p r] = cs_scc (spaugment (A)) ;
    end
    stats.ncc = length (r) - 1 ;
    clear p r
catch
    disp (lasterr) ;
    fprintf ('cs_scc failed\n') ;
    stats.ncc = failure ;
end
fprintf ('scc %d, time: %g\n', stats.ncc, toc) ;

%-------------------------------------------------------------------------------
% Dulmage-Mendelsohn permutation, and order each block
%-------------------------------------------------------------------------------

tic ;
if (m == n && stats.isGraph && isempty (strfind (kind, 'bipartite')))
    % for directed and undirected graphs (square matrices), the diagonal is
    % implicitly present.  Thus, nblocks is the same as ncc, and the graph has
    % full sprank.  dmperm *is* computed for bipartite graphs, however.
    skip_dmperm = 1 ;
end

if (skip_dmperm)

    fprintf ('skip dmperm\n') ;
    stats.nblocks = uncomputed ;
    stats.sprank = uncomputed ;

else

    try
        % find the Dulmage-Mendelsohn decomposition
        fprintf ('start dmperm:\n') ;
        % [p,q,r,s,cc,rr] = cs_dmperm (A) ;
        [p,q,r,s,cc,rr] = dmperm (A) ;
        nblocks = length (r) - 1 ;
        stats.nblocks = nblocks ;   % # of blocks in block-triangular form
        stats.sprank = rr(4)-1 ;    % structural rank
    catch
        disp (lasterr) ;
        fprintf ('dmperm failed\n') ;
        stats.nblocks = failure ;
        stats.sprank = failure ;
    end
end

fprintf ('nblocks %d\n', stats.nblocks) ;
fprintf ('sprank %d, time: %g\n', stats.sprank, toc) ;

%-------------------------------------------------------------------------------
% xmin and xmax
%-------------------------------------------------------------------------------

[i,j,x] = find (A) ;
stats.xmin = min (x) ;
stats.xmax = max (x) ;
fprintf ('xmin %32.16g xmax %32.16g\n', stats.xmin, stats.xmax) ;
if (stats.xmin == 0 || stats.xmax == 0)
    error ('explicit zeros in the matrix!') ;
end

%-------------------------------------------------------------------------------

fprintf ('ssstats done\n') ;
