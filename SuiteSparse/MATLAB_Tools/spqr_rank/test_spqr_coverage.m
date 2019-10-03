function test_spqr_coverage
%TEST_SPQR_COVERAGE statement coverage test of spqr_rank functions
% Example
%   test_spqr_coverage
%
% This test exercises all but a handful of statements in the spqr_rank toolbox.
% A lot of output is generated, and some of it will contain error messages.
% This is expected, since the spqr_* functions do a lot of error checking, and
% this code exercises essentially all of those tests with many pathological
% matrices.  Sample output of this test is in private/test_spqr_coverage.txt.
%
% The only statements not tested are a few marked "untested" in the code.
% These are for a few rare error conditions which might occur, or which might
% actually be theoretically impossible.  Since we have no proof that these
% conditions cannot occur, we include code to check for them and handle the
% potential errors.  But we cannot test this error-handling code itself.
%
% After running this test, you can view the Coverage reports by going to the
% Current Folder window, selecting the Gear symbol and selecting Reports ->
% Coverage Report.  Some files report less than 100%, but this is the fault of
% MATLAB not accounting for unreachable statements.  For example, code such as
% this reports the "end" statement as untested:
%
%   if (...)
%       return
%   end
%
% Coverage is close to 100% for all files in the toolbox except for the testing/
% demonstration files (test_spqr_coverage, demo_spqr_rank and test_spqr_rank).
%
% See also demo_spqr_rank, test_spqr_rank

% Copyright 2012, Leslie Foster and Timothy A Davis.

profile clear
profile on

fprintf (['Exercises all but a handful of statements in the ', ...
    'spqr_rank toolbox\n']) ;

nfail = 0 ;

opts = struct('figures', 0, ...             % no figures
    'null_spaces', 1, ...
    'doprint', -1, ...                      % absolutely no printing
    'start_with_A_transpose', 0, ...
    'implicit_null_space_basis', 1, ...
    'repeatable', 1 , ...
    'nsvals', 1 , ...
    'get_details', 0, ...
    'tol_norm_type', 2) ;
nfail = nfail + demo_spqr_rank (3, opts) ;
opts.tol_norm_type = 1 ;
spqr_rank_opts (opts) ;    % exercise spqr_rank_opts with tol_norm_type = 1
nfail = nfail + demo_spqr_rank (3, opts) ;

A = magic (5) ;
A = A (1:4,:) ;
b = (1:4)' ;

try
    % incorrect usage: "A" not defined
    x = spqr_basic ;                                                        %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % broken stats struct
    stats.flag = 999 ;
    spqr_rank_stats (stats,1) ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: too many arguments
    x = spqr_basic (A, b, 1, opts) ;                                        %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: A and b mismatch
    c = (1:5)' ;
    x = spqr_basic (A, c) ;                                                 %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: 3rd argument not scalar
    x = spqr_basic (A, b, [1 2]) ;                                          %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: too many arguments
    [U, S, V, stats] = spqr_ssi (A, 1, opts) ;                              %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: too many arguments
    x = spqr_cod (A, b, 1, opts) ;                                          %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: too many arguments
    N = spqr_null (A, b, opts) ;                                            %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: too many arguments
    [U,S,V,stats] = spqr_ssp (A, [ ], 4, pi, opts) ;                        %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: too many arguments
    x = spqr_pinv (A, b, 1, opts) ;                                         %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % incorrect usage: A must be square
    [U, S, V, stats] = spqr_ssi (A) ;                                       %#ok
    fprintf ('usage error not caught\n') ;
    nfail = nfail + 1 ;
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

%-------------------------------------------------------------------------------
% check basic correct usage
%-------------------------------------------------------------------------------

x = spqr_basic (A, b, [ ]) ;
if (norm (A*x-b) > 1e-13)
    nfail = nfail + 1 ;
end

x = spqr_basic (A, b, eps) ;
if (norm (A*x-b) > 1e-13)
    nfail = nfail + 1 ;
end

x = spqr_basic (A, b, struct ('tol_norm_type', 1)) ;
if (norm (A*x-b) > 1e-13)
    nfail = nfail + 1 ;
end

c = spqr_ssp (A,1) ;
if (abs (c - norm (A)) > 0.01)
    nfail = nfail + 1 ;
end

opts2.get_details = 1 ;
fprintf ('\nopts for spqr_ssi with get_details = 1:\n') ;
spqr_rank_opts (opts2) ;
R = qr (A (:, 1:4)) ;
[U, S, V, stats] = spqr_ssi (R, opts2) ;                                    %#ok
c = stats.normest_R ;
spqr_rank_stats (stats) ;
if (abs (c - norm (R)) / norm (R) > 0.01)
    nfail = nfail + 1 ;
end

[s, stats] = spqr_ssi (inf) ;                                               %#ok

for t = 0:2
    opts2.get_details = t ;
    opts2.tol = 1e-6 ;
    fprintf ('\nopts for spqr_cod with get_details = %d:\n', t) ;
    spqr_rank_opts (opts2) ;
    [x, stats] = spqr_cod (A, b, opts2) ;
    e = norm (A*x-b) ;
    fprintf ('\nresults from spqr_cod with get_details = %d: err %g\n', t, e) ;
    spqr_rank_stats (stats) ;
    if (e  > 1e-12)
        nfail = nfail + 1 ;
    end

    [x stats N NT] = spqr_pinv (A, b, opts2) ;                              %#ok
    e = norm (A*x-b) ;
    fprintf ('\nresults from spqr_pinv with get_details = %d: err %g\n', t, e) ;
    spqr_rank_stats (stats, 1) ;
    if (e  > 1e-12)
        nfail = nfail + 1 ;
    end

    [x stats NT] = spqr_basic (A, b, opts2) ;                               %#ok
    e = norm (A*x-b) ;
    fprintf ('\nresults from spqr_basic with get_details = %d: err %g\n', t, e);
    spqr_rank_stats (stats, 1) ;
    if (e  > 1e-12)
        nfail = nfail + 1 ;
    end

end

C = magic (4) ; % matrix of rank 3
opts3.implicit_null_space_basis = 0 ;
for t = 0:2
    opts3.get_details = t ;
    [N stats] = spqr_null (C, opts3) ;                                      %#ok
    if (norm (C*N) > 1e-12)
        nfail = nfail + 1 ;
    end
end

[s stats] = spqr_ssp (A, [ ], 0) ;                                          %#ok
s = spqr_ssp (A, [ ], 0) ;                                                  %#ok
s = spqr_ssp (sparse (pi)) ;
if (abs (s - pi) > 1e-12)
    nfail = nfail + 1 ;
end

s = spqr_ssp (A, 10) ;                                                      %#ok

C = magic (4) ; % matrix of rank 3
B = (1:4) ;
opts6.implicit_null_space_basis = 0 ;
opts6.start_with_A_transpose = 1 ;
opts6.ssi_tol = 1e-6 ;
opts6.repeatable = 0 ;
opts6.get_details = 1 ;
fprintf ('\nopts for spqr_null with explicit basis N:\n') ;
spqr_rank_opts (opts6) ;
N1 = spqr_null (C) ;
N2 = spqr_null (C, opts6) ;
e = 0 ;
e = max (e, norm (N2'*C  - spqr_null_mult (N1, C))) ;
e = max (e, norm (N2'*C  - spqr_null_mult (N1, C, 0))) ;
e = max (e, norm (N2*B   - spqr_null_mult (N1, B, 1))) ;
e = max (e, norm (B'*N2' - spqr_null_mult (N1, B', 2))) ;
e = max (e, norm (C*N2   - spqr_null_mult (N1, C, 3))) ;
e = max (e, norm (B*N2   - spqr_null_mult (N1, B, 3))) ;

e = max (e, norm (N2'*C  - spqr_null_mult (N2, C))) ;
e = max (e, norm (N2'*C  - spqr_null_mult (N2, C, 0))) ;
e = max (e, norm (N2*B   - spqr_null_mult (N2, B, 1))) ;
e = max (e, norm (B'*N2' - spqr_null_mult (N2, B', 2))) ;
e = max (e, norm (C*N2   - spqr_null_mult (N2, C, 3))) ;
e = max (e, norm (B*N2   - spqr_null_mult (N2, B, 3))) ;

if (e > 1e-12)
    nfail = nfail + 1 ;
end

try
    % unrecognized method
    x = spqr_null_mult (N1, C, 42) ;                                        %#ok
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % unrecognized method
    x = spqr_null_mult (N2, C, 42) ;                                        %#ok
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

Ngunk = N1 ;
Ngunk.kind = 'gunk' ;

try
    % unrecognized struct
    x = spqr_null_mult (Ngunk, C, 0) ;                                      %#ok
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % unrecognized struct
    x = spqr_null_mult (Ngunk, B, 1) ;                                      %#ok
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % unrecognized struct
    x = spqr_null_mult (Ngunk, B', 2) ;                                     %#ok
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

try
    % unrecognized struct
    x = spqr_null_mult (Ngunk, C, 3) ;                                      %#ok
catch me
    fprintf ('Expected error caught: %s\n', me.message) ;
end

C = magic (4) ;
C (:,1) = 0.2 * C (:,2) + pi * C (:,3) ;
b = (1:4)' ;
opts6.get_details = 2 ;
[x,stats,N1,NT1] = spqr_cod (C,b) ;                                         %#ok
[x,stats,N2,NT2] = spqr_cod (C,b, opts6) ;                                  %#ok
fprintf ('\ndetailed stats from spqr_cod:\n') ;
spqr_rank_stats (stats, 1) ;
e = 0 ;

e = max (e, norm (N2'*C  - spqr_null_mult (N1, C))) ;
e = max (e, norm (N2'*C  - spqr_null_mult (N1, C, 0))) ;
e = max (e, norm (N2*B   - spqr_null_mult (N1, B, 1))) ;
e = max (e, norm (B'*N2' - spqr_null_mult (N1, B', 2))) ;
e = max (e, norm (C*N2   - spqr_null_mult (N1, C, 3))) ;
e = max (e, norm (B*N2   - spqr_null_mult (N1, B, 3))) ;

e = max (e, norm (N2'*C  - spqr_null_mult (N2, C))) ;
e = max (e, norm (N2'*C  - spqr_null_mult (N2, C, 0))) ;
e = max (e, norm (N2*B   - spqr_null_mult (N2, B, 1))) ;
e = max (e, norm (B'*N2' - spqr_null_mult (N2, B', 2))) ;
e = max (e, norm (C*N2   - spqr_null_mult (N2, C, 3))) ;
e = max (e, norm (B*N2   - spqr_null_mult (N2, B, 3))) ;

e = max (e, norm (NT2'*C  - spqr_null_mult (NT1, C))) ;
e = max (e, norm (NT2'*C  - spqr_null_mult (NT1, C, 0))) ;
e = max (e, norm (NT2*B   - spqr_null_mult (NT1, B, 1))) ;
e = max (e, norm (B'*NT2' - spqr_null_mult (NT1, B', 2))) ;
e = max (e, norm (C*NT2   - spqr_null_mult (NT1, C, 3))) ;
e = max (e, norm (B*NT2   - spqr_null_mult (NT1, B, 3))) ;

Nsparse = spqr_explicit_basis (N1) ;
NTsparse = spqr_explicit_basis (NT1) ;

e = max (e, norm (N2'*C  - spqr_null_mult (Nsparse, C))) ;
e = max (e, norm (N2'*C  - spqr_null_mult (Nsparse, C, 0))) ;
e = max (e, norm (N2*B   - spqr_null_mult (Nsparse, B, 1))) ;
e = max (e, norm (B'*N2' - spqr_null_mult (Nsparse, B', 2))) ;
e = max (e, norm (C*N2   - spqr_null_mult (Nsparse, C, 3))) ;
e = max (e, norm (B*N2   - spqr_null_mult (Nsparse, B, 3))) ;

e = max (e, norm (NT2'*C  - spqr_null_mult (NTsparse, C))) ;
e = max (e, norm (NT2'*C  - spqr_null_mult (NTsparse, C, 0))) ;
e = max (e, norm (NT2*B   - spqr_null_mult (NTsparse, B, 1))) ;
e = max (e, norm (B'*NT2' - spqr_null_mult (NTsparse, B', 2))) ;
e = max (e, norm (C*NT2   - spqr_null_mult (NTsparse, C, 3))) ;
e = max (e, norm (B*NT2   - spqr_null_mult (NTsparse, B, 3))) ;

Nfull = spqr_explicit_basis (N1, 'full') ;
NTfull = spqr_explicit_basis (NT1, 'full') ;

e = max (e, norm (N2'*C  - spqr_null_mult (Nfull, C))) ;
e = max (e, norm (N2'*C  - spqr_null_mult (Nfull, C, 0))) ;
e = max (e, norm (N2*B   - spqr_null_mult (Nfull, B, 1))) ;
e = max (e, norm (B'*N2' - spqr_null_mult (Nfull, B', 2))) ;
e = max (e, norm (C*N2   - spqr_null_mult (Nfull, C, 3))) ;
e = max (e, norm (B*N2   - spqr_null_mult (Nfull, B, 3))) ;

e = max (e, norm (NT2'*C  - spqr_null_mult (NTfull, C))) ;
e = max (e, norm (NT2'*C  - spqr_null_mult (NTfull, C, 0))) ;
e = max (e, norm (NT2*B   - spqr_null_mult (NTfull, B, 1))) ;
e = max (e, norm (B'*NT2' - spqr_null_mult (NTfull, B', 2))) ;
e = max (e, norm (C*NT2   - spqr_null_mult (NTfull, C, 3))) ;
e = max (e, norm (B*NT2   - spqr_null_mult (NTfull, B, 3))) ;

Nfull = spqr_explicit_basis (Nfull, 'full') ;        % doesn't change Nfull
e = max (e, norm (N2'*C  - spqr_null_mult (Nfull, C))) ;

if (e > 1e-12)
    nfail = nfail + 1 ;
end

% test spqr_wrapper
A = sparse (magic (4)) ;
[Q,R,C,p,info] = spqr_wrapper (A, [ ], eps, 'discard', 0) ;                 %#ok
if (~isequal (size (C), [4 0]) || norm (R'*R - A'*A, 1) > 1e-12)
    nfail = nfail + 1 ;
end

fprintf ('\ndefault ') ;
spqr_rank_opts
opts = spqr_rank_opts (struct, 1) ;
fprintf ('\n(default for spqr_ssp): ') ;
spqr_rank_opts (opts) ;

fprintf ('\ndescription of statistics:\n') ;
spqr_rank_stats
spqr_rank_stats ('ssi') ;
spqr_rank_stats ('ssp') ;
spqr_rank_stats ('null') ;
spqr_rank_stats (1) ;
spqr_rank_stats (2) ;

fprintf ('\n---------------------------------------------------------------\n');
fprintf ('test that illustrates the rare case of a miscalculated rank:\n') ;
install_SJget ;
Problem = SJget (182) ;
display (Problem)
A = Problem.A ;
opts8.get_details = 1;
opts8.tol = max(size(A))*eps(Problem.svals(1));
[N,stats] = spqr_null (A,opts8) ;                                           %#ok
opts8.repeatable = 0 ;
[N,stats] = spqr_null (A,opts8) ;                                           %#ok
opts8.repeatable = 384 ;
[N,stats] = spqr_null (A,opts8) ;                                           %#ok
spqr_rank_stats (stats,1) ;
if (stats.flag == 1)
    rank_svd = SJrank (Problem,stats.tol_alt) ;
else
    rank_svd = SJrank (Problem,stats.tol) ;
end
if stats.rank ~= rank_svd
   fprintf ('\nexpected rank mismatch %d %d\n', stats.rank, rank_svd) ;
end

% another rare case
fprintf ('\n---------------------------------------------------------------\n');
fprintf ('another rare case:\n') ;
Problem = SJget (240) ;
display (Problem) ;
A = Problem.A ;
m = size (A,1) ;
b = ones (m, 1) ;
opts9.repeatable = 22 ;
opts9.get_details = 1 ;
[x stats] = spqr_cod (A, b, opts9) ;                                        %#ok
spqr_rank_stats (stats,1) ;

% another rare case: early return from spqr_ssi
fprintf ('\n---------------------------------------------------------------\n');
fprintf ('another rare case:\n') ;
Problem = SJget (242) ;
display (Problem) ;
A = Problem.A ;
m = size (A,1) ;
b = ones (m, 1) ;
opts9.repeatable = 1 ;
[N stats] = spqr_null (A, opts9) ;                                          %#ok
spqr_rank_stats (stats,1) ;
x = spqr_pinv (A', b, opts9) ;                                              %#ok
[Q,R,C,p,info] = spqr_wrapper (A', [ ], eps, 'discard Q', 2) ;              %#ok
rank_spqr = size (R,1) ;
R11 = R (:, 1:rank_spqr) ;
opts9.tol = 1e-14 ;
[s stats] = spqr_ssi (R11, opts9) ;                                         %#ok
spqr_rank_stats (stats,1) ;

% test rare case in spqr_ssi
fprintf ('\n---------------------------------------------------------------\n');
fprintf ('Near-overflow in spqr_ssi, 2nd test:\n') ;
Problem = SJget (137) ;
display (Problem) ;
opts10.repeatable = 1 ;
opts10.get_details = 1 ;
A = Problem.A ;
[m n] = size (A) ;
tol = max(m,n) * eps (normest (A,0.01)) ;
for t = 0:1
    [Q,R,C,p,info] = spqr_wrapper (A, [ ], tol, 'discard Q', 2) ;           %#ok
    k = size (R,1) ;
    R = R (:, 1:k) ;
    [s, stats] = spqr_ssi (R, opts10) ;                                     %#ok
    spqr_rank_stats (stats,1) ;
    [s, stats] = spqr_ssi (R, opts10) ;                                     %#ok
    spqr_rank_stats (stats,1) ;
    A = A' ;
end

% test flag=1 case: rank correct with an alternate tolerance
fprintf ('\n---------------------------------------------------------------\n');
fprintf ('case where rank appears correct but with alternate tolerance:\n') ;
Problem = SJget (183) ;
display (Problem) ;
A = Problem.A ;
b = ones (size (A,1), 1) ;
opts11.get_details = 1 ;
opts11.repeatable = 3 ;
[x stats] = spqr_cod (A,b, opts11) ;                                        %#ok
spqr_rank_stats (stats,1) ;

if (nfail > 0)
    error ('nfail: %d\n', nfail) ;
end

%-------------------------------------------------------------------------------
% exhaustive tests on a handful of matrices
%-------------------------------------------------------------------------------

index = SJget ;

% these matrices trigger lots of special cases in the spqr_rank toolbox:
%  229 Regtools/foxgood_100
%  241 Regtools/heat_100
%  245 Regtools/i_laplace_100
%  249 Regtools/parallax_100
%  242 Regtools/heat_200
%  173 Sandia/oscil_dcop_24
%  183 Sandia/oscil_dcop_34
%  263 Regtools/wing_500
idlist = [ 215 229 241 245 249 242 173 183 263 134 ] ;

fprintf ('\nPlease wait ...\n') ;
for k = 1:length (idlist)
    id = idlist (k) ;
    fprintf ('%2d of %2d : id: %4d %s/%s\n', k, length (idlist), ...
        id, index.Group {id}, index.Name {id}) ;
    nfail = nfail + test_spqr_rank (id, 0) ;
end

if (nfail > 0)
    error ('One or more tests failed!') ;
end

fprintf ('\nAll tests passed.\n') ;

profile off

%-------------------------------------------------------------------------------
% coverage results
%-------------------------------------------------------------------------------

% This test_spqr_coverage script obtains near-100% coverage.
%
% These files obtain 100% coverage:
%
%   spqr_rank_opts.m
%   spqr_explicit_basis.m
%   private/tol_is_default
%   private/spqr_rank_order_fields
%   private/spqr_rank_form_basis
%   private/spqr_rank_deflation
%   private/spqr_rank_assign_stats
%   private/spqr_wrapper

% These files are effectively 100%.  The only non-tested statements are "end"
% statements that appear immediate after 'error' or 'break' statements, which
% are not reachable.  MATLAB should report 100% coverage in this case, but it
% doesn't.  This is a failure of MATLAB, not our code...
%
%   spqr_ssp
%   spqr_null
%   spqr_null_mult
%   spqr_basic
%   spqr_rank_stats
%   private/spqr_rank_get_inputs

% These files are not yet 100%, and "cannot" be.  These codes have a few checks
% for errors that should "never" occur.  We know of no matrices that trigger
% these conditions, but also no proof that the conditions cannot occur.  Lines
% that are untested are marked with "% untested" comments.
%
%   spqr_cod
%   spqr_ssi
%   spqr_pinv

% One part of this file is not tested (error handling if 'savepath' fails):
%
%   private/install_SJget

% These files are not fully covered, but do not need to be (test/demo code):
%
%   test_spqr_coverage
%   test_spqr_rank
%   demo_spqr_rank
%   files in the SJget folder

