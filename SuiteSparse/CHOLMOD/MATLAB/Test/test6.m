function test6
%TEST6 test sparse with large matrix, both real and complex
% compare times with MATLAB
% Example:
%   test6
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test6: test sparse with large matrix, both real and complex\n') ;

for do_complex = 0:1

    fprintf ('do_complex = %d\n', do_complex) ;
    randn ('state', 0) ;
    rand  ('state', 0) ;

    % Prob = UFget (437)
    Prob = UFget (749)							    %#ok
    A = Prob.A ;
    [m n] = size (A) ;							    %#ok

    if (do_complex)
%	A = A + 1i*sprand(A) ;
%	patch for MATLAB 7.2
	A = A + sparse(1:m,1:m,1i)*sprand(A) ;
    end

    tic
    [i j x] = find (A) ;
    t = toc ;
    fprintf ('find time %8.4f\n', t) ;

    % tic
    % [i1 j1 x1] = cholmod_find (A) ;
    % t = toc ;
    % fprintf ('cholmod_find time %8.4f (for testing only, it should be slow)\n', t) ;

    % if (any (i ~= i1))
    %     error ('i!') ;
    % end
    % if (any (j ~= j1))
    %     error ('j!') ;
    % end
    % if (any (x ~= x1))
    %     error ('x!') ;
    % end

    [m n ] = size (A) ;

    tic ;
    B = sparse2 (i,j,x,m,n) ;
    t1 = toc ;
    tic ;
    C = sparse (i,j,x,m,n) ;
    t2 = toc ;

    fprintf ('dtri time: cholmod2 %8.6f  matlab %8.6f\n', t1, t2) ;

    err = norm(A-B,1) ;
    if (err > 0)
	error ('dtri2 1') ;
    end

    err = norm(A-C,1) ;
    if (err > 0)
	error ('dtri2 1') ;
    end

    nz = length (x) ;
    p = randperm (nz) ;

    i2 = i(p) ;
    j2 = j(p) ;
    x2 = x(p) ;							    %#ok

    tic ;
    B = sparse2 (i,j,x,m,n) ;
    t1 = toc ;
    tic ;
    C = sparse (i,j,x,m,n) ;
    t2 = toc ;

    fprintf ('dtri time: cholmod2 %8.6f  matlab %8.6f (jumbled)\n', t1, t2) ;

    err = norm(A-B,1) ;
    if (err > 0)
	error ('dtri2 2') ;
    end

    err = norm(A-C,1) ;
    if (err > 0)
	error ('dtri2 1') ;
    end

    ii = [i2 ; i2] ;
    jj = [j2 ; j2] ;
    xx = rand (2*nz,1) ;
    if (do_complex)
	xx = xx + 1i* rand (2*nz,1) ;
    end

    tic ;
    D = sparse2 (ii,jj,xx,m,n) ;
    t1 = toc ;
    tic ;
    C = sparse (ii,jj,xx,m,n) ;
    t2 = toc ;
    err = norm (C-D,1) ;
    if (err > 0)
	error ('dtri2 3') ;
    end
    fprintf ('dtri time: cholmod2 %8.6f  matlab %8.6f (duplicates)\n', t1, t2) ;

    fprintf ('length %d nz %d\n', length (xx), nnz(D)) ;

    i2 = min (ii,jj) ;
    j2 = max (ii,jj) ;

    tic ;
    E = sparse2 (i2,j2,xx,n,n) ;
    t1 = toc ;
    tic ;
    F = sparse (i2, j2, xx, n, n) ;
    t2 = toc ;
    err = norm (E-F,1)							    %#ok
    if (err > 1e-13)
	error ('dtri2 4') ;
    end
    fprintf ('dtri time: cholmod2 %8.6f  matlab %8.6f (upper)\n', t1, t2) ;

    i2 = max (ii,jj) ;
    j2 = min (ii,jj) ;

    tic ;
    E = sparse2 (i2,j2,xx,n,n) ;
    t1 = toc ;
    tic ;
    F = sparse (i2, j2, xx, n, n) ;
    t2 = toc ;
    err = norm (E-F,1)							    %#ok
    if (err > 1e-13)
	error ('dtri2 5') ;
    end
    fprintf ('dtri time: cholmod2 %8.6f  matlab %8.6f (lower)\n', t1, t2) ;

    [ignore, i] = sort (ii) ;
    ii = ii (i) ;
    jj = jj (i) ;
    xx = xx (i) ;

    tic ;
    D = sparse2 (ii,jj,xx,m,n) ;
    t1 = toc ;
    tic ;
    C = sparse (ii,jj,xx,m,n) ;
    t2 = toc ;
    err = norm (C-D,1) ;
    if (err > 0)
	error ('dtri2 6') ;
    end
    fprintf ('dtri time: cholmod2 %8.6f  matlab %8.6f (sorted, dupl)\n', t1, t2) ;

end

fprintf ('test6 passed\n') ;
