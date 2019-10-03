function test20
%TEST20 test symbfact2, cholmod2, and lu on a few large matrices
% Example:
%   test20
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test20: test symbfact2, cholmod2, and lu on a few large matrices\n') ;

unsym = [409 899 901 291 827] ;						    %#ok
spd = [813 817] ;
% f = [ unsym spd ] ;
f = spd ;
spparms ('spumoni',0) ;

for i = f
    Prob = UFget (i)							    %#ok
    A = Prob.A ;
    clear Prob ;
    n = size (A,1) ;
    b = A*ones (n,1) ;
    if (any (i == spd))
	p = amd2 (A) ;
	count = symbfact2 (A (p,p)) ;
	count2 = symbfact (A (p,p)) ;
	if (any (count ~= count2))
	    error ('!') ;
	end
	fl = sum (count.^2) ;
	lnz = sum (count) ;
	unz = lnz ;
	tic
	x = cholmod2 (A,b) ;
	t = toc ;
    else
	% spparms ('spumoni',2) ;
	[L, U, P, Q] = lu (A) ;						    %#ok
	% fl = luflop (L,U) ;
	Lnz = full (sum (spones (L))) - 1 ;
	Unz = full (sum (spones (U')))' - 1 ;
	fl = 2*Lnz*Unz + sum (Lnz) ;
	lnz = nnz(L) ;
	unz = nnz(U) ;
	tic
	x=A\b ;
	t = toc ;
    end
    err = norm (A*x-b,1) ;
    clear L U P Q A x b
    fprintf ('lnz %d unz %d nnz(L+U) %d fl %g gflop %g\n t %g err %e\n', ...
	lnz, unz, lnz+unz-n, fl, 1e-9*fl/t, t, err) ;
    % pause
end

