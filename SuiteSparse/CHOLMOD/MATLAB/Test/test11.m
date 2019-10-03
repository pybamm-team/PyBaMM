function test11 (nmat)
%TEST11 compare CHOLMOD and MATLAB, save results in Results.mat
% also tests analyze
% Example:
%   test11(nmat)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test11 : compare CHOLMOD and MATLAB, save results in Results.mat\n');

rand ('state',0) ;
randn ('state',0) ;

index = UFget ;
f = find (index.posdef) ;
[ignore i] = sort (index.nrows (f)) ;
f = f (i) ;
clear ignore

% start after nd6k
% f = f ((find (f == 937) + 1):end) ;

skip = [937:939 1202:1211] ;

if (nargin > 0)
    nmat = max (0,nmat) ;
    nmat = min (nmat, length (f)) ;
    f = f (1:nmat) ;
end

fprintf ('test matrices sorted by dimension:\n') ;
for i = f
    if (any (i == skip))
	continue
    end
    fprintf ('%4d: %-20s %-20s %12d %d\n', i,  ...
	index.Group {i}, index.Name {i}, index.nrows (i), index.posdef (i)) ;
end

kk = 0 ;
nmat = length (f) ;
T1 = zeros (1,nmat) ;	% matlab time
T2 = zeros (1,nmat) ;	% cholmod2 time
E1 = zeros (1,nmat) ;	% matlab residual
E2 = zeros (1,nmat) ;	% cholmod2 residual
FL = zeros (1,nmat) ;	% cholmod2 flop count
LNZ = zeros (1,nmat) ;	% cholmod2 lnz

for kkk = 1:length(f)

    nn = f (kkk) ;

    if (any (nn == skip))
	continue
    end

    % try

	fprintf ('\n%3d: %s/%s\n', nn, index.Group {nn}, index.Name {nn}) ;
	Prob = UFget (nn) ;
	A = Prob.A ;
	clear Prob
	n = size (A,1) ;
	b = rand (n,1) ;

	% analyze
	[p count] = analyze (A) ;
	% LDL' flop count
	% fl = sum ((count-1).*(count-1) + 2*(count-1)) ;
	% LL' flop count
	fl = sum (count.^2) ;
	lnz = sum (count) ;
	fprintf ('n %d lnz %g fl %g\n', n, lnz, fl) ;
	clear p count

	% try
	    k2 = 0 ;
	    t2 = 0 ;
	    while (t2 < 1)
		tic
		x = cholmod2 (A,b) ;
		t = toc ;
		t2 = t2 + t ;
		k2 = k2 + 1 ;
	    end
	    t2 = t2 / k2 ;
	    e2 = norm (A*x-b,1) ;
	% catch
	%   e2 = Inf ;
	%   k2 = Inf ;
	%   t2 = Inf ;
	% end
	fprintf ('cholmod2: t: %10.5f e: %6.1e  mflop %6.0f\n', ...
	    t2, e2, 1e-6 * fl / t2) ;

	% try
	    k1 = 0 ;
	    t1 = 0 ;
	    while (t1 < 1)
		tic
		x = A\b ;
		t = toc ;
		t1 = t1 + t ;
		k1 = k1 + 1 ;
	    end
	    t1 = t1 / k1 ;
	    e1 = norm (A*x-b,1) ;
	% catch
	%   e1 = Inf ;
	%   k1 = Inf ;
	%   t1 = Inf ;
	% end
	fprintf ('matlab:  t: %10.5f e: %6.1e  mflop %6.0f', ...
	    t1, e1, 1e-6 * fl / t1) ;

	fprintf ('   cholmod2 speedup: %5.1f\n', t1/t2) ;

	kk = kk + 1 ;
	T1 (kk) = t1 ;
	T2 (kk) = t2 ;
	E1 (kk) = e1 ;
	E2 (kk) = e2 ;
	FL (kk) = fl ;
	LNZ (kk) = lnz ;
	%%% save Results T1 T2 E1 E2 FL LNZ f kkk

    % catch
    %	fprintf (' failed\n') ;
    % end

    clear A x b

end

% test11results
fprintf ('test11 passed\n') ;
