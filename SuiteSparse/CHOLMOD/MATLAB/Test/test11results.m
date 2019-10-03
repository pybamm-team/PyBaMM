function test11results
%TEST11RESULTS analyze results from test11.m
% Example:
%   test11results
% See also test11, cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

load Results
index = UFget ;

c = E1(1:kkk) < 1 & T1(1:kkk) > 0 ;
m = E2(1:kkk) < 1 & T2(1:kkk) > 0 ;
cgood = find (c) ;	%#ok
mgood = find (m) ;	%#ok
good  = find (c | m) ;
bad = find (~(c|m)) ;

fl_per_lnz = FL(1:kkk) ./ LNZ(1:kkk) ;
speedup = T1(1:kkk) ./ T2(1:kkk) ;

[ignore ii] = sort (fl_per_lnz (good)) ;
good = good (ii) ;

fprintf ('MATLABtime CHOLMOD(time,flop,nnz(L)) speedup problem\n') ;
for k = good
    i = f (k) ;
%    fprintf ('%4d: t1 %10.2f t2 %10.2f fl %6.1e lnz %6.1e   %s/%s\n', ...
%	i, T1(k), T2(k), FL(k), LNZ(k), index.Group{i}, index.Name{i}) ;
     fprintf ('%10.4f %10.4f  %6.1e  %6.1e  %5.2f   %s/%s\n', ...
	T1(k), T2(k), FL(k), LNZ(k), speedup(k), ...
	index.Group{i}, index.Name{i}) ;
end

fprintf ('\nfailed in both:\n') ;
for k = bad
     i = f (k) ;
     fprintf ('%10.4f %10.4f  %6.1e  %6.1e  %5.2f   %s/%s\n', ...
	T1(k), T2(k), FL(k), LNZ(k), speedup(k), ...
	index.Group{i}, index.Name{i}) ;
end

clf
loglog (fl_per_lnz (good), speedup (good), 'x') ;
axis ([1 4000 .1 50]) ;
xlabel ('Cholesky flop count / nnz(L)') ;
ylabel ('(MATLAB x=A\\b time) / (CHOLMOD time)') ;

drawnow
