function test12 (nmat)
%TEST12 test etree2 and compare with etree
% Example:
%   test12(nmat)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test12: test etree2 and compare with etree\n') ;

index = UFget ;

% only test matrices with nrows = 109000 or less.  large ones nearly always
% cause a MATLAB segfault.
% f = find (index.nrows < 109000) ;
f = 1:length (index.nrows) ;

% sort by row dimension
[ignore i] = sort (index.nrows (f)) ;
f = f (i) ;

if (nargin > 0)
    nmat = max (0,nmat) ;
    nmat = min (nmat, length (f)) ;
    f = f (1:nmat) ;
end

% MATLAB 7.0 (R14, sp2, linux) etree gives a segfault for these matrices:
s_skip =   [ 803 374 1287 1311 1308 957 958 1257 955 761 1282 1230 1256 ...
	    924 1302 537 820 821 822 1258 ...
	    844 845 1238 804 939 1270 1305 1208 1209 290 879 928 1307 1244 ...
	    1275 1276 1296 885 1269 959 542 1290 ] ;

sym_skip =  s_skip  ;

p_sym_skip =   1296  ;
p_symt_skip =  1296  ;

symt_skip= [ s_skip 592 593 809 ] ;

rowcol_skip = [ 646 ...
	     1224 803 588 589 374 1287 562 563 801 1311 1246 951 1308 950 ...
	     957 958 800 1257 564 565 955 761 1282 590 591 1230 1256 952 566 ...
	     567 924 1302 1293 1294 537 820 821 822 1306 849 1258 592 593 ...
	     1225 844 845 1226 1238 1227 804 939 1270 752 753 1305 809 1228 ...
	     1208 1209 1291 1292 1300 856 1229 290 879 928 1307 857 1244  ...
	     1275 1276 1296 885 858 859 1269 1263 959 542 1290 ] ;

col_skip = [ rowcol_skip 647 612 610 648 799 651 652 750 751 640 ] ;

row_skip= [ rowcol_skip 903 373 ] ;

% f = f (find (f == 1290) : end) ;

fprintf ('Matrices to test: %d\n', length (f)) ;


for i = f

    % try

	Problem = UFget (i) ;
	A = spones (Problem.A) ;
	[m n] = size (A) ;
	fprintf ('\n%4d: %-20s nrow: %6d ncol: %6d nnz: %10d\n', ...
	    i, Problem.name, m, n, nnz(A)) ;

	% if (max (m,n) < 500)
	%    A = full (A) ;
	% end

	% warmup, for accurate timing
	etree (sparse (1)) ;
	etree2 (sparse (1)) ;
	amd2 (sparse (1)) ;

	% test column etree
	skip = any (i == col_skip) | m > 109000 ;			    %#ok
	if (~skip)
	    tic ;
	    [parent post] = etree (A, 'col') ;
	    t1 = toc ;
	else
	    t1 = Inf ;
	end

	tic ;
	[my_parent my_post] = etree2 (A, 'col') ;
	t2 = toc ;

	if (~skip)
	    if (any (parent ~= my_parent))
		error ('parent invalid!') ;
	    end
	end
	fprintf ('etree(A,''col''): %8.4f  %8.4f              speedup %8.2f  ',...
		t1, t2, t1/t2);
	if (~skip)
	    if (any (post ~= my_post))
		fprintf ('postorder differs') ;
	    end
	end
	fprintf ('\n') ;

	% test row etree
	skip = any (i == row_skip) | m > 109000 ;			    %#ok
	if (~skip)
	    tic ;
	    [parent post] = etree (A', 'col') ;
	    t1 = toc ;
	else
	    t1 = Inf ;
	end

	tic ;
	[my_parent my_post] = etree2 (A, 'row') ;
	t2 = toc ;

	if (~skip)
	    if (any (parent ~= my_parent))
		error ('parent invalid!') ;
	    end
	end
	fprintf ('etree(A,''row''): %8.4f  %8.4f              speedup %8.2f  ',...
		t1, t2, t1/t2);
	if (~skip)
	    if (any (post ~= my_post))
		fprintf ('postorder differs') ;
	    end
	end
	fprintf ('\n') ;



	if (m == n)

	    for trial = 1:2

		if (trial == 1)
		    skip1 = any (i == sym_skip) | m > 109000 ;		    %#ok
		    skip2 = any (i == symt_skip) | m > 109000 ;		    %#ok
		else
		    skip1 = any (i == p_sym_skip) | m > 109000 ;	    %#ok
		    skip2 = any (i == p_symt_skip) | m > 109000 ;	    %#ok
		    fprintf ('after amd:\n') ;
		    p = amd2 (A) ;  % use AMD from SuiteSparse
		    A = A (p,p) ;
		end

		% test symmetric etree, using triu(A)
		if (~skip1)
		    tic ;
		    [parent post] = etree (A) ;
		    t1 = toc ;
		else
		    t1 = Inf ;
		end

		tic ;
		[my_parent my_post] = etree2 (A) ;
		t2 = toc ;

		if (~skip1)
		    if (any (parent ~= my_parent))
			error ('parent invalid!') ;
		    end
		end
		fprintf ('etree(A):       %8.4f  %8.4f              speedup %8.2f  ',...
			t1, t2, t1/t2);
		if (~skip1)
		    if (any (post ~= my_post))
			fprintf ('postorder differs') ;
		    end
		end
		fprintf ('\n') ;

		% test symmetric etree, using tril(A)
		if (~skip2)
		    tic ;
		    [parent post] = etree (A') ;
		    t1 = toc ;
		else
		    t1 = Inf ;
		end

		tic ;
		[my_parent my_post] = etree2 (A, 'lo') ;
		t2 = toc ;

		if (~skip2)
		    if (any (parent ~= my_parent))
			error ('parent invalid!') ;
		    end
		end
		fprintf('etree(A''):      %8.4f  %8.4f              speedup %8.2f  ',...
			t1, t2, t1/t2);
		if (~skip2)
		    if (any (post ~= my_post))
			fprintf ('postorder differs') ;
		    end
		end
		fprintf ('\n') ;

	    end

	end

    % catch
	% fprintf ('%d: failed\n', i) ;
    % end
end

fprintf ('test12 passed\n') ;
