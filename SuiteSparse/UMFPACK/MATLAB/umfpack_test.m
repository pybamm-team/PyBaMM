function umfpack_test (nmat)
%UMFPACK_TEST for testing umfpack (requires ssget)
%
% Example:
%   umfpack_test
%   umfpack_test (100)  % runs the first 100 matrices
% See also umfpack

% Copyright 1995-2007 by Timothy A. Davis.

index = ssget ;

f = find (index.nrows == index.ncols) ;
[ignore, i] = sort (index.nrows (f)) ;
f = f (i) ;

if (nargin < 1)
    nmat = length (f) ;
else
    nmat = min (nmat, length (f)) ;
end
nmat = max (nmat, 1) ;
f = f (1:nmat) ;

Control = umfpack ;
Control.prl = 0 ;

clf

h = waitbar (0, 'UMFPACK test') ;

for k = 1:nmat

    i = f (k) ;
    waitbar (k/nmat, h, 'UMFPACK test') ;

%    try

        fprintf ('\nmatrix: %s %s %d\n', ...
            index.Group{i}, index.Name{i}, index.nrows(i)) ;

        Prob = ssget (i) ;
        A = Prob.A ;
        n = size (A,1) ;

        b = rand (1,n) ;
        c = b' ;

	%-----------------------------------------------------------------------
	% symbolic factorization
	%-----------------------------------------------------------------------

	[P1, Q1, Fr, Ch, Info] = umfpack (A, 'symbolic') ;
	subplot (2,2,1)
	spy (A)
	title ('A')

	subplot (2,2,2)
	treeplot (Fr (1:end-1,2)') ;
	title ('supercolumn etree')

	%-----------------------------------------------------------------------
	% P(R\A)Q = LU
	%-----------------------------------------------------------------------

	[L,U,P,Q,R,Info] = umfpack (A, struct ('details',1)) ;
	err = lu_normest (P*(R\A)*Q, L, U) ;
	fprintf ('norm est PR\\AQ-LU: %g relative: %g\n', ...
	    err, err / norm (A,1)) ;

	subplot (2,2,3)
	spy (P*A*Q)
	title ('PAQ') ;

	cs = Info.number_of_column_singletons ; % (57) ;
	rs = Info.number_of_row_singletons ; % (58) ;

	subplot (2,2,4)
	hold off
        try
            spy (L+U)
        catch
        end
	hold on
	if (cs > 0)
	    plot ([0 cs n n 0] + .5, [0 cs cs 0 0]+.5, 'c') ;
	end
	if (rs > 0)
	    plot ([0 rs rs 0 0] + cs +.5, [cs cs+rs n n cs]+.5, 'r') ;
	end

	title ('LU factors')
	drawnow

	%-----------------------------------------------------------------------
	% PAQ = LU
	%-----------------------------------------------------------------------

	[L,U,P,Q] = umfpack (A) ;
	err = lu_normest (P*A*Q, L, U) ;
	fprintf ('norm est PAQ-LU:   %g relative: %g\n', ...
	    err, err / norm (A,1)) ;

	%-----------------------------------------------------------------------
	% solve
	%-----------------------------------------------------------------------

	x1 = b/A ;
	y1 = A\c ;
	m1 = norm (b-x1*A) ;
	m2 = norm (A*y1-c) ;

	% factor the transpose
	Control.irstep = 2 ;
	[x, info] = umfpack (A', '\', c, Control) ;
	lunz0 = info.nnz_in_L_plus_U ;
	r = norm (A'*x-c) ;

	fprintf (':: %8.2e  matlab: %8.2e %8.2e\n',  r, m1, m2) ;

	% factor the original matrix and solve xA=b
	for ir = 0:4
            Control.irstep = ir ;
	    [x, info] = umfpack (b, '/', A, Control) ;
	    r = norm (b-x*A) ;
	    if (ir == 0)
		lunz1 = info.nnz_in_L_plus_U ;
	    end
	    fprintf ('%d: %8.2e %d\n', ir, r, info.iterative_refinement_steps) ;
	end

	% factor the original matrix and solve Ax=b
	for ir = 0:4
            Control.irstep = ir ;
	    [x, info] = umfpack (A, '\', c, Control) ;
	    r = norm (A*x-c) ;
	    fprintf ('%d: %8.2e %d\n', ir, r, info.iterative_refinement_steps) ;
	end

	fprintf (...
	    'lunz trans %12d    no trans: %12d  trans/notrans: %10.4f\n', ...
	    lunz0, lunz1, lunz0 / lunz1) ;

	%-----------------------------------------------------------------------
	% get the determinant
	%-----------------------------------------------------------------------

	det1 = det (A) ;
	det2 = umfpack (A, 'det') ;
	[det3 dexp3] = umfpack (A, 'det') ;
	err = abs (det1-det2) ;
	err3 = abs (det1 - (det3 * 10^dexp3)) ;
	denom = det1 ;
	if (denom == 0)
	    denom = 1 ;
	end
	err = err / denom ;
	err3 = err3 / denom ;
	fprintf ('det:  %20.12e + (%20.12e)i MATLAB\n', ...
	    real(det1), imag(det1)) ;
	fprintf ('det:  %20.12e + (%20.12e)i umfpack\n', ...
	    real(det2), imag(det2)) ;
	fprintf ('det: (%20.12e + (%20.12e)i) * 10^(%g) umfpack\n', ...
	    real(det3), imag(det3), dexp3) ;
	fprintf ('diff %g %g\n', err, err3) ;

%    catch
%        % out-of-memory is OK, other errors are not
%        disp (lasterr) ;
%        if (isempty (strfind (lasterr, 'Out of memory')))
%            error (lasterr) ;                                               %#ok
%        else
%            fprintf ('test terminated early, but otherwise OK\n') ;
%        end
%    end

end

close (h) ;     % close the waitbar
