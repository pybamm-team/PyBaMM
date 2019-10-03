load sstest2_results.mat % TM T1 TU f
whos

index = ssget ;

tmax = max (max (TM))
tmin = min (TM (find (TM > 0)))

nmat = length (f) ;
k = nmat ;

    for kind = 1:4

	subplot (2,4,kind) ;
	r = TM (1:k,kind) ./ T1 (1:k,kind) ;
	rmin = min (r) ;
	rmax = max (r) ;
	loglog (TM (1:k,kind), r, 'o', ...
	    [tmin tmax], [1 1], 'r-', ...
	    [tmin tmax], [1.1 1.1], 'r-', ...
	    [tmin tmax], [1/1.1 1/1.1], 'r-', ...
	    [tmin tmax], [2 2], 'g-', ...
	    [tmin tmax], [1.5 1.5], 'g-', ...
	    [tmin tmax], [1/1.5 1/1.5], 'g-', ...
	    [tmin tmax], [.5 .5], 'g-' );
	if (k > 2)
	    axis ([tmin tmax rmin rmax]) ;
	    set (gca, 'XTick', [1e-5 1e-4  1e-3 1e-2 1e-1 1 10]) ;
	    set (gca, 'YTick', [.5 1/1.5 1/1.1 1 1.1 1.5 2]) ;
	end
	xlabel ('MATLAB time') ; 
	ylabel ('MATLAB/SM time') ; 
	if (kind == 1)
	    title ('real*real') ;
	elseif (kind == 2)
	    title ('complex*real') ;
	elseif (kind == 3)
	    title ('real*complex') ;
	elseif (kind == 4)
	    title ('complex*complex') ;
	end

    end
