function sstextwrite (filename, X)
%SSTEXTWRITE write a char array or cell array of strings to a text file
%
% sstextwrite (filename, X)
%
% X must either be a char array of size m-by-n, or a cell array of size m-by-1
% where each component X{i} has size 1-by-*.

ff = fopen (filename, 'w') ;
if (ff < 0)
    error ('cannot open file') ;
end

if (iscell (X))
    if (size (X, 2) ~= 1)
        error ('invalid object written to text file') ;
    end
    for i = 1:size (X,1)
        s = X {i} ;
        if (~ischar (s) || size (s,1) ~= 1)
            error ('invalid object written to text file') ;
        end
	fprintf (ff, '%s\n', deblank (X {i})) ;
    end
else
    if (~ischar (X))
        error ('invalid object written to text file') ;
    end
    for i = 1:size (X,1)
	fprintf (ff, '%s\n', deblank (X (i,:))) ;
    end
end

fclose (ff) ;

