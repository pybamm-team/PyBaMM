function [C, len] = sstextread (filename, as_cell)
%SSTEXTREAD read a text file as a char array or cell array of strings
%
% [C, len] = sstextread (filename, as_cell)
%
% If the longest line is > 1024 characters and as_cell is true, then C is
% returned as a cell array of strings.  Otherwise it is returned as char array.
% If as_cell is not present it defaults to true.  The length of the longest
% line is returned in len.

if (nargin < 2)
    as_cell = true ;
end

% first, determine # of lines and the longest line in the file
f = fopen (filename) ;
if (f < 0)
    error (['cannot open ' filename]) ;
end
len = 0 ;
nline = 0 ;
while (1)
    s = fgetl (f) ;
    if (~ischar (s))
        break 
    end
    % ignore trailing blanks
    s = deblank (s) ;
    len = max (len, length (s)) ;
    nline = nline + 1 ;
end
fclose (f) ;

% reopen the file
f = fopen (filename) ;
if (f < 0)
    error (['cannot open ' filename]) ;
end

if (len > 1024 && as_cell)

    % read the file as a cell array of strings
    fprintf ('%s as cell, len: %d\n', filename, len) ;
    C = cell (nline, 1) ;
    i = 0 ;
    while (1)
        s = fgetl (f) ;
        if (~ischar (s))
            break 
        end
        s = deblank (s) ;
        i = i + 1 ;
        len = length (s) ;
        if (len == 0)
            s = char (zeros (1,0)) ;
        end
        C {i} = s ;
    end

else

    % read in the file as a char array
    fprintf ('%s as char, len: %d\n', filename, len) ;
    C = repmat (' ', nline, len) ;
    i = 0 ;
    while (1)
        s = fgetl (f) ;
        if (~ischar (s))
            break 
        end
        s = deblank (s) ;
        i = i + 1 ;
        len = length (s) ;
        if (len > 0)
            C (i, 1:len) = s ;
        end
    end

end

fclose (f) ;

