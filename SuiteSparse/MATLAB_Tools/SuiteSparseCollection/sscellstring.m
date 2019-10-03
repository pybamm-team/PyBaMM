function s = sscellstring (C)
%SSCELLSTRING what iscellstr should do
%
% Return true if C is a cell array of size *-by-1 where each component is a
% 1-by-* string.  See also iscellstr.

[m n] = size (C) ;

if (n ~= 1)

    % C is not *-by-1
    s = false ;

elseif (~iscellstr (C))

    % C is not a cell array of strings
    s = false ;

else

    % iscellstr(C) reports true, but it allows any component C{i} to be a char
    % array of any size, not just a char row vector.  Now make sure each
    % component C{i} is a char row vector of size 1-by-*, not any array.

    s = true ;
    for i = 1:m
        if (~ischar (C {i}) || size (C {i}, 1) ~= 1)
            s = false ;
            break ;
        end
    end

end

