function result = waitex
%WAITEX same as the waitexample mexFunction, just in M instead of C.
% The only purpose of this function is to serve as a precise description of
% what the waitexample mexFunction does.
%
% Example:
%   waitex          % draw a waitbar, make progress, and then close the waitbar
%   h = waitex ;    % same as above, except leave the waitbar on the screen
%                   % and return the handle h to the waitbar.
%
% See also waitbar, waitexample.

% Copyright 2007, T. Davis

x = 0 ;
h = waitbar (0, 'Please wait...') ;
for i = 0:100
    if (i == 50)
        waitbar (i/100, h, 'over half way there') ;
    else
        waitbar (i/100, h) ;
    end
    % do some useless work
    for j = 0:1e5
        x = useless (x) ;
    end
end

if (nargout > 0)
    % h is return to the caller, leave the waitbar on the screen
    result = h ;
else
    % close the waitbar, and do not return the handle h
    close (h) ;
end

function x = useless (x)
%USELESS do some useless work (x = useless (x) just increments x)
x = x + 1 ;
