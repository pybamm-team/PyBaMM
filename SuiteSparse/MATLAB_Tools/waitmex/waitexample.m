function waitexample
%WAITEXAMPLE a C mexFunction that serves as an example for waitmex.
%
% Example:
%   waitexample      % draw a waitbar, make progress, and then close the waitbar
%   h = waitexample; % same as above, except leave the waitbar on the screen
%                    % and return the handle h to the waitbar.
%
% See also waitbar, waitex.

% Copyright 2007, T. Davis

error ('waitexample mexFunction not found ... it must be compiled first') ;
