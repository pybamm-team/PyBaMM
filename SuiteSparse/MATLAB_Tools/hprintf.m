function count = hprintf (varargin)
%HPRINTF fprintf with hypertext links not highlighted in the command window.
% hprintf does this by replacing the string "href=" with "HREF=".  If the file
% descriptor is not present, the output defaults to the command window, just
% like fprintf.  Note that all browsers accept either href= or HREF=, so your
% hypertext links will not be affected except within the MATLAB Command Window.
% General usage is identical to fprintf:
%
%      hprintf (format, arg1, arg2, ...)
%      hprintf (fid, format, arg1, arg2, ...)
%      count = hprintf ( ... same as above ... )
%
% Example:
%      fprintf ('<a href="http://www.mathworks.com">MathWorks</a>\n') ;
%      fprintf ('<a href="http://hitchhikers.movies.go.com">%d</a>\n', 42) ;
%      hprintf ('<a href="http://www.mathworks.com">MathWorks</a>\n') ;
%      hprintf ('<a href="http://hitchhikers.movies.go.com">%d</a>\n', 42) ;
%
% For a discussion, see Kristin's blog and the comments there at
% http://blogs.mathworks.com/desktop/2007/07/09
% (<a href="http://blogs.mathworks.com/desktop/2007/07/09">Kristin's blog</a>).
%
% NOTE: the examples above are modified by "help hprintf" so that you cannot
% see the HREF= text.  To see the examples properly (without hypertext
% highlighting) use:
%
%      edit hprintf
%
% To try the examples above, use hprintf with no inputs (note that this usage
% of hprintf also flags an error, to exactly mimic the fprintf behavior):
%
%      hprintf
%
% Here is a slightly more complex example that has the advantage of being
% printed properly by "help hprintf":
%
%      % a string template with hypertext contents:
%      str = '<a AREF="http://www.mathworks.com">MathWorks</a>\n' ;
%      % made into an active hypertext, which will be underlined when
%      % displayed in the command window:
%      hstr = strrep (str, 'AREF', 'href') ;
%      fprintf (hstr) ;
%      %          displays: '<a href="http://www.mathworks.com">MathWorks</a>'
%      hprintf (hstr) ;
%      %          displays: '<a HREF="http://www.mathworks.com">MathWorks</a>'
%
% See also fprintf, strrep, sprintf.

% Copyright 2007, T. Davis, with thanks to 'us' (us at neurol dot unizh dot ch)
% for suggestions.

% Aug 25, 2007

if (nargin < 1)

    % try hprintf
    help hprintf
    fprintf ('\nhypertext highlighting with fprintf:\n\n') ;
    fprintf ('<a href="http://www.mathworks.com">MathWorks</a>\n') ;
    fprintf ('<a href="http://hitchhikers.movies.go.com">%d</a>\n', 42) ;
    fprintf ('\nhypertext highlighting turned off with hprintf:\n\n') ;
    hprintf ('<a href="http://www.mathworks.com">MathWorks</a>\n') ;
    hprintf ('<a href="http://hitchhikers.movies.go.com">%d</a>\n\n', 42) ;

    % flag an error, to mimic fprintf behavior
    error ('Not enough input arguments') ;

elseif (nargout > 1)

    % mimic fprintf
    error ('Too many output arguments') ;

else

    if (ischar (varargin {1}))
        % mimic fprintf ('hello world %d\n', 42), with no file ID
        cnt = fprintf (strrep (sprintf (varargin {:}), 'href=', 'HREF=')) ;
    else
        % mimic fprintf (fid, 'hello world %d\n', 42), with file ID given
        cnt = fprintf (varargin {1}, ...
            strrep (sprintf (varargin {2:end}), 'href=', 'HREF=')) ;
    end
    if (nargout > 0)
        % return the fprintf output
        count = cnt ;
    end
end

