function files_out = gipper (directory, include, exclude, exclude_hidden)
%GIPPER zip selected files and subdirectories (gipper = grep + zip)
%
%   files = gipper (directory, include, exclude, exclude_hidden) ;
%
% Creates a zip file of all files and subdirectories in a directory.  A file in
% the directory or any of its subdirectories whose name matches any expression
% in 'include' via regexp is added to the zip file.  A file that matches any
% expression in 'exclude' is not added.  A subdirectory whose name or full
% pathname matches any expression in 'exclude' is not searched.  The name of
% the zip file is the name of the directory, with '.zip' appended.
%
% 'include' and 'exclude' are either cells of strings, or just single strings.
%
% With no outputs, a list of files is printed and the user is prompted before
% proceeding.  Otherwise, the gipper proceeds without prompting and returns a
% list of files that were added to the zip file.
%
% By default, all files and subdirectories of the directory are included, except
% that hidden files and directories (those whose names start with a dot, '.')
% are excluded.
%
% If any parameter is empty or not present, the defaults are used:
%   directory: defaults to the current directory
%   include: defaults to include all files and directories
%   exclude: defaults to exclude nothing, as modified by 'exclude_hidden'
%   exclude_hidden: 1 (exclude hidden files and directories)
%
% Empty directories or subdirectories are never included.
%
% Example:
%     % suppose 'X' is the name of the current directory.
%
%     % include all files in X (except hidden files) in the zip file ../X.zip
%     gipper
%
%     % create mytoolbox.zip archive of the 'X/mytoolbox' directory
%     gipper mytoolbox
%
%     % only include *.m files in ../X.zip
%     gipper '' '\.m$'
%
%     % create ../X.zip, but exclude compiled object and MEX files
%     gipper ('', '', { '\.o$' '\.obj$', ['\.' mexext '$'] })
%
%     % include everything, including hidden files, in ../X.zip
%     gipper ('', '', '', 0)
%
%     % zip mytoolbox, except hidden files and the mytoolbox/old directory
%     gipper mytoolbox '' old
%
%     % these are the same, except gipper also traverses subdirectories
%     gipper ('', { '\.m$', '\.*mat$' })
%     zip ('../X', { '*.m', '*.mat' })
%
% See also zip, regexp, unzip.

% NOTE: if the directory name is empty or not present, and you hit control-C
% while the gipper is running, your current directory will now be the parent.
% You must install the gipper first, by placing it in your MATLAB path.

% Copyright 2007, Timothy A. Davis, Win one for the gipper.
% Created May 2007, using MATLAB 7.4 (R2007a).  Requires MATLAB 6.5 or later.

% exclude hidden files and directories by default
if (nargin < 4)
    exclude_hidden = 1 ;
end

% exclude nothing by default (as modified by exclude_hidden)
if (nargin < 3)
    exclude = { } ;
end
exclude = cleanup (exclude) ;

% append the hidden file and directory rule, if requested
if (exclude_hidden)
    exclude = union (exclude, { '^\.', [ '\' '/' '\.' ] }) ;
end

% always exclude '.' and '..' files
exclude = union (exclude, { '^\.$', '^\.\.$' }) ;

% include all files by default
if (nargin < 2 || isempty (include))
    include = { '.' } ;
end
include = cleanup (include) ;

% operate on the current directory, if not specified
if (nargin < 1 || isempty (directory))
    here = pwd ;
    directory = here ((find (here == '/', 1, 'last') + 1) : end) ;
    % use try-catch so that if a failure occurs, we go back to current
    % directory.  Unfortunately, this mechanism does not catch a control-C.
    gipper_found = 0 ;
    try
	% run the gipper in the parent
	cd ('..') ;
	% if gipper.m is not in the path, it will no longer exist
	gipper_found = ~isempty (which ('gipper')) ;
	if (gipper_found)
	    if (nargout == 0)
		fprintf ('Note that if you terminate gipper with control-C, ') ;
		fprintf ('your\ndirectory be changed to the parent') ;
		fprintf (' (as in "cd ..").\n') ;
		gipper (directory, include, exclude, exclude_hidden) ;
	    else
		files_out = gipper (directory, include, exclude,exclude_hidden);
	    end
	end
    catch
	cd (here) ;
	rethrow (lasterror) ;
    end
    % go back to where we started
    cd (here) ;
    if (~gipper_found)
	fprintf ('To install the gipper, type "pathtool" and add\n') ;
	fprintf ('the directory in which it resides:\n') ;
	fprintf ('%s\n', which (mfilename)) ;
	error ('You must install the gipper first.') ;
    end
    return
else
    if (nargout == 0)
	fprintf ('\ngipper: creating %s%s%s.zip\n', pwd, '/', directory) ;
    end
end

% get the list of files to zip
n = 0 ;
files = { } ;
for file = dir (directory)'
    [files, n] = finder (files, n, directory, file.name, include, exclude) ;
end
files = files (1:n)' ;

% cannot create an empty zip file
if (isempty (files))
    warning ('gipper:nothing', 'nothing to zip; no zip file created') ;
    if (nargout > 0)
	files_out = files ;
    end
    return
end

% return the list of files, or confirm
if (nargout == 0)
    % print the list of files and ask for confirmation first
    fprintf ('Creating a zip archive containing these files:\n\n') ;
    for k = 1:length(files)
	fprintf ('    %s\n', files {k}) ;
    end
    fprintf ('\nCreating the zip archive: %s', directory) ;
    if (isempty (regexp (directory, '\.zip$', 'once')))
	fprintf ('.zip') ;
    end
    fprintf ('\n') ;
    reply = input ('Proceed? (yes or no, default is yes): ', 's') ;
    if (~isempty (reply) && lower (reply (1)) == 'n')
	fprintf ('zip file not created\n') ;
	return
    end
else
    % zip the files without asking
    files_out = files ;
end

% zip the files
zip (directory, files) ;


%-------------------------------------------------------------------------------
function [files, n] = finder (files, n, prefix, name, include, exclude)
% finder: return a list of files to zip
% fullname includes the entire path to the file or directory
fullname = [prefix '/' name] ;
if (isdir (fullname))
    % always traverse a subdirectory to look for files to include, unless the
    % directory name or fullname itself is explicitly excluded.
    if (~(grep (name, exclude) || grep (fullname, exclude)))
	% the directory is selected, recursively traverse it
	for file = dir (fullname)'
	    [files, n] = finder (files, n, fullname, file.name, ...
		include, exclude) ;
	end
    end
else
    % this is a file, apply the include/exclude rules to just the file name
    % itself not the fullname.
    if (grep (name, include) && ~grep (name, exclude))
	% the file is selected for the archive.  Use a dynamic-table approach
	% to speed up the dynamic growth of the table.
	n = n + 1 ;
	files {n} = fullname ;
	if (n == length (files))
	    files {2*n} = [ ] ;
	end
    end
end


%-------------------------------------------------------------------------------
function match = grep (string, list)
% grep: determine if a string matches an expression in a list
match = 0 ;
for expression = list
    if (~isempty (regexp (string, expression {1}, 'once')))
	match = 1 ;
	return ;
    end
end


%-------------------------------------------------------------------------------
function s = cleanup (s)
% cleanup: ensure the input list is in the proper format
s = s (:)' ;	    % make sure it is a row vector
if (ischar (s))
    s = { s } ;	    % if it is a string, convert it into a cell with one string
end

