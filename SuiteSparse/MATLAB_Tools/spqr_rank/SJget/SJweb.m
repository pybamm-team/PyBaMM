function stats = SJweb (matrix, opts)
%SJWEB opens the URL for a matrix.
%
%   SJweb(matrix) opens the URL for a matrix.  This parameter can be a string,
%   or an integer.  If it is a string with no "/" character, the web page for a
%   matrix group is displayed.  With no arguments, a list of all the matrix
%   groups is displayed.
%
%   Example:
%
%   If Problem = SJget ('HB/ash292'), the first four examples display
%   the same thing, the web page for the HB/ash292 matrix:
%
%       SJweb (6)
%       SJweb ('HB/ash292')
%       stats = SJweb (6)
%
%   The latter also returns statistics about the matrix or matrix group.
%   To display the web page for the HB (Harwell-Boeing) group:
%
%       SJweb ('HB')
%
%   To display the home page for the SJSU Singular matrix collection:
%
%       SJweb
%       SJweb (0)
%       SJweb ('')
%
%   The latter two are useful if a second optional parameter is specified.
%   The second optional argument is a string passed as additional parameters to
%   the MATLAB web command.  To use the system web browser instead of the MATLAB
%   browser, for example, use SJweb ('HB/ash292', '-browser').
%
%   See also web, SJget, SJget_defaults.

%   Derived from the ssget toolbox on March 18, 2008.
%   Copyright 2007, Tim Davis, University of Florida.

params = SJget_defaults ;
SJ_Index = SJget ;

if (nargin < 1)
    matrix = '' ;
end
if (nargin < 2)
    opts = '' ;
end

% get the matrix group, name, and id
[group name id] = SJget_lookup (matrix, SJ_Index) ;

url = params.url ;
len = length (url) ;
if (strcmp (url ((len-3):len), '/mat'))
    % remove the trailing '/mat'
    url = url (1:(len-4)) ;
end

% open the web page for the matrix, group, or whole collection
if (id == 0)
    if (isempty (group))
	eval (['web ' url '/index.html ' opts])
    else
	eval (['web ' url '/html/' group '/index.html ' opts])
    end
else
    eval (['web ' url '/html/' group '/' name '.html ' opts])
end

% return stats
if (nargout > 0)

    if (id == 0)

	if (isempty (group))

	    % return stats about the whole collection
	    stats.nmatrices = length (SJ_Index.nrows) ;
	    stats.LastRevisionDate = SJ_Index.LastRevisionDate ;
	    stats.DownloadTime = datestr (SJ_Index.DownloadTimeStamp) ;

	else

	    % return stats about one matrix group
	    nmat = length (SJ_Index.nrows) ;
	    ngroup = 0 ;
	    for i = 1:nmat
		if (strcmp (group, SJ_Index.Group {i}))
		    ngroup = ngroup + 1 ;
		end
	    end
	    stats.nmatrices = ngroup ;
	    stats.LastRevisionDate = SJ_Index.LastRevisionDate ;
	    stats.DownloadTime = datestr (SJ_Index.DownloadTimeStamp) ;

	end
    else

	% look up the matrix statistics
	stats.Group = group ;
	stats.Name = name ;
	stats.nrows = SJ_Index.nrows (id) ;
	stats.ncols = SJ_Index.ncols (id) ;
	stats.nnz = SJ_Index.nnz (id) ;
	stats.nzero = SJ_Index.nzero (id) ;
	stats.pattern_symmetry = SJ_Index.pattern_symmetry (id) ;
	stats.numerical_symmetry = SJ_Index.numerical_symmetry (id) ;
	stats.isBinary = SJ_Index.isBinary (id) ;
	stats.isReal = SJ_Index.isReal (id) ;

	stats.nnzdiag = SJ_Index.nnzdiag (id) ;
	stats.posdef = SJ_Index.posdef (id) ;

	stats.amd_lnz = SJ_Index.amd_lnz (id) ;
	stats.amd_flops = SJ_Index.amd_flops (id) ;
	stats.amd_vnz = SJ_Index.amd_vnz (id) ;
	stats.amd_rnz = SJ_Index.amd_rnz (id) ;
	stats.metis_lnz = SJ_Index.metis_lnz (id) ;
	stats.metis_flops = SJ_Index.metis_flops (id) ;
	stats.metis_vnz = SJ_Index.metis_vnz (id) ;
	stats.metis_rnz = SJ_Index.metis_rnz (id) ;
	stats.nblocks = SJ_Index.nblocks (id) ;
	stats.sprank = SJ_Index.sprank (id) ;
	stats.nzoff = SJ_Index.nzoff (id) ;
	stats.dmperm_lnz = SJ_Index.dmperm_lnz (id) ;
	stats.dmperm_unz = SJ_Index.dmperm_unz (id) ;
	stats.dmperm_flops = SJ_Index.dmperm_flops (id) ;
	stats.dmperm_vnz = SJ_Index.dmperm_vnz (id) ;
	stats.dmperm_rnz = SJ_Index.dmperm_rnz (id) ;

	stats.RBtype = SJ_Index.RBtype (id,:) ;
	stats.cholcand = SJ_Index.cholcand (id) ;
	stats.ncc = SJ_Index.ncc (id) ;

    end
end
