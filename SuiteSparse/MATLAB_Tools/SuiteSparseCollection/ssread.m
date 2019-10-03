function Problem = ssread (directory, tmp)
%SSREAD read a Problem in Matrix Market or Rutherford/Boeing format
% containing a set of files created by sswrite, in either Matrix Market or
% Rutherford/Boeing format. See sswrite for a description of the Problem struct.
%
% Usage: Problem = ssread (directory)
%
% Example:
%
%   load west0479
%   clear Problem
%   Problem.name = 'HB/west0479' ;
%   Problem.title = '8 STAGE COLUMN SECTION, ALL SECTIONS RIGOROUS (CHEM.ENG.)';
%   Problem.A = west0479 ;
%   Problem.id = 267 ;          % the id number of west0479 in the collection
%   Problem.date = '1983' ;
%   Problem.author = 'A. Westerberg' ;
%   Problem.ed = 'I. Duff, R. Grimes, J. Lewis'
%   Problem.kind = 'chemical process simulation problem' ;
%   sswrite (Problem, 'RB/', '') ;
%   Prob3 = ssread ('RB/HB/west0479')
%   isequal (Problem, Prob3)
%
% This part of the example requires CHOLMOD, for the mread function:
%
%   sswrite (Problem, 'MM/') ;
%   Prob2 = ssread ('MM/HB/west0479')
%   isequal (Problem, Prob2)
%
% You can also compare this Problem with the version in the SuiteSparse Matrix
% Collection, with ssget(267) or ssget('HB/west0479').  Note that this includes
% the 22 explicit zero entries present in the west0479 Harwell/Boeing matrix,
% but not included in the MATLAB west0479.mat demo matrix.  Those entries are
% present in the SuiteSparse Matrix Collection.  This example assumes your current
% directory is the RBio directory, containing the west0479 problem in the
% RBio/Test directory:
%
%   Prob5 = ssget ('HB/west0479')
%   Prob6 = ssread ('Test/west0479')
%   isequal (Prob5, Prob6)
%
% The directory can be a compressed tar file of the form "name.tar.gz", in
% which case the tarfile is uncompressed into a temporary directory, and
% the temporary directory is deleted when done.  The '.tar.gz' should not be
% part of the directory argument.  In this case, a 2nd input argument can be
% provided:  Problem = ssread (directory, tmp).  The problem is extracted into
% the tmp directory.  If tmp is not present, the output of the tempdir function
% is used instead.
%
% Note that ssget is much faster than ssread.  ssread is useful if you are
% short on disk space, and want to have just one copy of the collection that
% can be read by MATLAB (via ssread) and a non-MATLAB program (the MM or RB
% versions of the collection).
%
% See also sswrite, mread, mwrite, RBread, RBread, ssget, untar, tempdir.

% Optionally uses the CHOLMOD mread mexFunction, for reading Problems in
% Matrix Market format.

% Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% determine the Problem name from the directory name
%-------------------------------------------------------------------------------

directory = regexprep (directory, '[\/\\]', '/') ;
t = find (directory == '/') ;
if (isempty (t))
    name = directory ;
else
    name = directory (t(end)+1:end) ;
end

%-------------------------------------------------------------------------------
% open the directory, or untar the tar.gz file
%-------------------------------------------------------------------------------

d = dir (directory) ;
is_tar = 0 ;

if (isempty (d))
    % look for a .tar.gz file
    if (nargin < 2)
	tmpdir = [tempname '_ssread_' name] ;
    else
	tmpdir = [tmp '/' name] ;
    end
    try
	% try untaring the problem
	untar ([directory '.tar.gz'], tmpdir) ;
    catch
	% untar failed, make sure tmpdir is deleted
	try
	    rmdir (tmpdir, 's') ;
	catch
	end
	error (['unable to read problem: ' directory]) ;
    end
    directory = [tmpdir '/' name] ;
    d = dir (directory) ;
    is_tar = 1 ;
end 

%-------------------------------------------------------------------------------
% read the problem
%-------------------------------------------------------------------------------

try

    %---------------------------------------------------------------------------
    % get name, title, id, kind, date, author, editor, notes from master file
    %---------------------------------------------------------------------------

    masterfile = [directory '/' name] ;
    [Problem notes RB] = get_header (masterfile) ;

    %---------------------------------------------------------------------------
    % get the A and Zero matrices from the master file and add to the Problem
    %---------------------------------------------------------------------------

    if (RB)
	% read in the primary Rutherford/Boeing file
	[Problem.A Zeros] = RBread ([masterfile '.rb']) ;
    else
	% read in the primary Matrix Market file.  Get patterns as binary.
	[Problem.A Zeros] = mread ([masterfile '.mtx'], 1) ;
    end
    if (nnz (Zeros) > 0)
	Problem.Zeros = Zeros ;
    end

    % add the notes after A and Zeros
    if (~isempty (notes))
	Problem.notes = notes ;
    end

    namelen = length (name) ;

    %---------------------------------------------------------------------------
    % read b, x, aux (incl. any aux.cell sequences), stored as separate files
    %---------------------------------------------------------------------------

    for k = 1:length(d)

	% get the next filename in the directory
	file = d(k).name ;
	fullfilename = [directory '/' file] ;

	if (length (file) < length (name) + 1)

	    % unrecognized file; skip it
	    continue

	elseif (strcmp (file, [name '.mtx']))

	    % skip the master file; already read in
	    continue

	elseif (strcmp (file, [name '_b.mtx']))

	    % read in b as a Matrix Market file
	    Problem.b = mtx_read (fullfilename, RB) ;

	elseif (strcmp (file, [name '_x.mtx']))

	    % read in x as a Matrix Market file
	    Problem.x = mtx_read (fullfilename, RB) ;

	elseif (strcmp (file, [name '_b.rb']))

	    % read in b as a Rutherford/Boeing file
	    Problem.b = RBread (fullfilename) ;

	elseif (strcmp (file, [name '_x.rb']))

	    % read in x as a Rutherford/Boeing file
	    Problem.x = RBread (fullfilename) ;

	elseif (strcmp (file (1:length(name)+1), [name '_']))

	    % read in an aux component, in the form name_whatever.mtx
	    thedot = find (file == '.', 1, 'last') ;
	    ext = file (thedot:end) ;

	    if (strcmp (ext, '.txt'))

                % get a txt file as either a char array or cell array of strings
                C = sstextread (fullfilename, Problem.id > 2776) ;

	    elseif (strcmp (ext, '.mtx'))

		% read a full or sparse auxiliary matrix in the Matrix Market
		% form, or a full auxiliary matrix in the Rutherford/Boeing form.
		C = mtx_read (fullfilename, RB) ;

	    elseif (strcmp (ext, '.rb'))

		% read in a sparse matrix, for a Rutherford/Boeing collection
		C = RBread (fullfilename) ;

	    else

		% this file is not recognized - skip it.
		C = [ ] ;

	    end

	    % determine the name of the component and place it in the Problem
	    if (~isempty (C))
		% Determine if this is part of an aux.whatever cell sequence.
		% These filenames have the form name_whatever_#.mtx, where name
		% is the name of the Problem, and # is a number (1 or more
		% digts) greater than zero.  If # = i, this becomes the
		% aux.whatever{i} matrix.
		suffix = file (namelen+2:thedot-1) ;
		t = find (suffix == '_', 1, 'last') ;
		what = suffix (1:t-1) ;
		i = str2num (suffix (t+1:end)) ;			    %#ok
		if (~isempty (i) && i > 0 && ~isempty (what))
		    % this is part of aux.whatever{i} cell array
		    Problem.aux.(what) {i,1} = C ;
		elseif (~isempty (suffix))
		    % this is not a cell, simply an aux.whatever matrix
		    Problem.aux.(suffix) = C ;
		end
	    end
	end
    end

    %---------------------------------------------------------------------------
    % delete the uncompressed version of the tar file
    %---------------------------------------------------------------------------

    if (is_tar)
	rmdir (tmpdir, 's') ;
    end

catch

    %---------------------------------------------------------------------------
    % catch the error, delete the temp directory, and rethrow the error
    %---------------------------------------------------------------------------

    try
	if (is_tar)
	    rmdir (tmpdir, 's') ;
	end
    catch
    end
    rethrow (lasterror) ;

end


%-------------------------------------------------------------------------------
% get_header: get the header of the master file (Group/name/name.txt or .mtx)
%-------------------------------------------------------------------------------

function [Problem, notes, RB] = get_header (masterfile)
% Get the name, title, id, kind, date, author, editor and notes from the master
% file.  The name, title, and id are required.  They appear as structured
% comments in the Matrix Market file (masterfile.mtx) or in the text file for
% a problem in Rutherford/Boeing format (masterfile.txt).  RB is returned as
% 1 if the problem is in Rutherford/Boeing format, 0 otherwise.

% first assume it's in Matrix Market format
f = fopen ([masterfile '.mtx'], 'r') ;
if (f < 0)
    % oops, that failed.  This must be a problem in Rutherford/Boeing format
    RB = 1 ;
    f = fopen ([masterfile '.txt'], 'r') ;
    if (f < 0)
	% oops again, this is not a valid problem in the SuiteSparse collection
	error (['invalid problem: ' masterfile]) ;
    end
else
    % we found the Matrix Market file
    RB = 0 ;
end
Problem = [ ] ;
notes = [ ] ;

while (1)

    % get the next line
    s = fgetl (f) ;
    if (~ischar (s) || length (s) < 3 || s (1) ~= '%')
	% end of file or end of leading comments ... no notes found
	fclose (f) ;
	[Problem notes] = valid_problem (Problem, [ ]) ;
	return ;
    end

    % remove the leading '% ' and get the first token
    s = s (3:end) ;
    [t r] = strtok (s) ;

    % parse the line
    if (strcmp (t, 'name:'))

	% get the Problem.name.  It must be of the form Group/Name.
	Problem.name = strtrim (r) ;
	if (length (find (Problem.name == '/')) ~= 1)
	    fclose (f) ;
	    error (['invalid problem name ' Problem.name]) ;
	end

    elseif (s (1) == '[')

	% get the Problem.title
	k = find (s == ']', 1, 'last') ;
	if (isempty (k))
	    fclose (f) ;
	    error ('invalid problem title') ;
	end
	Problem.title = s (2:k-1) ;

    elseif (strcmp (t, 'id:'))

	% get the Problem.id
	Problem.id = str2num (r) ;					    %#ok
	if (isempty (Problem.id) || Problem.id < 0)
	    fclose (f) ;
	    error ('invalid problem id') ;
	end

    elseif (strcmp (t, 'kind:'))

	% get the Problem.kind
	Problem.kind = strtrim (r) ;

    elseif (strcmp (t, 'date:'))

	% get the Problem.date
	Problem.date = strtrim (r) ;

    elseif (strcmp (t, 'author:'))

	% get the Problem.author
	Problem.author = strtrim (r) ;

    elseif (strcmp (t, 'ed:'))

	% get the Problem.ed
	Problem.ed = strtrim (r) ;

    elseif (strcmp (t, 'notes:'))

	% get the notes, which always appear last
	k = 0 ;
	notes = [ ] ;
	while (1)
	    % get the next line
	    s = fgetl (f) ;
	    if (~ischar (s) || length (s) < 2 || ~strcmp (s (1:2), '% '))
		% end of file or end of notes ... convert notes to char array
		fclose (f) ;
		[Problem notes] = valid_problem (Problem, notes) ;
		return ;
	    end
	    % add the line to the notes
	    k = k + 1 ;
	    notes {k} = s ;						    %#ok
	end
    end
end


%-------------------------------------------------------------------------------
% valid_problem: determine if a problem is valid, and finalizes the notes
%-------------------------------------------------------------------------------

function [Problem, notes] = valid_problem (Problem, notes)
% make sure the required fields (name, title, id, date, author, ed) are present.
% Convert notes to char, and strip off the leading '% ', inserted when the notes
% were printed in the Matrix Market file.
if (~isfield (Problem, 'name') || ~isfield (Problem, 'title') || ...
    ~isfield (Problem, 'id') || ~isfield (Problem, 'date') || ...
    ~isfield (Problem, 'author') || ~isfield (Problem, 'ed') || ...
    ~isfield (Problem, 'kind'))
    error ('invalid Problem mfile') ;
end
if (~isempty (notes))
    notes = char (notes) ;
    notes = notes (:, 3:end) ;
end


%-------------------------------------------------------------------------------
% mtx_read: read a *.mtx file
%-------------------------------------------------------------------------------

% In the Rutherford/Boeing form, a *.mtx file is used only for full matrices,
% using a tiny subset of the Matrix Market format.  In the Matrix Market form,
% the *.mtx is used for all b, x, and aux matrices (both full and sparse).

function C = mtx_read (file, RB)

if (~RB)

    % Get a Matrix Market file, using full Matrix Market features.
    C = mread (file, 1) ;

else

    % mread is not installed.  The RB format uses a tiny subset of the Matrix
    % Market format for full matrices: just the one header line, and no comment
    % or blank lines permitted.  Allowable header lines are:
    %	%%MatrixMarket matrix array real general
    %	%%MatrixMarket matrix array complex general
    % This tiny subset can be read by ssfull_read.
    C = ssfull_read (file) ;

end

