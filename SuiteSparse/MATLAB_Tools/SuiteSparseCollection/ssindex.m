function ss_index = ssindex (matrixlist)
%SSINDEX create the index for the SuiteSparse Matrix Collection
%
% ss_index = ssindex (matrixlist)
%
% matrixlist: an integer list, in the range of 1 to the length of the
%   ss_listing.txt file, containing a list of matrices for which to modify
%   the ss_index entries.  If matrixlist is not present, then the ss_index
%   is created from scratch.
%
% ss_index:  a struct containing the index information, with the
%   following fields, assuming that there are n matrices in the collection:
%
%   LastRevisionDate: a string with the date and time the index was updated.
%   DowloadTimeStamp: date and time the index was last downloaded.
%
%   Group: an n-by-1 cell array.  Group {i} is the group for matrix i.
%   Name: n-by-1 cell array.  Name {i} is the name of matrix i.
%
%   The following fields are n-by-1 vectors unless otherwise specified.
%   nrows(id) gives the number of rows of the matrix with id = Problem.id,
%   for example.
%
%   nrows           number of rows
%   ncols           number of columns
%   nnz             number of entries in A
%   RBtype          Rutherford/Boeing type, an n-by-3 char array
%   isBinary        1 if binary, 0 otherwise
%   isReal          1 if real, 0 if complex
%   cholcand        1 if a candidate for sparse Cholesky, 0 otherwise
%   numerical_symmetry  numeric symmetry (0 to 1, where 1=symmetric)
%   pattern_symmetry    pattern symmetry (0 to 1, where 1=symmetric)
%   nnzdiag         nnz (diag (A)) if A is square, 0 otherwise
%   nzero           nnz (Problem.Zeros)
%   amd_lnz         nnz(L) for chol(C(p,p)) where, C=A+A', p=amd(C)
%   amd_flops       flop count for chol(C(p,p)) where, C=A+A', p=amd(C)
%   amd_vnz         nnz in Householder vectors for qr(A(:,colamd(A)))
%   amd_rnz         nnz in R for qr(A(:,colamd(A)))
%   nblocks         # of blocks from dmperm
%   sprank          sprank(A)
%   ncc             # of strongly connected components
%   posdef          1 if positive definite, 0 otherwise
%   isND            1 if a 2D/3D problem, 0 otherwise
%   isGraph         1 if a graph, 0 otherwise
%   lowerbandwidth      lower bandwidth, [i j]=find(A), max(0,max(i-j))
%   upperbandwidth      upper bandwidth, [i j]=find(A), max(0,max(j-i))
%   rcm_lowerbandwidth  lower bandwidth after symrcm
%   rcm_upperbandwidth  upper bandwidth after symrcm
%   xmin            smallest nonzero value
%   xmax            largest nonzero value
%
% If the statistic is intentionally not computed, it is set to -2.  Some
% statistics are not computed for rectangular or structurally singular
% matrices, for example.  If an attempt to compute the statistic was made, but
% failed, it is set to -1.  If no attempt yet has been made to compute the
% entry, it is set to -3.
%
% Example:
%   ssindex
%   ssindex (267:300)
%
% If updating the ss_index.mat file, the function first loads it from its
% default location, via ssget.  This function then saves the new ss_index into
% the ss_index.mat file in the current working directory (not overwriting the
% old copy, unless it resides in the current working directory).  It creates
% the ssstats.csv file used by ssgui.java and sskinds.m and places it in the
% current working directory.
%
% See also ssstats.

% Copyright 2006-2018, Timothy A. Davis

% Requires the SuiteSparse set of packages: CHOLMOD, RBio, CSparse

%   10/13/2001: Created by Erich Mirabal

%-------------------------------------------------------------------------------
% initialize an empty index
%-------------------------------------------------------------------------------

% load the filenames
topdir = sslocation ;
files = textread ([topdir 'files/ss_listing.txt'], '%s') ;

% if no input, assume we have to do the whole file list
create_new = 0 ;
if (nargin < 1)
    matrixlist = 1:length(files) ;
    create_new = 1 ;
elseif (~isempty (matrixlist))
    % validate the input : range is limited by the files variable
    if (min (matrixlist) < 1) || (max (matrixlist) > length (files))
        error ('%s: %s', mfilename, 'Invalid input parameter.') ;
    end
end

if (~create_new)
    % load the index from file
    fprintf ('Loading existing ss_index.mat file\n') ;
    try
        load ss_index
        fprintf ('loaded ss_index in current directory:\n%s\n', pwd) ;
        dir
    catch
        fprintf ('loading ss_index = ssget\n') ;
        ss_index = ssget ;
    end
end

% get the current sskinds
if (create_new)
    kinds = cell (length (files),1) ;
else
    try
        kinds = sskinds ;
    catch
        kinds = cell (2,1) ;
    end
    for i = matrixlist
        kinds {i} = '' ;
    end
end

% revision tracking device
ss_index.LastRevisionDate = datestr (now) ;

% the index structure needs a download date for version tracking
ss_index.DownloadTimeStamp = now ;

% start the index from scratch
if (create_new)

    fprintf ('Creating new ss_index.mat file\n') ;
    nothing = -3 * ones (1, length (files)) ;

    ss_index.Group = cell (size (files)) ;
    ss_index.Name = cell (size (files)) ;       
    ss_index.nrows = nothing ;
    ss_index.ncols = nothing ;
    ss_index.nnz = nothing ;
    ss_index.nzero = nothing ;
    ss_index.pattern_symmetry = nothing ;
    ss_index.numerical_symmetry = nothing ;
    ss_index.isBinary = nothing ;
    ss_index.isReal = nothing ;
    ss_index.nnzdiag = nothing ;
    ss_index.posdef = nothing ;
    ss_index.amd_lnz   = nothing ;
    ss_index.amd_flops = nothing ;
    ss_index.amd_vnz   = nothing ;
    ss_index.amd_rnz   = nothing ;
    ss_index.nblocks   = nothing ;
    ss_index.sprank    = nothing ;
    ss_index.RBtype = char (' '*ones (length (files),3)) ;
    ss_index.cholcand = nothing ;
    ss_index.ncc = nothing ;
    ss_index.isND = nothing ;
    ss_index.isGraph = nothing ;
    ss_index.lowerbandwidth = nothing ;
    ss_index.upperbandwidth = nothing ;
    ss_index.rcm_lowerbandwidth = nothing ;
    ss_index.rcm_upperbandwidth = nothing ;
    ss_index.xmin = nothing ;
    ss_index.xmax = nothing ;

else

    % make sure we have the right length for the arrays
    if length (ss_index.nrows) < max (matrixlist)

        len = max (matrixlist) - length (ss_index.nrows) ;
        nothing = -ones (1, len) ;

        if (len > 0)
            for i = matrixlist
                ss_index.Group {i} = '' ;
                ss_index.Name {i} = '' ;
            end
            ss_index.nrows      = [ss_index.nrows nothing] ;
            ss_index.ncols      = [ss_index.ncols nothing] ;
            ss_index.nnz        = [ss_index.nnz nothing] ;
            ss_index.nzero      = [ss_index.nzero nothing] ;
            ss_index.pattern_symmetry = [ss_index.pattern_symmetry nothing] ;
            ss_index.numerical_symmetry = [ss_index.numerical_symmetry nothing];
            ss_index.isBinary   = [ss_index.isBinary nothing] ;
            ss_index.isReal     = [ss_index.isReal nothing] ;
            ss_index.nnzdiag    = [ss_index.nnzdiag nothing] ;
            ss_index.posdef     = [ss_index.posdef nothing] ;
            ss_index.amd_lnz    = [ss_index.amd_lnz nothing] ;
            ss_index.amd_flops  = [ss_index.amd_flops nothing] ;
            ss_index.amd_vnz    = [ss_index.amd_vnz nothing] ;
            ss_index.amd_rnz    = [ss_index.amd_rnz nothing] ;
            ss_index.nblocks    = [ss_index.nblocks nothing] ;
            ss_index.sprank     = [ss_index.sprank nothing] ;
            ss_index.RBtype     = [ss_index.RBtype ; char (' '*ones (len,3))] ;
            ss_index.cholcand   = [ss_index.cholcand nothing] ;
            ss_index.ncc        = [ss_index.ncc nothing] ;
            ss_index.isND       = [ss_index.isND nothing] ;
            ss_index.isGraph    = [ss_index.isGraph nothing] ;
            ss_index.lowerbandwidth     = [ss_index.lowerbandwidth nothing] ;
            ss_index.upperbandwidth     = [ss_index.upperbandwidth nothing] ;
            ss_index.rcm_lowerbandwidth = [ss_index.rcm_upperbandwidth nothing];
            ss_index.rcm_upperbandwidth = [ss_index.rcm_upperbandwidth nothing];
            ss_index.xmin = [ss_index.xmin nothing] ;
            ss_index.xmax = [ss_index.xmax nothing] ;
        end
    end
end

if (length (matrixlist) > 0)
    fprintf ('Will process %d files\n', length (matrixlist)) ;
end

nmat = length (ss_index.nrows) ;
filesize = zeros (nmat,1) ;

%-------------------------------------------------------------------------------
% look through the directory listing
%-------------------------------------------------------------------------------

for i = matrixlist

    % note that the matrix is not loaded in this for loop
    ffile = deblank (files {i}) ;

    % group is the first part of the string up to the character before
    % the last file separator
    gi = find (ffile == '/') ;
    gi = gi (end) ;
    groupN = char (ffile (1:gi-1)) ;

    % name is the last section of the string after the last file separator
    matrixN = char (ffile (gi+1:end)) ;

    % get the directory info of the .mat file
    fileInfo = dir ([topdir 'mat/' ffile '.mat']) ;

    % set the file's data into the data arrays
    ss_index.Name {i} = matrixN ;
    ss_index.Group {i} = groupN ;

    if (length (fileInfo) > 0)                                              %#ok
        filesize (i) = fileInfo.bytes ;
    else
        filesize (i) = -1 ;
    end

end

if (length (matrixlist) > 0)
    fprintf ('\n======================================================\n') ;
    fprintf ('Matrices will processed in the following order:\n') ;
    for i = matrixlist
        ffile = deblank (files {i}) ;
        fprintf ('Matrix %d: %s filesize %d\n', i, ffile, filesize (i)) ;
        if (filesize (i) == -1)
            fprintf ('skip this file (not found)\n') ;
            continue ;
        end
    end
end

fprintf ('Hit enter to continue\n') ;
pause

%-------------------------------------------------------------------------------
% load the matrices
%-------------------------------------------------------------------------------

% known to be positive definite / indefinite:
known_posdef = [ 939 1252 1267 1268 1423 1453 1455 2541:2547 ] ;
known_indef = [ 1348:1368 1586 1411 1901:1905] ;

% known to be irreducible, but dmperm takes too long:
known_irreducible = [ 1902:1905 ] ;
% known_irreducible = [ ] ;

t = tic ;

for k = 1:length (matrixlist)

    %---------------------------------------------------------------------------
    % get the matrix
    %---------------------------------------------------------------------------

    id = matrixlist (k) ;
    ffile = deblank (files {id}) ;
    fprintf ('\n============================== Matrix %d: %s\n', id, ffile) ;
    if (filesize (id) == -1)
	fprintf ('skip this file\n') ;
	continue ;
    end
    load ([topdir 'mat/' ffile]) ;

    % display the Problem struct
    disp (Problem) ;

    %---------------------------------------------------------------------------
    % get all stats
    %---------------------------------------------------------------------------

    kinds {id} = Problem.kind ;

    fprintf ('%s/%s\n', ss_index.Group {id}, ss_index.Name {id}) ;

    if (~isequal (Problem.name, [ss_index.Group{id} '/' ss_index.Name{id}]))
        error ('name mismatch!') ;
    end
    if (Problem.id ~= id)
        error ('id mismatch!') ;
    end

    skip_chol = (any (id == known_posdef) || any (id == known_indef)) ;
    skip_dmperm = any (id == known_irreducible) ;

    if (isfield (Problem, 'Zeros'))
	stats = ssstats (Problem.A, Problem.kind, skip_chol, ...
            skip_dmperm, Problem.Zeros) ;
    else
	stats = ssstats (Problem.A, Problem.kind, skip_chol, ...
            skip_dmperm) ;
    end

    %---------------------------------------------------------------------------
    % fix special cases
    %---------------------------------------------------------------------------

    if (stats.posdef < 0)
	if (any (id == known_posdef))
	    fprintf ('known posdef\n') ;
	    stats.posdef = 1 ;
	elseif (any (id == known_indef))
	    fprintf ('known indef\n') ;
	    stats.posdef = 0 ;
	end
    end

    if (any (id == known_irreducible) && stats.sprank < 0)
	% full sprank, and not reducible to block triangular form,
	% but dmperm takes too long
	fprintf ('known irreducible\n') ;
	stats.sprank = stats.nrows  ;
	stats.nblocks = 1 ;
    end

    % display the stats
    disp (stats) ;

    %---------------------------------------------------------------------------
    % save the stats in the index
    %---------------------------------------------------------------------------

    ss_index.nrows (id) = stats.nrows ;
    ss_index.ncols (id) = stats.ncols ;
    ss_index.nnz (id) = stats.nnz ;
    ss_index.nzero (id) = stats.nzero ;
    ss_index.pattern_symmetry (id) = stats.pattern_symmetry ;
    ss_index.numerical_symmetry (id) = stats.numerical_symmetry ;
    ss_index.isBinary (id) = stats.isBinary ;
    ss_index.isReal (id) = stats.isReal ;
    ss_index.nnzdiag (id) = stats.nnzdiag ;
    ss_index.posdef (id) = stats.posdef ;
    ss_index.amd_lnz (id) = stats.amd_lnz ;
    ss_index.amd_flops (id) = stats.amd_flops ;
    ss_index.amd_vnz (id) = stats.amd_vnz ;
    ss_index.amd_rnz (id) = stats.amd_rnz ;
    ss_index.nblocks (id) = stats.nblocks ;
    ss_index.sprank (id) = stats.sprank ;
    ss_index.RBtype (id,:) = stats.RBtype ;
    ss_index.cholcand (id) = stats.cholcand ;
    ss_index.ncc (id) = stats.ncc ;
    ss_index.isND (id) = stats.isND ;
    ss_index.isGraph (id) = stats.isGraph ;
    ss_index.lowerbandwidth (id) = stats.lowerbandwidth ;
    ss_index.upperbandwidth (id) = stats.upperbandwidth ;
    ss_index.rcm_lowerbandwidth (id) = stats.rcm_lowerbandwidth ;
    ss_index.rcm_upperbandwidth (id) = stats.rcm_upperbandwidth ;
    ss_index.xmin (id) = stats.xmin ;
    ss_index.xmax (id) = stats.xmax ;

    %---------------------------------------------------------------------------
    % clear the problem and save the index and ssstats.csv
    %---------------------------------------------------------------------------

    clear Problem
    fprintf ('time since last save: %g\n', toc (t)) ;
    if (toc (t) > 20 || k == length (matrixlist))
        t = tic ;
        fprintf ('\n ... saving ss_index ...\n') ;
        save ss_index ss_index

        fprintf ('\nCreating ssstats.csv in current directory:\n')
        fprintf ('%s/ssstats.csv\n', pwd) ;
        f = fopen ('ssstats.csv', 'w') ;
        fprintf (f, '%d\n', nmat) ;
        fprintf (f, '%s\n', ss_index.LastRevisionDate) ;
        for id = 1:nmat
            fprintf (f,'%s,%s,%d,%d,%d,%d,%d,%d,%d,%.16g,%.16g,%s\n', ...
                ss_index.Group {id}, ...
                ss_index.Name {id}, ...
                ss_index.nrows (id), ...
                ss_index.ncols (id), ...
                ss_index.nnz (id), ...
                ss_index.isReal (id), ...
                ss_index.isBinary (id), ...
                ss_index.isND (id), ...
                ss_index.posdef (id), ...
                ss_index.pattern_symmetry (id), ...   % formatted with %.16g
                ss_index.numerical_symmetry (id), ... % formatted with %.16g
                kinds {id}) ;
        end
        fclose (f) ;

        % flush the diary
        if (strcmp (get (0, 'Diary'), 'on'))
            diary off
            diary on
        end
    end
end
