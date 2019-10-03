function Problem = SJget (matrix, SJ_Index, refresh_mat)
%SJGET loads a matrix from the SJSU Singular Matrix Collection.
%
%   Problem = SJget(matrix) loads a matrix from the SJSU Singular matrix collection,
%   specified as either a number (1 to the # of matrices in the collection) or
%   as a string (the name of the matrix).  With no input parameters, index=SJget
%   returns an index of matrices in the collection.  A local copy of the matrix
%   is saved (be aware that as of August 2008 the entire collection is over 1GB
%   in size).  If no input or output arguments are provided, the index is
%   printed.  With a 2nd parameter (Problem = SJget (matrix, index)), the index
%   file is not loaded.  This is faster if you are loading lots of matrices.
%   Also SJget('refresh') will force downloading the index file from the 
%   SJsingular web site. SJget(matrix,'refresh') or 
%   SJget(matrix,index,'refresh') will force downloading Problem
%   matrix from the SJsingular web site, rather than using a 
%   locally stored copy.
%
%   Examples:
%     index = SJget ;                     % loads index of the collection
%     index = SJget ('refresh') ;         % forces download of new index
%                                         % from the SJsingular web site
%     Problem = SJget (6)                 % get the 6th problem
%     Problem = SJget ('HB/ash292')       % get a problem by name
%     Problem = SJget (6,'refresh')       % forces download of 6th problem
%                                         % from the SJSingular web site
%     Problem = SJget (6, index)          % alternatives to get problems
%     Problem = SJget ('HB/ash292', index)
%     Problem = SJget (6,index,'refresh') % forces download of 6th problem
%                                         % from the SJsingular web site
%
%   or one can search using SJget:
%
%   index = SJget ;     % get index of the SJSU Singular Matrix Collection
%   ids = find (1000 <= index.numrank & index.numrank <= 1200) ;
%   [ignore, i] = sort (index.numrank (ids)) ;    %sort by numrank
%   ids = ids (i) ;
%   for id = ids
%      Prob = SJget (id);        % Prob is a struct (matrix, name, meta-data, ...)
%	   A = Prob.A ;              % A has numerical rank between 1000 and 1200
%	   disp([index.numrank(id), size(A)]) % list the numerical rank and
%                                % size of A
%   end

%   See also SJgrep, SJweb, SJget_example, SJget_defaults, urlwrite.

%   Derived from the ssget toolbox on March 18, 2008, modified 2/1/2009
%   Copyright 2007, Tim Davis, University of Florida.

%-------------------------------------------------------------------------------
% get the parameter settings
%-------------------------------------------------------------------------------

params = SJget_defaults ;
indexfile = sprintf ('%sSJ_Index.mat', params.dir) ;
indexurl = sprintf ('%s/SJ_Index.mat', params.url) ;

%-------------------------------------------------------------------------------
% get the index file (download a new one if necessary)
%-------------------------------------------------------------------------------

refresh_matrix = 0;
if nargin >= 2
    % SJget (..,'refresh') forces downloading matrix .. from the web    
    if (ischar (SJ_Index))
        if (strcmp(SJ_Index, 'refresh'))
            refresh_matrix = 1;
        end
    end
end
if nargin == 3
    % SJget (..,..,'refresh') forces downloading matrix .. from the web
    if (ischar (refresh_mat))
        if (strcmp (refresh_mat, 'refresh'))
            refresh_matrix = 1 ;
        end
    end
end

refresh = 0 ;
if nargin == 0
    % if the user passed in a zero or no argument at all, return the index file
    matrix = 0 ;
else
    % SJget ('refresh') downloads the latest index file from the web
    if (ischar (matrix))
        if (strcmp (matrix, 'refresh'))
            matrix = 0 ;
            refresh = 1 ;
        end
    end
end

if (~refresh)
    try
        % load the existing index file
        if (nargin < 2 )
        load (indexfile) ;
        end
        if (nargin == 2 && refresh_matrix == 1)
            load (indexfile) ;
        end
        % see if the index file is old; if so, download a fresh copy
        refresh = (SJ_Index.DownloadTimeStamp + params.refresh < now) ;
    catch
        % oops, no index file.  download it.
        refresh = 1 ;
    end
end

if (refresh)
    % a new SJ_Index.mat file to get access to new matrices (if any)
    fprintf ('downloading %s\n', indexurl) ;
    fprintf ('to %s\n', indexfile) ;
    urlwrite (indexurl, indexfile) ;
    load (indexfile) ;
    SJ_Index.DownloadTimeStamp = now ;
    save (indexfile, 'SJ_Index') ;
end

%-------------------------------------------------------------------------------
% return the index file if requested
%-------------------------------------------------------------------------------

% if the user passed in a zero or no argument at all, return the index file
if nargin == 0
    matrix = 0 ;
end

if (matrix == 0)
    if (nargout == 0)
        % no output arguments have been passed, so print the index file
        fprintf ('\nSJSU Singular matrix collection index:  %s\n', ...
            SJ_Index.LastRevisionDate) ;
        fprintf ('\nLegend:\n') ;
        fprintf ('num. rank:    numerical rank \n') ;
        fprintf ('struc. rank:  structural rank \n') ;        
        fprintf ('type:      real\n') ;
        fprintf ('           complex\n') ;
        fprintf ('           binary:  all entries are 0 or 1\n') ;
        nmat = length (SJ_Index.nrows) ;
        for j = 1:nmat
            if (mod (j, 25) == 1)
                fprintf ('\n') ;
                fprintf ('ID   Group/Name                       nrows-by-  ncols  num. rank  struct. rank  type\n') ;
            end
            s = sprintf ('%s/%s', SJ_Index.Group {j}, SJ_Index.Name {j}) ;
            fprintf ('%4d %-30s %7d-by-%7d %10d ', ...
            j, s, SJ_Index.nrows (j), SJ_Index.ncols (j), SJ_Index.numrank (j)) ;
            %psym = SJ_Index.pattern_symmetry (j) ;
            %nsym = SJ_Index.numerical_symmetry (j) ;
            fprintf('%13d ', SJ_Index.sprank (j) );
            %if (psym < 0)
            %    fprintf ('  -  ') ;
            %else
            %    fprintf (' %4.2f', psym) ;
            %end
            %if (nsym < 0)
            %    fprintf ('  -  ') ;
            %else
            %    fprintf (' %4.2f', nsym) ;
            %end
            if (SJ_Index.isBinary (j))
                fprintf (' binary\n') ;
            elseif (~SJ_Index.isReal (j))
                fprintf (' complex\n') ;
            else
                fprintf (' real\n') ;
            end
        end
    else
        Problem = SJ_Index ;
    end
    return ;
end

%-------------------------------------------------------------------------------
% determine if the matrix parameter is a matrix index or name
%-------------------------------------------------------------------------------

[group matrix id] = SJget_lookup (matrix, SJ_Index) ;

if (id == 0)
    error ('invalid matrix') ;
end

%-------------------------------------------------------------------------------
% download the matrix (if needed) and load it into MATLAB

matdir = sprintf ('%s%s%s%s.mat', params.dir, group) ;
matfile = sprintf ('%s%s%s.mat', matdir, filesep, matrix) ;
maturl = sprintf ('%s/%s/%s.mat', params.url, group, matrix) ;

if (~exist (matdir, 'dir'))
    mkdir (matdir) ;
end

if (exist (matfile, 'file') && refresh_matrix == 0)
    load (matfile)
else
    fprintf ('downloading %s\n', maturl) ;
    fprintf ('to %s\n', matfile) ;
    urlwrite (maturl, matfile) ;
    load (matfile)
    save (matfile, 'Problem') ;
end
