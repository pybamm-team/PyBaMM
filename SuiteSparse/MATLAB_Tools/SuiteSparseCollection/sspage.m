function sspage (matrix, index)
%SSPAGE create images for a matrix in SuiteSparse Matrix Collection
%
% Usage:
%      sspage (matrix, index)
%
% matrix: id or name of matrix to create the images for.
% index: the ss_index, from ssget.
%
% Example:
%
%   sspage (267)
%   sspage ('HB/west0479')
%
% See also ssget, cspy, ssgplot.

% Copyright 2006-2018, Timothy A. Davis

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

if (nargin < 2)
    index = ssget ;
end

%-------------------------------------------------------------------------------
% get the Problem and its contents
%-------------------------------------------------------------------------------

Problem = ssget (matrix,index) ;
disp (Problem)
fullname = Problem.name ;
s = strfind (fullname, '/') ;
grp = fullname (1:s-1) ;
name = fullname (s+1:end) ;
id = Problem.id ;

% create the primary directory
topdir = sslocation ;
files = [topdir 'files'] ;
if (~exist (files, 'dir'))
    mkdir (files) ;
end

% create the group directory
if (~exist ([files '/' grp], 'dir'))
    mkdir ([files '/' grp]) ;
end

% determine the full path of the problem
fullpath = regexprep ([files '/' fullname], '[\/\\]', '/') ;

ptitle = Problem.title ;

z = 0 ;
if (isfield (Problem, 'Zeros'))
    z = nnz (Problem.Zeros) ;
    Problem = rmfield (Problem, 'Zeros') ;
end

nblocks = index.nblocks (id) ;
ncc = index.ncc (id) ;

has_b = isfield (Problem, 'b') ;
has_x = isfield (Problem, 'x') ;
has_aux = isfield (Problem, 'aux') ;

if (has_b)
    b = Problem.b ;
    Problem = rmfield (Problem, 'b') ;
    if (iscell (b))
	b = sprintf ('cell %d-by-%d\n', size (b)) ;
    elseif (issparse (b))
	b = sprintf ('sparse %d-by-%d\n', size (b)) ;
    else
	b = sprintf ('full %d-by-%d\n', size (b)) ;
    end
end

if (has_x)
    x = Problem.x ;
    Problem = rmfield (Problem, 'x') ;
    if (iscell (x))
	x = sprintf ('cell %d-by-%d\n', size (x)) ;
    elseif (issparse (x))
	x = sprintf ('sparse %d-by-%d\n', size (x)) ;
    else
	x = sprintf ('full %d-by-%d\n', size (x)) ;
    end
end

nodename = [ ] ;
if (has_aux)
    aux = Problem.aux ;
    Problem = rmfield (Problem, 'aux') ;
    auxfields = fields (aux) ;
    has_coord = isfield (aux, 'coord') ;
    has_nodename = isfield (aux, 'nodename') ;
    auxs = cell (1, length (auxfields)) ;
    for k = 1:length(auxfields)
	siz = size (aux.(auxfields{k})) ;
	if (iscell (aux.(auxfields{k})))
	    auxs {k} = sprintf ('cell %d-by-%d\n', siz) ;
	elseif (issparse (aux.(auxfields{k})))
	    auxs {k} = sprintf ('sparse %d-by-%d\n', siz) ;
	else
	    auxs {k} = sprintf ('full %d-by-%d\n', siz) ;
	end
    end
    if (has_coord)
	xyz = aux.coord ;
    end
    if (has_nodename)
	nodename = aux.nodename ;
    end
    clear aux
else
    has_coord = 0 ;
end

kind = Problem.kind ;
if (isfield (Problem, 'notes'))
    notes = Problem.notes ;
else
    notes = '' ;
end

au = Problem.author ;
ed = Problem.ed ;
da = Problem.date ;

m = index.nrows (id) ;
n = index.ncols (id) ;
nz = index.nnz (id) ;
nnzdiag = index.nnzdiag (id) ;

if (strfind (kind, 'graph'))
    bipartite = ~isempty (strfind (kind, 'bipartite')) ;
    directed = ~isempty (regexp (kind, '\<directed', 'once')) ;
else
    bipartite = (m ~= n) ;
    directed = (index.pattern_symmetry (id) < 1) ;
end

%-------------------------------------------------------------------------------
% create the pictures
%-------------------------------------------------------------------------------

A = Problem.A ;
clear Problem

%---------------------------------------------------------------------------
% create the gplot
%---------------------------------------------------------------------------

do_gplot = has_coord ;
if (do_gplot)
    ssgplot (A, xyz, directed, nodename) ;
    print (gcf, '-dpng', '-r128', [fullpath '_gplot.png']) ;
    print (gcf, '-dpng', '-r512', [fullpath '_gplot_big.png']) ;
end

%---------------------------------------------------------------------------
% create the thumbnail picture
%---------------------------------------------------------------------------

cspy (A, 16) ;
print (gcf, '-dpng', '-r12', [fullpath '_thumb.png']) ;

%---------------------------------------------------------------------------
% create the regular picture
%---------------------------------------------------------------------------

cspy (A, 128) ;
print (gcf, '-dpng', '-r64', [fullpath '.png']) ;

%---------------------------------------------------------------------------
% create the dmperm figure, but not for graphs
%---------------------------------------------------------------------------

do_dmspy = (nblocks > 1) ;
if (do_dmspy)
    try
        cs_dmspy (A, 128) ;
        title ('Dulmage-Mendelsohn permutation') ;
    catch
        fprintf ('dmspy failed\n') ;
        delete ([fullpath '_dmperm.png']) ;
        do_dmspy = 0 ;
    end
    print (gcf, '-dpng', '-r64', [fullpath '_dmperm.png']) ;
end

%---------------------------------------------------------------------------
% create the ccspy figure
%---------------------------------------------------------------------------

do_scc = (ncc > 1) ;
if (do_dmspy && m == n && nnzdiag == n)
    % don't do scc for a square matrix with zero-free diagonal
    do_scc = 0 ;
end
if (do_scc)
    try
        ccspy (A, bipartite, 128) ;
        if (bipartite)
            title ('connected components of the bipartite graph') ;
        else
            title ('strongly connected components of the graph') ;
        end
        print (gcf, '-dpng', '-r64', [fullpath '_scc.png']) ;
    catch
        fprintf ('ccspy failed\n') ;
        delete ([fullpath '_cc.png']) ;
        do_scc = 0 ;
    end
end

