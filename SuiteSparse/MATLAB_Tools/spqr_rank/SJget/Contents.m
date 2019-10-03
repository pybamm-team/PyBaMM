% SJget: MATLAB interface for the SJSU Singular Matrix Collection.
%
% Files of most interest:
%    SJget.m		    primary user interface,
%                       SJget will get summary information
%                       for all matrices or detailed
%                       information for a single matrix
%    SJget_install.m	installation for SJget
%
% additional files
%    Contents.m		    this file, help for SJget
%    README.txt		    additional help
%    SJget_example.m	demo for SJget
%    SJplot.m           picture a plot of the full or partial
%                       singular value spectrum
%    SJweb.m		    opens the URL for a matrix or collection
%    SJrank.m           calculates numerical rank for a specified
%                       tolerance using precomputed singular values
%
% some additional utilities:
%    SJget_defaults.m	default parameter settings for SJget
%    SJget_lookup.m	get the group, name, and id of a matrix
%    SJgrep.m           search for matrices in the SJSU Singular 
%                       Matrix Collection.
%
% Example:
%   help SJget

% Derived from the ssget toolbox on 18 March 2008.
% Copyright 2007, Timothy A. Davis

