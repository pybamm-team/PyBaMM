function params = SJget_defaults
%SJGET_DEFAULTS returns default parameter settings for SJget.
%   Usage:  params = SJget_defaults ; Returns the default parameter settings for
%   SJget.  This file may be editted to change these settings:
%
%   params.url: SJSU Singular matrix web site
%   params.dir: your local directory for downloaded singular matrices.
%   params.refresh:  how many days should elapse before re-downloading the
%       index file (for obtaining access to new matrices in the collection).
%
%   Example:
%       params = SJget_defaults ;
%
%   See also SJget.

%   Derived from the ssget toolbox on March 18, 2008.
%   Copyright 2007, Tim Davis, University of Florida.


%-------------------------------------------------------------------------------
% define base information about the SJSU Singular matrix collection
params.site_name = 'SJSU Singular Matrix Database';
params.site_url = 'http://www.math.sjsu.edu/singular/matrices';
params.maintainer = 'Leslie Foster';
params.maintainer_url = 'http://www.math.sjsu.edu/~foster/';

%-------------------------------------------------------------------------------
% location of the SJSU Singular matrix collection
params.url = [ params.site_url '/mat' ] ;

%-------------------------------------------------------------------------------
% decode the current directory for this M-file
s = which (mfilename) ;
i = find (s == filesep) ;
s = s (1:i(end)) ;

%-------------------------------------------------------------------------------
% define the directory to download into.  Should end in file separator.
params.dir = [ s 'mat' filesep ] ;

%-------------------------------------------------------------------------------
% define how often to check for a new index file (in # of days)
% inf will force the program to ignore the need to refresh
params.refresh = 90 ;
