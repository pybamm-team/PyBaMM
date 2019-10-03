function waitmex
%WAITMEX a small library for using a waitbar within a C mexFunction.
% This waitmex function compiles an example mexFunction, adds the current
% directory to your MATLAB path, and runs two examples (one in C, one in M).
%
%    in C                           MATLAB M-file equivalent
%    ----                           ------------------------
%    h = waitbar_create (x,msg) ;   h = waitbar (x,msg)
%    waitbar_update (x,h,NULL) ;    waitbar (x,h)
%    waitbar_update (x,h,msg) ;     waitbar (x,h,msg)
%    waitbar_destroy (h) ;          close (h)
%    waitbar_return (h) ;           for returning h from a mexFunction
%
% Example:
%   waitmex
%
%
% See also waitex, waitexample, waitbar, pathtool.

% Copyright 2007, T. Davis

help waitmex

fprintf ('\ncompiling an example:\n') ;
fprintf ('mex waitexample.c waitmex.c\n') ;
mex waitexample.c waitmex.c
addpath (pwd) ;

fprintf ('\ntrying the example mexFunction:\n') ;
fprintf ('waitexample\n') ;
waitexample

fprintf ('\ntrying the m-file equivalent of waitexample:\n') ;
fprintf ('waitex\n') ;
waitex
