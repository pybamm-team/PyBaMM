function install_SJget
%INSTALL_SJGET sets the MATLAB path to include a path to SJget
% Sets the MATLAB path to include the interface to the SJSU Singular Matrix 
% Collection
%
% Example
%   install_SJget
%
% See also gallery.

% Copyright 2012, Leslie Foster and Timothy A Davis

% This is the file spqr_rank/private/install_SJget.m.  The SJget toolbox
% is in spqr_rank/SJget.

if ( exist ('SJget', 'file') ~= 2)
    here = mfilename ('fullpath') ;
    k = strfind (here, 'private') ;
    SJgetpath = [here(1:k-1) 'SJget'] ;
    fprintf ('Adding SJget to your path:\n%s\n', SJgetpath) ;
    addpath(SJgetpath)
    disp (' ') ;
    disp ('Saving the current MATLAB path so that access to matrices') ;
    disp ('from the SJSU singular matrix collection is available when') ;
    disp ('restarting MATLAB.') ;
    lastwarn ('') ;
    try
        savepath    % comment out this line to avoid saving the path
        err = lastwarn ;
    catch me
        err = me.message ;
    end
    if (~isempty (err))
        fprintf ('error: %s\n', err) ;
        fprintf ('unable to save path, see ''doc pathdef'' for more info\n') ;
    end
    disp (' ') ;
    disp (' ') ;
end

