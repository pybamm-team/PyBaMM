function umfpack_report (Control, Info)
%UMFPACK_REPORT prints optional control settings and statistics
%
%   Example:
%       umfpack_report (Control, Info) ;
%
% Prints the current Control settings for umfpack, and the statistical
% information returned by umfpack in the Info array.  If Control is
% an empty matrix, then the default control settings are printed.
%
% Control and Info are structs.
%
% Alternative usages:
%
%       umfpack_report ([ ], Info) ;    print the default control parameters
%                                       and the Info array.
%       umfpack_report (Control) ;      print the control parameters only.
%       umfpack_report ;                print the default control parameters
%
% See also umfpack, umfpack_make, umfpack_details,
% umfpack_demo, and umfpack_simple.

% Copyright 1995-2007 by Timothy A. Davis.

%-------------------------------------------------------------------------------
% get inputs, use defaults if input arguments not present
%-------------------------------------------------------------------------------

% The contents of Control and Info are defined in umfpack.h
if (nargin < 1 || isempty (Control))
    Control = umfpack ;
end
fprintf ('\nUMFPACK Control:\n') ;
disp (Control) ;
if (nargin > 1 && ~isempty (Info))
    fprintf ('\nUMFPACK Info:\n') ;
    disp (Info) ;
end

