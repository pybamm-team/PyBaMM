function testRB1
%testRB1: test the RBio toolbox.
%
% Example:
%   testRB1
%
% See also UFget, RBread, RBreade, testRB2.

% Copyright 2006, Timothy A. Davis

files = {
'bcsstk01.rb'
'farm.rb'
'lap_25.pse'
'lap_25.rb'
'west0479.rb'
'west0479.rua' } ;

mtypes = {
'rsa',
'ira',
'psa',
'psa',
'rua'
'rua' } ;

for k = 1:length(files)
    file = files {k} ;
    % fprintf ('%s : ', file) ;
    if (file (end) == 'e')
	[A Z] = RBreade (file) ;
    else
	[A Z] = RBread (file) ;
    end
    mtype = RBtype (A) ;
    if (any (mtype ~= mtypes {k}))
        fprintf ('test failed: %s %s type differs\n', mtype, mtypes {k}) ;
    end
    RBwrite ('temp.rb', A, Z) ;
    [A2 Z2] = RBread ('temp.rb') ;
    if (~isequal (A, A2))
	fprintf ('test failed: %s (A differs)\n', file) ;
	error ('!') ;
    end
    if (~isequal (Z, Z2))
	fprintf ('test failed: %s (Z differs)\n', file) ;
	error ('!') ;
    end
end

load west0479
C = west0479 ;
RBwrite ('temp.rb', C, 'WEST0479 chemical eng. problem', 'west0479') ;
A = RBread ('temp.rb') ;
if (~isequal (A, C))
    error ('test failed: west0479 (MATLAB version)') ;
end
if (any (mtype ~= 'rua'))
    fprintf ('test failed: %s %s type differs\n', mtype, 'rua') ;
end

if (~strcmp (RBtype (A), 'rua'))
    error ('test failed: RBtype(A)\n') ;
end
if (~strcmp (RBtype (spones (A)), 'pua'))
    error ('test failed: RBtype(spones(A))\n') ;
end
if (~strcmp (RBtype (2*spones (A)), 'iua'))
    error ('test failed: RBtype(2*spones(A))\n') ;
end
C = A+A' ;
if (~strcmp (RBtype (C), 'rsa'))
    error ('test failed: RBtype(A+A'')\n') ;
end

delete ('temp.rb') ;
fprintf ('RB test 1: passed\n') ;
