function test27
%TEST27 test nesdis with one matrix (HB/west0479)
% Example:
%   test27
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

fprintf ('=================================================================\n');
fprintf ('test27: test nesdis\n') ;

Prob = UFget ('HB/west0479') ;
dg (Prob.A) ;

fprintf ('test27 passed\n') ;
