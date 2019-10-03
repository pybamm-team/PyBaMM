function private_stream = spqr_repeatable (seed)
%SPQR_REPEATABLE ensure repeatable results, or use the default random stream.
% Uses RandStream for repeatable results, which is not available on MATLAB 7.6
% or earlier (R2008a).  For that version of MATLAB (or earlier), the seed is
% ignored and the default random stream is always used.
% Not user-callable.

% Copyright 2012, Leslie Foster and Timothy A Davis

% Since this code is called very often, use 'version', which is perhaps 100
% times faster than 'verLessThan'.
v = sscanf (version, '%d.%d.%d') ;
v = 10.^(0:-1:-(length(v)-1)) * v ;

if (v < 7.7)
    % MATLAB 7.6 and earlier do not have RandStream, so spqr_rank ignores
    % the opts.repeatable option and just uses the default random stream.
    private_stream = [ ] ;
elseif (seed == 1)
    % use a new strearm with the default random seed
    private_stream = RandStream ('mt19937ar') ;
elseif (seed > 1)
    % use a new stream with a specific random seed
    private_stream = RandStream ('mt19937ar', 'seed', seed) ;
else
    % do not use the private stream
    private_stream = [ ] ;
end

