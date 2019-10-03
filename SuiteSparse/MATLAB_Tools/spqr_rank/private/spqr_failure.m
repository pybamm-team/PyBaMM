function [stats x N NT] = spqr_failure (f, stats, get_details, start_tic)
%SPQR_FAILURE clean-up from failure
% Not user-callable.

% Copyright 2012, Leslie Foster and Timothy A Davis

stats.flag = f ;
x = [ ] ;
N = [ ] ;
NT = [ ] ;
if get_details == 1
    % order the fields of stats in a convenient order (the fields when
    % get_details is 0 or 2 are already in a good order)
    stats.time = -1 ;
    stats = spqr_rank_order_fields (stats) ;
    stats.time = toc (start_tic) ;
end
