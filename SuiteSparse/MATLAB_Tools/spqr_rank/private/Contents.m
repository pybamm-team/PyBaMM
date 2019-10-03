% PRIVATE
%
%   Routines called by the SPQR_RANK package.  These routines are not 
%   intended to be directly called by the user.
%
% Files
%   install_SJget          - sets the MATLAB path to include a path to SJget
%   spqr_failure           - clean-up from failure
%   spqr_rank_assign_stats - set flag and other statistics.
%   spqr_rank_deflation    - constructs pseudoinverse or basic solution using deflation.
%   spqr_rank_form_basis   - forms the basis for the null space of a matrix.
%   spqr_rank_get_inputs   - get the inputs and set the default options.
%   spqr_rank_order_fields - orders the fields of stats in a convenient order.
%   spqr_repeatable        - ensure repeatable results, or use the default random stream.
%   spqr_wrapper           - wrapper around spqr to get additional statistics
%   tol_is_default         - return true if tol is default, false otherwise
