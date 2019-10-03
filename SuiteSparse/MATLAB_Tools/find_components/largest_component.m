function C = largest_component (A)
%LARGEST_COMPONENT finds the largest connected component in an image.
% C = largest_component (A) returns an image C whose entries are equal to 1
% if A(i,j) is in the largest component of A, or zero otherwise.  In case of
% a tie, the largest component with the largest label A(i,j) is returned.
% If still tied, the component the smallest index i is returned (where i is the
% linear index of A(i) for all entries in the component).
%
% Example:
%
%   A = [ 1 2 2 3
%         1 1 2 3
%         0 0 1 2
%         0 1 3 3 ]
%   C = largest_component (A)
%
%   returns C = [
%         0 1 1 0
%         0 0 1 0
%         0 0 0 0
%         0 0 0 0 ]
%
% See also FIND_COMPONENTS, FIND_COMPONENTS_EXAMPLE, DMPERM

% Copyright 2008, Timothy A. Davis, http://www.suitesparse.com

% return the new binary image with just the largest component
C = zeros (size (A), class (A)) ;
C (find_components (A)) = 1 ;
