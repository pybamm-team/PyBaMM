function rho = ldl_normest (A, L, D)
%LDL_NORMEST estimate the 1-norm of A-L*D*L' without computing L*D*L'
%
% Example:
%
%       rho = ldl_normest (A, L, D)
%
% which estimates the computation of the 1-norm:
%
%       rho = norm (A-L*D*L', 1)
%
% See also condest, normest

% Copyright 2006-2007, William W. Hager and Timothy A. Davis

[m n] = size (A) ;

if (m ~= n)
    error ('A must be square') ;
end
if (nnz (A-A') ~= 0)
    error ('A must be symmetric') ;
end

if (nargin < 3)
    D = speye (n) ;
end

notvisited = ones (m, 1) ;  % nonvisited(j) is zero if j is visited, 1 otherwise
rho = 0 ;    % the global rho

for trial = 1:3 % {

   x = notvisited ./ sum (notvisited) ;
   rho1 = 0 ;    % the current rho for this trial

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%% COMPUTE Ex1 = E*x EFFICIENTLY: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Ex1 = (A*x) - L*(U*x) ;
   Ex1 = (A*x) - L*(D*(L'*x)) ;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   rho2 = norm (Ex1, 1) ;

   while rho2 > rho1 % {

        rho1 = rho2 ;
        y = 2*(Ex1 >= 0) - 1 ;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% COMPUTE z = E'*y EFFICIENTLY: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % z = (A'*y) - U'*(L'*y) ;
        z = (A*y) - L*(D*(L'*x)) ;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [zj, j] = max (abs (z .* notvisited)) ;
        j = j (1) ;
        if (abs (z (j)) > z'*x) % {
            x = zeros (m, 1) ;
            x (j) = 1 ;
            notvisited (j) = 0 ;
        else % } {
            break ;
        end % }

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% COMPUTE Ex1 = E*x EFFICIENTLY: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Ex1 = (A*x) - L*(D*(L'*x)) ;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        rho2 = norm (Ex1, 1) ;

    end % }

    rho = max (rho, rho1) ;

end % }
