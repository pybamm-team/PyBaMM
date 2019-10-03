function [numrank,flag]= SJrank(matrix,tol)
%   SJrank(matrix,tol) calculates numerical rank of a matrix using the 
%   precomputed singular values in the SJsingular database.
%
%   Input:
%       matrix -- either:
%          a problem structure downloaded from the SJsingular database
%          a number: 1 to the # of matrices in the collection
%          a string: the name of a matrix in the collection
%          In the last two cases SJget must be installed first from
%          http://www.math.sjsu.edu/singular/matrices/SJget.html
%       tol (optional) -- calculate the number of singular values greater 
%          than tol (the default value is  max(size(A))*eps(norm of A))
%   Output:
%       numrank -- the number of singular values greater than tol for
%          the matrix A defined in the problem structure. numrank will
%          be -1 if the SJsingular database does not have sufficient
%          information (i.e. precomputed singular values) to determine the
%          numerical rank at the current tolerance.
%       flag  -- 
%          flag = 0 is consistent with a correctly calculated numerical
%             rank -- tol does not lie in the interval of uncertainty of 
%             any calculated singular value
%          flag = 1 if the calculated numerical rank may be incorrect
%             since tol lies in the interval of uncertainty of at least
%             one calculated singular value
%          flag = -1 when no calculations have been done to check the
%             accuracy of the numerical rank.  This will occur when
%             there are no precomputed bounds of the accuracy of the
%             calculated singular values or when numrank = -1.
%          flag is NaN if any component of svals or svals_err is not finite
%   Examples:
%         %if matrix information has already been downloaded
%         nrank = SJrank(Problem)
%         % or    (assuming that SJget is installed)
%         [nrank, flag] = SJrank('Meszaros/model6')
%         % or
%         tol = 1.e-5;
%         nrank = SJrank(403, tol)
%   See also SJplot.

% Written by Leslie Foster 9/5/2008, Version 1.0
% Copyright 2008, Leslie Foster


   % check that the input is ok
   if isstruct(matrix) 
       if isfield(matrix,'svals')
          svals = matrix.svals;
          if isfield(matrix,'svals_err')
             svals_err = matrix.svals_err;
          end
       else
           error(['Invalid matrix. The structure ', ...
               inputname(1),' is missing a svals field.'])
       end
   else
       if isempty(which('SJget'))
           error(['SJget is not installed. It is available at ',...
               'http://www.math.sjsu.edu/singular/matrices/SJget.html .'])
       end
       matrix = SJget(matrix);
       svals = matrix.svals;
       if isfield(matrix,'svals_err')
          svals_err = matrix.svals_err;
       end
   end 
   [m,n] = size(matrix.A);   
   if nargin < 2 || isempty(tol)
      tol = max(m,n)*eps(svals(1));
   end
   if nargin >= 2 && tol < 0
      error('tolerance is negative')
   end
   

    % calculate the numerical rank
    numrank = find(svals > tol);
    loc_svals = find( svals >= 0 );
    if ( isempty(numrank) )
        if isempty(loc_svals)
            numrank = -1;
        else 
            numrank = 0;
        end
    else
        numrank = numrank(end);
        if numrank < min(m,n) && svals(numrank+1) < 0
            numrank = -1;
        end
    end

    if nargout >= 1
        %determine if error bounds on sing. values are small enough to
        %   be consistent with a correctly calculated numerical rank
        if numrank >= 0 && exist('svals_err','var')
          if ( min(isfinite(svals)) == 1 && min(isfinite(svals_err)) == 1 )
            loc_svals_err = find(svals_err >= 0 );
            svp= svals(loc_svals_err);
            svp_err = svals_err(loc_svals_err);
            % check, without subtraction, if tol is in any 
            % interval [svp-svp_err, svp+svp_err]            
            flag = max(  ( tol <= (svp_err +  svp) )  & ... 
                       ( (svp <= tol + svp_err) ) ) ;
          else
            flag = NaN;
          end
        else
            flag = -1;
        end
    end
    
end
