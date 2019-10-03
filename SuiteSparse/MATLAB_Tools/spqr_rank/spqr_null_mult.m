function B = spqr_null_mult (N,A,method)
%SPQR_NULL_MULT multiplies a matrix by numerical null space from spqr_rank methods
%
%  Multiplies the matrix A by N, the orthonormal basis for
%  a numerical null space produced by spqr_basic, spqr_null, spqr_pinv,
%  or spqr_cod.  N can be stored either implicitly (see the above
%  routines for details) or as an explicit matrix.
%
% Example of use:  B = spqr_null_mult(N,A,method) ;
%    method = 0: N'*A  (the default)
%    method = 1: N*A
%    method = 2: A*N'
%    method = 3: A*N
% Also if N is stored implicitly then to create an explicit representation
% of the orthonormal null space basis
%    Nexplicit = spqr_null_mult(N,eye(size(N.X,2)),1) ;
% creates a full matrix and
%    Nexplicit = spqr_null_mult(N,speye(size(N.X,2)),1) ;
% creates a sparse matrix.
%
% If N is stored implicitly then N can have fields:
%    N.P contains a permutation vector (N.P is not always present)
%    N.Q contains sparse Householder transforms from spqr
%    N.X is a sparse matrix with orthonormal columns
%
% N.kind determines how the implicit orthonormal basis is represented.
%   N.kind = 'P*Q*X':  the basis is N.P*N.Q*N.X
%   N.kind = 'Q*P*X':  the basis is N.Q*N.P*N.X
%   N.kind = 'Q*X':    the basis is N.Q*N.X, and N.P does not appear in N
%
% Examples
%
%   N = spqr_null (A) ;              % find nullspace of A
%   AN = spqr_null_mult (N,A,3) ;    % compute A*N, which will have a low norm.
%
% See also spqr, spqr_basic, spqr_cod, spqr_null, spqr_pinv, spqr_ssi, spqr_ssp.

% Copyright 2012, Leslie Foster and Timothy A Davis.

% If N is stored implicitly, to potentially improve efficiency, code is
% selected based on the number of rows and columns in A and N.

if nargin < 3
    method = 0 ;
end
[m n] = size (A) ;

if isstruct (N)

    %---------------------------------------------------------------------------
    % N held implicitly
    %---------------------------------------------------------------------------

    p = size (N.X, 2) ;

    switch method

        case 0

            %-------------------------------------------------------------------
            % B = N'*A
            %-------------------------------------------------------------------

            switch N.kind

                case 'Q*X'

                    if n <= p
                        % B = X' * ( Q' * A )
                        B = spqr_qmult (N.Q, A, 0) ;
                        B = N.X'*B ;
                    else
                        % B = ( X' * Q' ) * A
                        B = spqr_qmult (N.Q, (N.X)', 2) ;
                        B = B*A ;
                    end

                case 'Q*P*X'

                    if n <= p
                        % B = X' * P' * ( Q' * A ) ;
                        B = spqr_qmult (N.Q,A,0) ;
                        p (N.P) = 1:length (N.P) ;
                        B = B(p,:) ;
                        B = N.X'*B ;
                    else
                        % B = ( ( P * X )' * Q' ) * A ;
                        B = N.X(N.P,:)';
                        B = spqr_qmult (N.Q,B,2) ;
                        B = B * A ;
                    end

                case 'P*Q*X'

                    if n <= p
                        % B = X' * ( Q' * ( P' * A ) ) ;
                        p( N.P) = 1:length(N.P) ;
                        B = A(p,:) ;
                        B = spqr_qmult (N.Q,B,0) ;
                        B = N.X'*B ;
                    else
                        % B = ( ( X' * Q' ) * P' ) * A ;
                        B = spqr_qmult (N.Q, (N.X)', 2) ;
                        B = B(:,N.P) ;
                        B = B * A ;
                    end

                otherwise
                    error ('unrecognized N struct') ;
            end

        case 1

            %-------------------------------------------------------------------
            % B = N*A
            %-------------------------------------------------------------------

            switch N.kind

                case 'Q*X'

                    % B = Q * ( X * A ) ;
                    B = N.X * A ;
                    B = spqr_qmult (N.Q,B,1) ;

                case 'Q*P*X'

                    % B = Q * ( P * ( X * A ) ) ;
                    B = N.X * A ;
                    B = B(N.P,:) ;
                    B = spqr_qmult (N.Q,B,1) ;

                case 'P*Q*X'

                    % B = P * ( Q * ( X * A ) ) ;
                    B = N.X * A ;
                    B = spqr_qmult (N.Q,B,1) ;
                    B = B(N.P,:) ;

                otherwise
                    error ('unrecognized N struct') ;
            end

        case 2

            %-------------------------------------------------------------------
            % B = A*N'
            %-------------------------------------------------------------------

            switch N.kind

                case 'Q*X'

                    % B = (A * X') * Q'
                    B = A * (N.X)' ;
                    B = spqr_qmult (N.Q,B,2) ;

                case 'Q*P*X'

                    % B = ( ( A * X' ) * P' ) * Q'
                    B = A * (N.X)' ;
                    B = B(:,N.P) ;
                    B = spqr_qmult (N.Q,B,2) ;

                case 'P*Q*X'

                    % B = ( ( A * X' ) * Q' ) * P'
                    B = A * (N.X)' ;
                    B = spqr_qmult (N.Q,B,2) ;
                    B = B(:,N.P) ;

                otherwise
                    error ('unrecognized N struct') ;
            end

        case 3

            %-------------------------------------------------------------------
            % B = A*N
            %-------------------------------------------------------------------

            switch N.kind

                case 'Q*X'

                    if m <= p
                        % B = ( A * Q ) * X
                        B = spqr_qmult (N.Q,A,3) ;
                        B = B*N.X ;
                    else
                        % B = A * ( Q * X )
                        B = spqr_qmult (N.Q, N.X, 1 ) ;
                        B = A * B ;
                    end

                case 'Q*P*X'

                    if m <= p
                        % B = ( ( A * Q ) * P ) * X
                        B = spqr_qmult (N.Q,A,3) ;
                        p(N.P) = 1:length(N.P) ;
                        B = B(:,p) ;
                        B = B*N.X ;
                    else
                        % B = A * ( Q * ( P * X ) )
                        B = N.X(N.P,:) ;
                        B = spqr_qmult (N.Q, B, 1 ) ;
                        B = A * B ;
                    end

                case 'P*Q*X'

                    if m <= p
                        % B = ( ( A * P ) * Q ) * X
                        p(N.P) = 1:length(N.P) ;
                        B = A(:,p) ;
                        B = spqr_qmult (N.Q,B,3) ;
                        B = B*N.X ;
                    else
                        % B = A * ( P * ( Q * X ) )
                        B = spqr_qmult (N.Q, N.X, 1) ;
                        B = B(N.P,:) ;
                        B = A * B ;
                    end

                otherwise
                    error ('unrecognized N struct') ;
            end

        otherwise
            error ('unrecognized method') ;
    end

else

    %---------------------------------------------------------------------------
    % N held as an explicit matrix
    %---------------------------------------------------------------------------

    switch method
        case 0
            B = N'*A ;
        case 1
            B = N*A ;
        case 2
            B = A*N' ;
        case 3
            B = A*N ;
        otherwise
            error ('unrecognized method') ;
    end

end
