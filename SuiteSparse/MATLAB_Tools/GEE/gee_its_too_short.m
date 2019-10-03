function b = gee_its_too_short (A, b)
%GEE_ITS_TOO_SHORT x=A\b, no pivoting (thus unstable!), just bare bones
% Example:
%   x = gee_its_too_short (A,b) ;       % x=A\b, with no pivoting (!)
% See also: mldivide, gee_its_simple, gee_its_short

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

n = size (A,1) ;
for k = 1:n
    A (k+1:n,k) = A (k+1:n,k) / A (k,k) ;
    A (k+1:n,k+1:n) = A (k+1:n,k+1:n) - A (k+1:n,k) * A (k,k+1:n) ;
    b (k+1:n,:) = b (k+1:n,:) - A (k+1:n,k) * b (k,:) ;
end
for k = n:-1:1
    b (k,:) = b (k,:) / A (k,k) ;
    b (1:k-1,:) = b (1:k-1,:) - A (1:k-1,k) * b (k,:) ;
end
