function b = gee_its_short (A, b)
%GEE_ITS_SHORT x=A\b, few comments, no error checking (just for line count)
% Example:
%   x = gee_its_short (A,b) ;       % same as x=A\b for square A
% See also: mldivide, gee_its_simple

% Copyright 2006-2007, Timothy A. Davis, http://www.suitesparse.com

n = size (A,1) ;
for k = 1:n
    [x i] = max (abs (A (k:n,k))) ;
    i = i+k-1 ;
    A ([k i],:) = A ([i k],:) ;
    b ([k i],:) = b ([i k],:) ;
    A (k+1:n,k) = A (k+1:n,k) / A (k,k) ;
    A (k+1:n,k+1:n) = A (k+1:n,k+1:n) - A (k+1:n,k) * A (k,k+1:n) ;
    b (k+1:n,:) = b (k+1:n,:) - A (k+1:n,k) * b (k,:) ;
end
for k = n:-1:1
    b (k,:) = b (k,:) / A (k,k) ;
    b (1:k-1,:) = b (1:k-1,:) - A (1:k-1,k) * b (k,:) ;
end
