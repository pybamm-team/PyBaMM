function [L, w] = chol_updown2 (L, sigma, w)
%CHOL_UPDOWN2 Cholesky update/downdate (real and complex)
% (real or complex)
% Example:
%   [L, w] = chol_updown2 (L, sigma, w)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

beta = 1 ;
n = size (L,1) ;
if (n == 1)
    wnew = L\w ;
    L = sqrt (L*L'+sigma*w*w') ;
    w = wnew ;
    return ;
end
for k = 1:n
    alpha = w(k) / L(k,k) ;
    beta2 = sqrt (beta*beta + sigma*alpha*conj(alpha)) ;
    gamma = sigma * conj(alpha) / (beta2 * beta) ;
    if (sigma > 0)
        % update
        delta = beta / beta2 ;
        L (k,k) = delta * L (k,k) + gamma * w (k) ;
        phase = abs (L (k, k))/L (k, k) ;
        L (k, k) = phase*L (k, k) ;

        w1 = w (k+1:n) ;
        w (k+1:n) = w (k+1:n) - alpha * L (k+1:n,k) ;
        L (k+1:n,k) = phase * (delta * L (k+1:n,k) + gamma * w1) ;

    else
        % downdate
        delta = beta2 / beta ;
        L (k,k) = delta * L (k,k) ;
        phase = abs (L (k, k))/L (k, k) ;
        L (k, k) = phase*L (k, k) ;

        w (k+1:n) = w (k+1:n) - alpha * L (k+1:n,k) ;
        L (k+1:n,k) = phase * (delta * L (k+1:n,k) + gamma * w (k+1:n)) ;

    end
    w (k) = alpha ;
    beta = beta2 ;
end
