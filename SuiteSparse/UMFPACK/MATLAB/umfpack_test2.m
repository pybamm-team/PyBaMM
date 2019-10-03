function umfpack_test2
%UMFPACK_TEST2 try all UMFPACK strategies and orderings.
% Requires ssget, CHOLMOD, CAMD, CCOLAMD, COLAMD, METIS.

Prob = ssget (45) ;
A = Prob.A ;
n = size (A,1) ;
b = rand (n,1) ;

Strategy = {'auto', 'unsymmetric', 'symmetric'} ;
Ordering = { 'amd', 'default', 'metis', 'none', 'given' } ;
Scale = { 'none', 'sum', 'max' } ;

for k1 = 1:length (Strategy)
    for k2 = 1:length (Ordering)
        for k3 = 1:length (Scale)

            strategy = Strategy {k1} ;
            ordering = Ordering {k2} ;
            scale = Scale {k3} ;

            opts.strategy = strategy ;
            opts.ordering = ordering ;
            opts.scale = scale ;

            clear L U P Q R
            L = [ ] ;
            U = [ ] ;
            P = speye (n) ;
            Q = speye (n) ;

            try
                tic ;
                [L,U,P,Q,R] = umfpack (A, opts) ;
                t = toc ;
                lunz = nnz (L+U) ;
            catch
                disp (lasterr) ;
                t = inf ;
                lunz = 0 ;
            end

            fprintf ('%11s %7s %4s : time %10.4f lunz %10.3e million\n', ...
                strategy, ordering, scale, t, lunz / 1e6) ;

            subplot (1,2,1) ; cspy (P*A*Q) ;
            subplot (1,2,2) ; cspy (L+U) ;
            drawnow

        end
    end
end
