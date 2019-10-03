% test the performance of the sparse subset Ax=b solver

clear
rng ('default')
index = UFget ;
f = find (index.posdef & (index.amd_lnz > 0)) ;
f = setdiff (f, [1425 228 353 354]) ; % not really posdef
[ignore i] = sort (index.amd_lnz (f)) ;
f = f (i) ;
nmat = length (f)

figure (1) ; clf ; hold off
figure (2) ; clf ; hold off
maxresid = 0 ;

for k = 1:nmat
    id = f (k) ;
    Prob = UFget (id, index)
    A = Prob.A ;
    n = size (A,1) ;

    if (exist ('timelog.m', 'file'))
        delete ('timelog.m') ;
    end
    clear results
    mwrite ('/tmp/A.mtx', A) ;
    system ('./cholmod_l_demo < /tmp/A.mtx > /tmp/output') ;

    % get the results = [i, xlen, flops, time]
    % also get lnz and t = dense solve time
    timelog

    if (isreal (A))
        fl = 2 * n + 4 * lnz ;
    else
        fl = 16 * n + 16 * lnz ;
    end

    figure (1) 
    clf
    hold off
    loglog (results (:,3), results (:,4), 'o', fl, t, 'ro') ;
    xlabel ('flops') ;
    ylabel ('time') ;

    figure (2) 
    loglog (results (:,3), results (:,4), 'o', fl, t, 'ro') ;
    xlabel ('flops') ;
    ylabel ('time') ;
    hold on

    maxresid = max (maxresid, resid) ;
    fprintf ('resid %g maxresid %g\n', resid, maxresid) ;
    drawnow

end
