function testRB3
% Exhaustive test of RBio.  Requires UFget.  This is a very extensive test;
% it tests all matrices in the UF collection, and runs until it runs out of
% memory.

index = UFget ;
[ignore f] = sort (index.nnz) ;
title = 'test file' ;
key = 'key' ;

for k = 1:length (f)

    clear id Prob A Z A2 Z2 mtype title2 key2 mtype2 name m n

    try
        id = f (k) ;
        Prob = UFget (id, index) ;
        A = Prob.A ;
        [m n] = size (A) ;
        if (isfield (Prob, 'Z'))
            Z = Prob.Zeros ;
        else
            Z = sparse (m,n) ;
        end
        name = Prob.name ;
        clear Prob ;

        % check the matrix type
        [mtype2 mkind skind] = RBtype (A+Z) ;
        mtype = index.RBtype (id,:) ;
        fprintf ('%4d : %4d %s %s : %s\n', k, id, mtype, mtype2, name) ;
        if (any (mtype ~= mtype2))
            error ('mtype!') ;
        end

        % write the matrix out
        mtype2 = RBwrite ('temp.rb', A, Z, title, key) ;
        if (any (mtype ~= mtype2))
            fprintf ('%s : %s\n', mtype, mtype2) ;
            error ('mtype RBwrite!') ;
        end

        % read it back it
        [A2 Z2 title2 key2 mtype2] = RBread ('temp.rb') ;
        if (any (title ~= title2) || any (key ~= key2) || any (mtype ~= mtype2))
            fprintf ('%s : %s\n', title, title2) ;
            fprintf ('%s : %s\n', key, key2) ;
            fprintf ('%s : %s\n', mtype, mtype2) ;
            error ('RBread!') ;
        end
        if (nnz (A-A2) > 0)
            error ('matrix A mismatch!') ;
        end
        if (nnz (Z-Z2) > 0)
            error ('matrix Z mismatch!') ;
        end

    catch
        disp (lasterr) ;
        break
    end
end

delete temp.rb
