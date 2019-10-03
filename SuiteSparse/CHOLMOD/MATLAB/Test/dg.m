function dg(A)
%DG order and plot A*A', using CHOLMOD's nested dissection
% used by test27.m
% Example:
%   dg(A)
% See also cholmod_test

% Copyright 2007, Timothy A. Davis, http://www.suitesparse.com

A = spones (A) ;
[p cp cm] = nesdis (A, 'row') ;

% get the corresponding column ordering.  Order the columns
% in increasing order of min(find(A(:,j)))

[m n] = size (A) ;
C = A (p,:) ;
qmin = zeros (1,n) ;
for j = 1:n
    qmin (j) = min (find (C (:,j))) ;		%#ok
end
[ignore q] = sort (qmin) ;

C = C (:,q) ;
clf
subplot (2,3,1) ; treeplot (cp)
drawnow
subplot (2,3,2) ; spy (C)
drawnow
% axis off
subplot (2,3,3) ; spy (C*C')
drawnow
% axis off

ncomp = max(cm) ;
fprintf ('# of components: %d\n', ncomp)

% cs = [cm(p) n+1] ;
% cboundaries = find (diff (cs)) ;

fprintf ('size of root %d out of %d rows\n', length (find (cm == ncomp)), m);

[cnt h pa po R] = symbfact2 (A (p,:), 'row') ;
% rc = full (sum (R)) ;

for k = 1:ncomp
    fprintf ('node %4d : parent %4d size %6d work %g\n', ...
    k, cp (k), length (find (cm == k)), sum (cnt (find (cm == k)).^2) ) ;   %#ok
end

subplot (2,3,4) ; spy (A*A') ;
drawnow

subplot (2,3,5) ; spy (R+R') ;
drawnow

pamd = amd2 (A*A') ;        % use AMD from SuiteSparse, not built-in
[cnt h pa po R] = symbfact2 (A (pamd,:), 'row') ;
subplot (2,3,6) ; spy (R+R') ;
drawnow

% s = bisect (A, 'row') ;
% [ignore pp] = sort (s) ;
% E = A(pp,:) ;
% subplot (2,3,4) ; spy (E*E')
% fprintf ('bisect: %d\n', length (find (s == 2))) ;

% spy (C*C')
% hold on
% for j = cboundaries
%     plot ([1 n], [j j], 'r', [j j], [1 n], 'r') ;
% end    


