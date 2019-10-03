function btf_test (nmat)
%BTF_TEST test for BTF
% Requires CSparse (or CXSparse) and UFget
% Example:
%   btf_test
% See also btf, maxtrans, strongcomp, dmperm, UFget,
%   test1, test2, test3, test4, test5, test6.

if (nargin < 1)
    nmat = 200 ;
end

test1 (nmat) ;
test2 (nmat) ;
test3 (nmat) ;
test4 (nmat) ;
test5 (nmat) ;
test6 ;

