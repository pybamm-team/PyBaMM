function kinds = UFkinds
%UFKINDS get 'kind' of each problem in the SuiteSparse Matrix Collection.
% UFkinds is deprecated; use sskinds instead

warning ('UFget:deprecated', 'UFget is deprecated; use ssget instead') ;
kinds = sskinds ;

