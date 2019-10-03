function [S name kind Problem] = dimacs10 (matrix)
%DIMACS10 returns a graph from the DIMACS10 test set, via the SuiteSparse Collection.
% The original graphs are at http://www.cc.gatech.edu/dimacs10 ; this function
% downloads them from the SuiteSparse Matrix Collection, converting them as
% needed.
%
%   [S name kind Problem] = dimacs10 (matrix) ;
%   index = dimacs10 ;
%
% matrix: either an integer in the range 1:212 (the number of graphs in the
%       DIMACS10 collection), or a string with the name of a DIMACS10 graph.
%
% S: adjacency matrix of the DIMACS10 graph.
% name: DIMACS10 name of the graph.  The name of the graph in
%       the SuiteSparse Matrix Collection is Problem.name.
% kind: a string describing the DIMACS10 graph:
%       'undirected multigraph'.  S(i,j) is the # of edges (i,j).  Only a
%           multigraph can have self-edges (diag (S)).
%       'undirected weighted graph'.  S(i,j) is the weight of edge (i,j).
%       'undirected graph'.  S is binary.
% Problem:  a struct containing the source of the DIMACS10 graph, containing
%       the matrix Problem.A.  The graph S is always symmetric, but Problem.A
%       might be unsymmetric.  This function performs all the necessary
%       conversions to obtain the DIMACS10 graph.
%
% If the nodes of S have xy or xyz coordinates, they are returned in
% Problem.aux.coord as an n-by-2 or n-by-3 matrix, where n = size(S,1).
% Otherwise, Problem.aux will not be present in the Problem struct.
% Nodeweights are present in Problem.aux, as nodevalue or (for the
% redistricting problems) population and area.
%
% With no input arguments, an index of the DIMACS10 graphs is returned.
%
% Example
%
%   % read in the same graph in two different ways
%   S = dimacs10 (1)
%   S = dimacs10 ('adjnoun')
%
%   % read in all 212 graphs, in increasing order of nnz(S)
%   index = dimacs10 ;
%   [~,list] = sort (index.nnz) ;
%   for id = list'
%       [S name kind] = dimacs10 (id) ;
%       fprintf ('%s : %s : n: %d nnz %d\n', name, kind, size(S,1), nnz(S)) ;
%       spy (S) ;
%       drawnow
%   end
%
% To write the resulting DIMACS10 graph to a Matrix Market or Rutherford/Boeing
% file for use outside of MATLAB, see sswrite in the
% SuiteSparse/SuiteSparseCollection toolbox.
%
% See also gallery, ssget.

% Copyright 2012, Timothy A. Davis, http://www.suitesparse.com

% matrices 139 to 212 added in July 2012
% not yet in the collection (too large): clustering/uk-2007-05

% DIMACS graph                          SuiteSparse Matrix
%                                       (empty if in DIMACS10/ group)
graphs = {
'clustering/adjnoun',                   'Newman/adjnoun'
'clustering/as-22july06',               'Newman/as-22july06'
'clustering/astro-ph',                  'Newman/astro-ph'
'clustering/caidaRouterLevel',          [ ]
'clustering/celegans_metabolic',        'Arenas/celegans_metabolic'
'clustering/celegansneural',            'Newman/celegansneural'
'clustering/chesapeake',                [ ]
'clustering/cnr-2000',                  'LAW/cnr-2000'
'clustering/cond-mat-2003',             'Newman/cond-mat-2003'
'clustering/cond-mat-2005',             'Newman/cond-mat-2005'
'clustering/cond-mat',                  'Newman/cond-mat'
'clustering/dolphins',                  'Newman/dolphins'
'clustering/email',                     'Arenas/email'
'clustering/eu-2005',                   'LAW/eu-2005'
'clustering/football',                  'Newman/football'
'clustering/hep-th',                    'Newman/hep-th'
'clustering/in-2004',                   'LAW/in-2004'
'clustering/jazz',                      'Arenas/jazz'
'clustering/karate',                    'Newman/karate'
'clustering/lesmis',                    'Newman/lesmis'
'clustering/netscience',                'Newman/netscience'
'clustering/PGPgiantcompo',             'Arenas/PGPgiantcompo'
'clustering/polblogs',                  'Newman/polblogs'
'clustering/polbooks',                  'Newman/polbooks'
'clustering/power',                     'Newman/power'
'clustering/road_central',              [ ]
'clustering/road_usa',                  [ ]
'coauthor/citationCiteseer',            [ ]
'coauthor/coAuthorsCiteseer',           [ ]
'coauthor/coAuthorsDBLP',               [ ]
'coauthor/coPapersCiteseer',            [ ]
'coauthor/coPapersDBLP',                [ ]
'delaunay/delaunay_n10',                [ ]
'delaunay/delaunay_n11',                [ ]
'delaunay/delaunay_n12',                [ ]
'delaunay/delaunay_n13',                [ ]
'delaunay/delaunay_n14',                [ ]
'delaunay/delaunay_n15',                [ ]
'delaunay/delaunay_n16',                [ ]
'delaunay/delaunay_n17',                [ ]
'delaunay/delaunay_n18',                [ ]
'delaunay/delaunay_n19',                [ ]
'delaunay/delaunay_n20',                [ ]
'delaunay/delaunay_n21',                [ ]
'delaunay/delaunay_n22',                [ ]
'delaunay/delaunay_n23',                [ ]
'delaunay/delaunay_n24',                [ ]
'dyn-frames/hugebubbles-00000',         [ ]
'dyn-frames/hugebubbles-00010',         [ ]
'dyn-frames/hugebubbles-00020',         [ ]
'dyn-frames/hugetrace-00000',           [ ]
'dyn-frames/hugetrace-00010',           [ ]
'dyn-frames/hugetrace-00020',           [ ]
'dyn-frames/hugetric-00000',            [ ]
'dyn-frames/hugetric-00010',            [ ]
'dyn-frames/hugetric-00020',            [ ]
'kronecker/kron_g500-logn16',           [ ]
'kronecker/kron_g500-logn17',           [ ]
'kronecker/kron_g500-logn18',           [ ]
'kronecker/kron_g500-logn19',           [ ]
'kronecker/kron_g500-logn20',           [ ]
'kronecker/kron_g500-logn21',           [ ]
'kronecker/kron_g500-simple-logn16',    'DIMACS10/kron_g500-logn16'
'kronecker/kron_g500-simple-logn17',    'DIMACS10/kron_g500-logn17'
'kronecker/kron_g500-simple-logn18',    'DIMACS10/kron_g500-logn18'
'kronecker/kron_g500-simple-logn19',    'DIMACS10/kron_g500-logn19'
'kronecker/kron_g500-simple-logn20',    'DIMACS10/kron_g500-logn20'
'kronecker/kron_g500-simple-logn21',    'DIMACS10/kron_g500-logn21'
'matrix/af_shell10',                    'Schenk_AFE/af_shell10'
'matrix/af_shell9',                     'Schenk_AFE/af_shell9'
'matrix/audikw1',                       'GHS_psdef/audikw_1'
'matrix/cage15',                        'vanHeukelum/cage15'
'matrix/ecology1',                      'McRae/ecology1'
'matrix/ecology2',                      'McRae/ecology2'
'matrix/G3_circuit',                    'AMD/G3_circuit'
'matrix/kkt_power',                     'Zaoui/kkt_power'
'matrix/ldoor',                         'GHS_psdef/ldoor'
'matrix/nlpkkt120',                     'Schenk/nlpkkt120'
'matrix/nlpkkt160',                     'Schenk/nlpkkt160'
'matrix/nlpkkt200',                     'Schenk/nlpkkt200'
'matrix/nlpkkt240',                     'Schenk/nlpkkt240'
'matrix/thermal2',                      'Schmid/thermal2'
'numerical/adaptive',                   [ ]
'numerical/channel-500x100x100-b050',   [ ]
'numerical/packing-500x100x100-b050',   [ ]
'numerical/venturiLevel3',              [ ]
'random/rgg_n_2_15_s0',                 [ ]
'random/rgg_n_2_16_s0',                 [ ]
'random/rgg_n_2_17_s0',                 [ ]
'random/rgg_n_2_18_s0',                 [ ]
'random/rgg_n_2_19_s0',                 [ ]
'random/rgg_n_2_20_s0',                 [ ]
'random/rgg_n_2_21_s0',                 [ ]
'random/rgg_n_2_22_s0',                 [ ]
'random/rgg_n_2_23_s0',                 [ ]
'random/rgg_n_2_24_s0',                 [ ]
'streets/asia_osm',                     [ ]
'streets/belgium_osm',                  [ ]
'streets/europe_osm',                   [ ]
'streets/germany_osm',                  [ ]
'streets/great-britain_osm',            [ ]
'streets/italy_osm',                    [ ]
'streets/luxembourg_osm',               [ ]
'streets/netherlands_osm',              [ ]
'walshaw/144',                          [ ]
'walshaw/3elt',                         'AG-Monien/3elt'
'walshaw/4elt',                         'Pothen/barth5'
'walshaw/598a',                         [ ]
'walshaw/add20',                        'Hamm/add20'
'walshaw/add32',                        'Hamm/add32'
'walshaw/auto',                         [ ]
'walshaw/bcsstk29',                     'HB/bcsstk29'
'walshaw/bcsstk30',                     'HB/bcsstk30'
'walshaw/bcsstk31',                     'HB/bcsstk31'
'walshaw/bcsstk32',                     'HB/bcsstk32'
'walshaw/bcsstk33',                     'HB/bcsstk33'
'walshaw/brack2',                       'AG-Monien/brack2'
'walshaw/crack',                        'AG-Monien/crack'
'walshaw/cs4',                          [ ]
'walshaw/cti',                          [ ]
'walshaw/data',                         [ ]
'walshaw/fe_4elt2',                     [ ]
'walshaw/fe_body',                      [ ]
'walshaw/fe_ocean',                     [ ]
'walshaw/fe_pwt',                       'Pothen/pwt'
'walshaw/fe_rotor',                     [ ]
'walshaw/fe_sphere',                    [ ]
'walshaw/fe_tooth',                     [ ]
'walshaw/finan512',                     'Mulvey/finan512'
'walshaw/m14b',                         [ ]
'walshaw/memplus',                      'Hamm/memplus'
'walshaw/t60k',                         [ ]
'walshaw/uk',                           [ ]
'walshaw/vibrobox',                     'Cote/vibrobox'
'walshaw/wave',                         'AG-Monien/wave'
'walshaw/whitaker3',                    'AG-Monien/whitaker3'
'walshaw/wing',                         [ ]
'walshaw/wing_nodal',                   [ ]
'clustering/G_n_pin_pout',              [ ]
'clustering/preferentialAttachment',    [ ]
'clustering/smallworld',                [ ]
'clustering/uk-2002',                   'LAW/uk-2002'
'numerical/333SP',                      [ ]
'numerical/AS365',                      [ ]
'numerical/M6',                         [ ]
'numerical/NACA0015',                   [ ]
'numerical/NLR',                        [ ]
'redistrict/ak2010',                    [ ]
'redistrict/al2010',                    [ ]
'redistrict/ar2010',                    [ ]
'redistrict/az2010',                    [ ]
'redistrict/ca2010',                    [ ]
'redistrict/co2010',                    [ ]
'redistrict/ct2010',                    [ ]
'redistrict/de2010',                    [ ]
'redistrict/fl2010',                    [ ]
'redistrict/ga2010',                    [ ]
'redistrict/hi2010',                    [ ]
'redistrict/ia2010',                    [ ]
'redistrict/id2010',                    [ ]
'redistrict/il2010',                    [ ]
'redistrict/in2010',                    [ ]
'redistrict/ks2010',                    [ ]
'redistrict/ky2010',                    [ ]
'redistrict/la2010',                    [ ]
'redistrict/ma2010',                    [ ]
'redistrict/md2010',                    [ ]
'redistrict/me2010',                    [ ]
'redistrict/mi2010',                    [ ]
'redistrict/mn2010',                    [ ]
'redistrict/mo2010',                    [ ]
'redistrict/ms2010',                    [ ]
'redistrict/mt2010',                    [ ]
'redistrict/nc2010',                    [ ]
'redistrict/nd2010',                    [ ]
'redistrict/ne2010',                    [ ]
'redistrict/nh2010',                    [ ]
'redistrict/nj2010',                    [ ]
'redistrict/nm2010',                    [ ]
'redistrict/nv2010',                    [ ]
'redistrict/ny2010',                    [ ]
'redistrict/oh2010',                    [ ]
'redistrict/ok2010',                    [ ]
'redistrict/or2010',                    [ ]
'redistrict/pa2010',                    [ ]
'redistrict/ri2010',                    [ ]
'redistrict/sc2010',                    [ ]
'redistrict/sd2010',                    [ ]
'redistrict/tn2010',                    [ ]
'redistrict/tx2010',                    [ ]
'redistrict/ut2010',                    [ ]
'redistrict/va2010',                    [ ]
'redistrict/vt2010',                    [ ]
'redistrict/wa2010',                    [ ]
'redistrict/wi2010',                    [ ]
'redistrict/wv2010',                    [ ]
'redistrict/wy2010',                    [ ]
'star-mixtures/vsp_barth5_1Ksep_50in_5Kout',        [ ]
'star-mixtures/vsp_bcsstk30_500sep_10in_1Kout',     [ ]
'star-mixtures/vsp_befref_fxm_2_4_air02',           [ ]
'star-mixtures/vsp_bump2_e18_aa01_model1_crew1',    [ ]
'star-mixtures/vsp_c-30_data_data',                 [ ]
'star-mixtures/vsp_c-60_data_cti_cs4',              [ ]
'star-mixtures/vsp_data_and_seymourl',              [ ]
'star-mixtures/vsp_finan512_scagr7-2c_rlfddd',      [ ]
'star-mixtures/vsp_mod2_pgp2_slptsk',               [ ]
'star-mixtures/vsp_model1_crew1_cr42_south31',      [ ]
'star-mixtures/vsp_msc10848_300sep_100in_1Kout',    [ ]
'star-mixtures/vsp_p0291_seymourl_iiasa',           [ ]
'star-mixtures/vsp_sctap1-2b_and_seymourl',         [ ]
'star-mixtures/vsp_south31_slptsk',                 [ ]
'star-mixtures/vsp_vibrobox_scagr7-2c_rlfddd',      [ ]
} ;

%-------------------------------------------------------------------------------
% return the index, if requested
%-------------------------------------------------------------------------------

multigraph = 57:62 ;
weightedgraph = [ 6 20 148:197 ] ;

ngraphs = size (graphs,1) ;
if (nargin == 0)

    n_nnz = [
        112        850   %   1 : clustering/adjnoun
      22963      96872   %   2 : clustering/as-22july06
      16706     242502   %   3 : clustering/astro-ph
     192244    1218132   %   4 : clustering/caidaRouterLevel
        453       4050   %   5 : clustering/celegans_metabolic
        297       4296   %   6 : clustering/celegansneural
         39        340   %   7 : clustering/chesapeake
     325557    5477938   %   8 : clustering/cnr-2000
      31163     240058   %   9 : clustering/cond-mat-2003
      40421     351382   %  10 : clustering/cond-mat-2005
      16726      95188   %  11 : clustering/cond-mat
         62        318   %  12 : clustering/dolphins
       1133      10902   %  13 : clustering/email
     862664   32276936   %  14 : clustering/eu-2005
        115       1226   %  15 : clustering/football
       8361      31502   %  16 : clustering/hep-th
    1382908   27182946   %  17 : clustering/in-2004
        198       5484   %  18 : clustering/jazz
         34        156   %  19 : clustering/karate
         77        508   %  20 : clustering/lesmis
       1589       5484   %  21 : clustering/netscience
      10680      48632   %  22 : clustering/PGPgiantcompo
       1490      33430   %  23 : clustering/polblogs
        105        882   %  24 : clustering/polbooks
       4941      13188   %  25 : clustering/power
   14081816   33866826   %  26 : clustering/road_central
   23947347   57708624   %  27 : clustering/road_usa
     268495    2313294   %  28 : coauthor/citationCiteseer
     227320    1628268   %  29 : coauthor/coAuthorsCiteseer
     299067    1955352   %  30 : coauthor/coAuthorsDBLP
     434102   32073440   %  31 : coauthor/coPapersCiteseer
     540486   30491458   %  32 : coauthor/coPapersDBLP
       1024       6112   %  33 : delaunay/delaunay_n10
       2048      12254   %  34 : delaunay/delaunay_n11
       4096      24528   %  35 : delaunay/delaunay_n12
       8192      49094   %  36 : delaunay/delaunay_n13
      16384      98244   %  37 : delaunay/delaunay_n14
      32768     196548   %  38 : delaunay/delaunay_n15
      65536     393150   %  39 : delaunay/delaunay_n16
     131072     786352   %  40 : delaunay/delaunay_n17
     262144    1572792   %  41 : delaunay/delaunay_n18
     524288    3145646   %  42 : delaunay/delaunay_n19
    1048576    6291372   %  43 : delaunay/delaunay_n20
    2097152   12582816   %  44 : delaunay/delaunay_n21
    4194304   25165738   %  45 : delaunay/delaunay_n22
    8388608   50331568   %  46 : delaunay/delaunay_n23
   16777216  100663202   %  47 : delaunay/delaunay_n24
   18318143   54940162   %  48 : dyn-frames/hugebubbles-00000
   19458087   58359528   %  49 : dyn-frames/hugebubbles-00010
   21198119   63580358   %  50 : dyn-frames/hugebubbles-00020
    4588484   13758266   %  51 : dyn-frames/hugetrace-00000
   12057441   36164358   %  52 : dyn-frames/hugetrace-00010
   16002413   47997626   %  53 : dyn-frames/hugetrace-00020
    5824554   17467046   %  54 : dyn-frames/hugetric-00000
    6592765   19771708   %  55 : dyn-frames/hugetric-00010
    7122792   21361554   %  56 : dyn-frames/hugetric-00020
      65536    4912469   %  57 : kronecker/kron_g500-logn16
     131072   10228360   %  58 : kronecker/kron_g500-logn17
     262144   21165908   %  59 : kronecker/kron_g500-logn18
     524288   43562265   %  60 : kronecker/kron_g500-logn19
    1048576   89239674   %  61 : kronecker/kron_g500-logn20
    2097152  182082942   %  62 : kronecker/kron_g500-logn21
      65536    4912142   %  63 : kronecker/kron_g500-simple-logn16
     131072   10227970   %  64 : kronecker/kron_g500-simple-logn17
     262144   21165372   %  65 : kronecker/kron_g500-simple-logn18
     524288   43561574   %  66 : kronecker/kron_g500-simple-logn19
    1048576   89238804   %  67 : kronecker/kron_g500-simple-logn20
    2097152  182081864   %  68 : kronecker/kron_g500-simple-logn21
    1508065   51164260   %  69 : matrix/af_shell10
     504855   17084020   %  70 : matrix/af_shell9
     943695   76708152   %  71 : matrix/audikw1
    5154859   94044692   %  72 : matrix/cage15
    1000000    3996000   %  73 : matrix/ecology1
     999999    3995992   %  74 : matrix/ecology2
    1585478    6075348   %  75 : matrix/G3_circuit
    2063494   12964640   %  76 : matrix/kkt_power
     952203   45570272   %  77 : matrix/ldoor
    3542400   93303392   %  78 : matrix/nlpkkt120
    8345600  221172512   %  79 : matrix/nlpkkt160
   16240000  431985632   %  80 : matrix/nlpkkt200
   27993600  746478752   %  81 : matrix/nlpkkt240
    1227087    7352268   %  82 : matrix/thermal2
    6815744   27248640   %  83 : numerical/adaptive
    4802000   85362744   %  84 : numerical/channel-500x100x100-b050
    2145852   34976486   %  85 : numerical/packing-500x100x100-b050
    4026819   16108474   %  86 : numerical/venturiLevel3
      32768     320480   %  87 : random/rgg_n_2_15_s0
      65536     684254   %  88 : random/rgg_n_2_16_s0
     131072    1457506   %  89 : random/rgg_n_2_17_s0
     262144    3094566   %  90 : random/rgg_n_2_18_s0
     524288    6539532   %  91 : random/rgg_n_2_19_s0
    1048576   13783240   %  92 : random/rgg_n_2_20_s0
    2097152   28975990   %  93 : random/rgg_n_2_21_s0
    4194304   60718396   %  94 : random/rgg_n_2_22_s0
    8388608  127002786   %  95 : random/rgg_n_2_23_s0
   16777216  265114400   %  96 : random/rgg_n_2_24_s0
   11950757   25423206   %  97 : streets/asia_osm
    1441295    3099940   %  98 : streets/belgium_osm
   50912018  108109320   %  99 : streets/europe_osm
   11548845   24738362   % 100 : streets/germany_osm
    7733822   16313034   % 101 : streets/great-britain_osm
    6686493   14027956   % 102 : streets/italy_osm
     114599     239332   % 103 : streets/luxembourg_osm
    2216688    4882476   % 104 : streets/netherlands_osm
     144649    2148786   % 105 : walshaw/144
       4720      27444   % 106 : walshaw/3elt
      15606      91756   % 107 : walshaw/4elt
     110971    1483868   % 108 : walshaw/598a
       2395      14924   % 109 : walshaw/add20
       4960      18924   % 110 : walshaw/add32
     448695    6629222   % 111 : walshaw/auto
      13992     605496   % 112 : walshaw/bcsstk29
      28924    2014568   % 113 : walshaw/bcsstk30
      35588    1145828   % 114 : walshaw/bcsstk31
      44609    1970092   % 115 : walshaw/bcsstk32
       8738     583166   % 116 : walshaw/bcsstk33
      62631     733118   % 117 : walshaw/brack2
      10240      60760   % 118 : walshaw/crack
      22499      87716   % 119 : walshaw/cs4
      16840      96464   % 120 : walshaw/cti
       2851      30186   % 121 : walshaw/data
      11143      65636   % 122 : walshaw/fe_4elt2
      45087     327468   % 123 : walshaw/fe_body
     143437     819186   % 124 : walshaw/fe_ocean
      36519     289588   % 125 : walshaw/fe_pwt
      99617    1324862   % 126 : walshaw/fe_rotor
      16386      98304   % 127 : walshaw/fe_sphere
      78136     905182   % 128 : walshaw/fe_tooth
      74752     522240   % 129 : walshaw/finan512
     214765    3358036   % 130 : walshaw/m14b
      17758     108392   % 131 : walshaw/memplus
      60005     178880   % 132 : walshaw/t60k
       4824      13674   % 133 : walshaw/uk
      12328     330500   % 134 : walshaw/vibrobox
     156317    2118662   % 135 : walshaw/wave
       9800      57978   % 136 : walshaw/whitaker3
      62032     243088   % 137 : walshaw/wing
      10937     150976   % 138 : walshaw/wing_nodal
     100000    1002396   % 139 : clustering/G_n_pin_pout
     100000     999970   % 140 : clustering/preferentialAttachment
     100000     999996   % 141 : clustering/smallworld
   18520486  523574516   % 142 : clustering/uk-2002
    3712815   22217266   % 143 : numerical/333SP
    3799275   22736152   % 144 : numerical/AS365
    3501776   21003872   % 145 : numerical/M6
    1039183    6229636   % 146 : numerical/NACA0015
    4163763   24975952   % 147 : numerical/NLR
      45292     217098   % 148 : redistrict/ak2010
     252266    1230482   % 149 : redistrict/al2010
     186211     904310   % 150 : redistrict/ar2010
     241666    1196094   % 151 : redistrict/az2010
     710145    3489366   % 152 : redistrict/ca2010
     201062     974574   % 153 : redistrict/co2010 
      67578     336352   % 154 : redistrict/ct2010 
      24115     116056   % 155 : redistrict/de2010 
     484481    2346294   % 156 : redistrict/fl2010 
     291086    1418056   % 157 : redistrict/ga2010 
      25016     124126   % 158 : redistrict/hi2010 
     216007    1021170   % 159 : redistrict/ia2010 
     149842     728264   % 160 : redistrict/id2010 
     451554    2164464   % 161 : redistrict/il2010 
     267071    1281716   % 162 : redistrict/in2010 
     238600    1121798   % 163 : redistrict/ks2010 
     161672     787778   % 164 : redistrict/ky2010 
     204447     980634   % 165 : redistrict/la2010 
     157508     776610   % 166 : redistrict/ma2010 
     145247     700378   % 167 : redistrict/md2010 
      69518     335476   % 168 : redistrict/me2010 
     329885    1578090   % 169 : redistrict/mi2010 
     259777    1227102   % 170 : redistrict/mn2010 
     343565    1656568   % 171 : redistrict/mo2010 
     171778     839980   % 172 : redistrict/ms2010 
     132288     638668   % 173 : redistrict/mt2010 
     288987    1416620   % 174 : redistrict/nc2010 
     133769     625946   % 175 : redistrict/nd2010 
     193352     913854   % 176 : redistrict/ne2010 
      48837     234550   % 177 : redistrict/nh2010 
     169588     829912   % 178 : redistrict/nj2010 
     168609     830970   % 179 : redistrict/nm2010 
      84538     416998   % 180 : redistrict/nv2010 
     350169    1709544   % 181 : redistrict/ny2010 
     365344    1768240   % 182 : redistrict/oh2010 
     269118    1274148   % 183 : redistrict/ok2010 
     196621     979512   % 184 : redistrict/or2010 
     421545    2058462   % 185 : redistrict/pa2010 
      25181     125750   % 186 : redistrict/ri2010 
     181908     893160   % 187 : redistrict/sc2010 
      88360     410722   % 188 : redistrict/sd2010 
     240116    1193966   % 189 : redistrict/tn2010 
     914231    4456272   % 190 : redistrict/tx2010 
     115406     572066   % 191 : redistrict/ut2010 
     285762    1402128   % 192 : redistrict/va2010 
      32580     155598   % 193 : redistrict/vt2010 
     195574     947432   % 194 : redistrict/wa2010 
     253096    1209404   % 195 : redistrict/wi2010 
     135218     662922   % 196 : redistrict/wv2010 
      86204     427586   % 197 : redistrict/wy2010
      32212     203610   % 198 : star-mixtures/vsp_barth5_1Ksep_50in_5Kout
      58348    4033156   % 199 : star-mixtures/vsp_bcsstk30_500sep_10in_1Kout
      14109     196448   % 200 : star-mixtures/vsp_befref_fxm_2_4_air02
      56438     601602   % 201 : star-mixtures/vsp_bump2_e18_aa01_model1_crew1
      11023     124368   % 202 : star-mixtures/vsp_c-30_data_data
      85830     482160   % 203 : star-mixtures/vsp_c-60_data_cti_cs4
       9167     111732   % 204 : star-mixtures/vsp_data_and_seymourl
     139752    1104040   % 205 : star-mixtures/vsp_finan512_scagr7-2c_rlfddd
     101364     778736   % 206 : star-mixtures/vsp_mod2_pgp2_slptsk
      45101     379952   % 207 : star-mixtures/vsp_model1_crew1_cr42_south31
      21996    2442056   % 208 : star-mixtures/vsp_msc10848_300sep_100in_1Kout
      10498     107736   % 209 : star-mixtures/vsp_p0291_seymourl_iiasa
      40174     281662   % 210 : star-mixtures/vsp_sctap1-2b_and_seymourl
      39668     379828   % 211 : star-mixtures/vsp_south31_slptsk
      77328     871172   % 212 : star-mixtures/vsp_vibrobox_scagr7-2c_rlfddd
    ] ;

    index.DIMACS10name = graphs (:,1) ;
    index.ssname = graphs (:,2) ;
    index.n = n_nnz (:,1) ;
    index.nnz = n_nnz (:,2) ;

    for id = 1:ngraphs
        name = graphs {id, 1} ;         % DIMACS10 name of the graph
        ssname = graphs {id, 2} ;       % its name in the ss Collection
        if (isempty (ssname))
            slash = find (name == '/') ;
            index.ssname {id} = ['DIMACS10/' (name (slash+1:end))] ;
        end
    end

    index.kind = cell (ngraphs,1) ;
    for id = 1:ngraphs
        index.kind {id} = 'undirected graph' ;
    end
    for id = multigraph
        index.kind {id} = 'undirected multigraph' ;
    end
    for id = weightedgraph
        index.kind {id} = 'undirected weighted graph' ;
    end

    S = index ;
    name = '' ;
    kind = '' ;
    Problem = [ ] ;
    return
end

%-------------------------------------------------------------------------------
% look up the matrix in the index
%-------------------------------------------------------------------------------

if (ischar (matrix))

    % S = dimacs10 ('clustering/astro-ph') or dimacs10 ('astro-ph').
    % find the matrix id.
    id = 0 ;
    noslash = isempty (strfind (matrix, '/')) ;
    for k = 1:ngraphs
        name = graphs {k,1} ;
        if (noslash)
            % S = dimacs10 ('astro0ph')
            slash = find (name == '/') ;
            name = name (slash+1:end) ;
        end
        if (isequal (name, matrix))
            % found it
            id = k ;
            break ;
        end
    end
    if (id == 0)
        error ('dimacs10:invalid', 'no such graph') ;
    end

else

    if (isscalar (matrix) && matrix >= 1 && matrix <= ngraphs)
        % S = dimacs10 (3) returns the clustering/astro-ph problem
        id = matrix ;
    else
        error ('dimacs10:invalid', 'input argument invalid') ;
    end

end

%-------------------------------------------------------------------------------
% get the matrix, converting it if necessary
%-------------------------------------------------------------------------------

name = graphs {id, 1} ;         % DIMACS10 name of the graph
ssname = graphs {id, 2} ;       % its name in the ss Collection

if (isempty (ssname))

    %---------------------------------------------------------------------------
    % the DIMACS10 graph is identical to the ss matrix
    %---------------------------------------------------------------------------

    slash = find (name == '/') ;
    ssname = ['DIMACS10/' (name (slash+1:end))] ;
    Problem = ssget (ssname) ;
    S = Problem.A ;

else

    %---------------------------------------------------------------------------
    % the DIMACS10 graph is derived from the ss matrix
    %---------------------------------------------------------------------------

    Problem = ssget (ssname) ;
    S = Problem.A ;

    addzeros   = [ 69:70 76 77 109 110 131 134 ] ;
    removediag = [ 5 8 14 17 23 63:82 107 109 110 112:116 125 129 131 134 142] ;
    makebinary = [ 3 5 9:11 15 16 18 21 22 63:82 109 110 129 131 134 ] ;
    symmetrize = [ 6 8 14 17 23 142 ] ;
    removenull = 82 ;

    if (any (id == weightedgraph))

        % make sure S is symmetric
        if (any (id == symmetrize))
            S = S + S' ;
        end

    else

        % add the explicit zeros
        if (any (id == addzeros))
            S = S + Problem.Zeros ;
        end

        % remove the diagonal, if present, and make binary
        if (any (id == removediag))
            S = dimacs10_convert_to_graph (S, any (id == makebinary)) ; 
        elseif (any (id == makebinary))
            S = spones (S) ;
        end

        % make sure S is symmetric and binary
        if (any (id == symmetrize))
            % this is required only for a few graphs
            S = spones (S + S') ;
        end

        % remove empty rows/columns
        if (any (id == removenull))
            nonnull = find (sum (S) > 0) ;
            S = S (nonnull, nonnull) ;
        end
    end
end

%-------------------------------------------------------------------------------
% determine the graph type
%-------------------------------------------------------------------------------

if (any (id == multigraph))
    % corresponds to format '100'
    kind = 'undirected multigraph' ;
elseif (any (id == weightedgraph))
    % corresponds to format '1', '11', or '10'
    kind = 'undirected weighted graph' ;
else
    % corresponds to format '0'
    kind = 'undirected graph' ;
end

