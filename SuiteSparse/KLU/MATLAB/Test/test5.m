function test5
%test5: KLU test
% Example:
%   test5
%
% test circuit matrices in the UF sparse matrix collection.
%
% See also klu

% Copyright 2004-2012, University of Florida

do_diary = 0 ;

if (do_diary)
    diary off
    s = date ;
    t = clock ;
    s = sprintf ('diary test5_%s_%d-%d-%d.txt\n', s, t (4), t(5), fix(t(6)));
    eval (s) ;
end

% ATandT frequency-domain circuits, exclude these:
% freq = [ 283     284     285     286      ] ;

% sorted in order of MATLAB 7.3 x=A\b time on storm
% (AMD Opteron, 64-bit, 8GB mem, 2 cores)
circ = [
1195    % Rajat/rajat11 n: 135 nz 665
1198    % Rajat/rajat14 n: 180 nz 1475
1189    % Rajat/rajat05 n: 301 nz 1250
1169    % Sandia/oscil_trans_01 n: 430 nz 1614
1112    % Sandia/oscil_dcop_01 n: 430 nz 1544
1346    % Rajat/rajat19 n: 1157 nz 3699
1199    % Hamrle/Hamrle1 n: 32 nz 98
1106    % Sandia/fpga_trans_01 n: 1220 nz 7382
1188    % Rajat/rajat04 n: 1041 nz 8725
1055    % Sandia/fpga_dcop_01 n: 1220 nz 5892
1108    % Sandia/init_adder1 n: 1813 nz 11156
1196    % Rajat/rajat12 n: 1879 nz 12818
1053    % Sandia/adder_trans_01 n: 1814 nz 14579
539     % Hamm/add20 n: 2395 nz 13151
371     % Bomhof/circuit_2 n: 4510 nz 21199
540     % Hamm/add32 n: 4960 nz 19848
1186    % Rajat/rajat02 n: 1960 nz 11187
466     % Grund/meg4 n: 5860 nz 25258
465     % Grund/meg1 n: 2904 nz 58142
370     % Bomhof/circuit_1 n: 2624 nz 35823
1200    % Hamrle/Hamrle2 n: 5952 nz 22162
1197    % Rajat/rajat13 n: 7598 nz 48762
1187    % Rajat/rajat03 n: 7602 nz 32653
372     % Bomhof/circuit_3 n: 12127 nz 48137
1185    % Rajat/rajat01 n: 6833 nz 43250
1183    % IBM_Austin/coupled n: 11341 nz 97193
1376    % Rajat/rajat27 n: 20640 nz 97353
543     % Hamm/memplus n: 17758 nz 99147
1109    % Sandia/mult_dcop_01 n: 25187 nz 193276
1371    % Rajat/rajat22 n: 39899 nz 195429
1375    % Rajat/rajat26 n: 51032 nz 247528
1414    % IBM_EDA/ckt11752_tr_0 n: 49702 nz 332807
541     % Hamm/bcircuit n: 68902 nz 375558
1413    % IBM_EDA/ckt11752_dc_1 n: 49702 nz 333029
542     % Hamm/hcircuit n: 105676 nz 513072
1316    % Rajat/rajat15 n: 37261 nz 443573
1190    % Rajat/rajat06 n: 10922 nz 46983
1191    % Rajat/rajat07 n: 14842 nz 63913
1372    % Rajat/rajat23 n: 110355 nz 555441
373     % Bomhof/circuit_4 n: 80209 nz 307604
544     % Hamm/scircuit n: 170998 nz 958936
1412    % AMD/G2_circuit n: 150102 nz 726674    is this solid state device?
1415    % Sandia/ASIC_100k n: 99340 nz 940621
1416    % Sandia/ASIC_100ks n: 99190 nz 578890
1420    % Sandia/ASIC_680ks n: 682712 nz 1693767
1192    % Rajat/rajat08 n: 19362 nz 83443
1193    % Rajat/rajat09 n: 24482 nz 105573
1323    % IBM_EDA/trans4 n: 116835 nz 749800
1320    % IBM_EDA/dc1 n: 116835 nz 766396
1194    % Rajat/rajat10 n: 30202 nz 130303
1418    % Sandia/ASIC_320ks n: 321671 nz 1316085
1417    % Sandia/ASIC_320k n: 321821 nz 1931828
1343    % Rajat/rajat16 n: 94294 nz 476766
1345    % Rajat/rajat18 n: 94294 nz 479151
1344    % Rajat/rajat17 n: 94294 nz 479246
1377    % Rajat/rajat28 n: 87190 nz 606489
1369    % Rajat/rajat20 n: 86916 nz 604299
1374    % Rajat/rajat25 n: 87190 nz 606489
1370    % Rajat/rajat21 n: 411676 nz 1876011
1419    % Sandia/ASIC_680k n: 682862 nz 2638997
% these are large, so are skipped for this test:
% 1396    % Rajat/rajat29 n: 643994 nz 3760246
% 1201    % Hamrle/Hamrle3 n: 1447360 nz 5514242
% 1397    % Rajat/rajat30 n: 643994 nz 6175244
% 1421    % AMD/G3_circuit n: 1585478 nz 7660826
% 1398    % Rajat/rajat31 n: 4690002 nz 20316253
% 1373    % Rajat/rajat24 n: 358172 nz 1946979
]' ;

fprintf ('Running KLU on %d circuits.\n', length (circ)) ;

index = UFget ;

opts_noscale.scale = -1 ;
opts_sum.scale = 1 ;
opts_max.scale = 2 ;            % default scaling

h = waitbar (0, 'KLU test 5 of 5') ;
nmat = length (circ) ;

try

    for kk = 1:nmat

        k = circ (kk) ;
        Prob = UFget (k, index) ;

        waitbar (kk/nmat, h) ;

        A = Prob.A ;
        n = size (A,1) ;
        b = rand (n,1) ;
        fprintf ('\n%d : %s n: %d nz %d\n', k, Prob.name, n, nnz (A)) ;

        try
            tic ;
            x2 = klu (A, '\', b, opts_noscale) ;
            t2 = toc ;
            e2 = norm (A*x2-b) ;
        catch                                                               %#ok
            t2 = inf ;
            e2 = inf ;
        end
        fprintf ('KLU no scale:  err %8.2e t: %8.4f\n', e2, t2) ;

        try
            tic ;
            x4 = klu (A, '\', b, opts_max) ;
            t4 = toc ;
            e4 = norm (A*x4-b) ;
        catch                                                               %#ok
            t4 = inf ;
            e4 = inf ;
        end
        fprintf ('KLU max scale: err %8.2e t: %8.4f\n', e4, t4) ;

        try
            tic ;
            x3 = klu (A, '\', b, opts_sum) ;
            t3 = toc ;
            e3 = norm (A*x3-b) ;
        catch                                                               %#ok
            t3 = inf ;
            e3 = inf ;
        end
        fprintf ('KLU sum scale: err %8.2e t: %8.4f\n', e3, t3) ;

        tic
        x1 = A\b ;
        t1 = toc ;
        e1 = norm (A*x1-b) ;
        fprintf ('matlab:        err %8.2e t: %8.4f\n', e1, t1) ;

        fprintf ('                                 speedup %8.2f\n', t1 / t4) ;
        clear Prob

        if (do_diary)
            diary off
            diary on
        end
    end

catch me
    disp (me.message) ;
end

close (h) ;
