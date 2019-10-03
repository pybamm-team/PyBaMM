#!/bin/csh
# acov.di: construct gcov files for AMD, di version

gcov -o amd_i_1 amd_1.c
gcov -o amd_i_2 amd_2.c
gcov -o amd_i_aat amd_aat.c
gcov -o amd_i_control amd_control.c
gcov -o amd_i_defaults amd_defaults.c
gcov -o amd_i_info amd_info.c
gcov -o amd_i_order amd_order.c
gcov -o amd_i_post_tree amd_post_tree.c
gcov -o amd_i_postorder amd_postorder.c
gcov -o amd_i_preprocess amd_preprocess.c
gcov -o amd_i_valid amd_valid.c

