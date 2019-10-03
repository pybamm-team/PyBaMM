#include "cholmod.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <signal.h>

#define Size_max ((size_t) (-1))

/* -------------------------------------------------------------------------- */
/* double, SuiteSparse_long */
/* -------------------------------------------------------------------------- */

#ifdef DLONG
#define Real double
#define Int SuiteSparse_long
#define Int_max SuiteSparse_long_max
#define CHOLMOD(name) cholmod_l_ ## name
#define LONG
#define DOUBLE
#define ITYPE CHOLMOD_LONG
#define DTYPE CHOLMOD_DOUBLE

/* -------------------------------------------------------------------------- */
/* double, int: this is the default */
/* -------------------------------------------------------------------------- */

#else

#ifndef DINT
#define DINT
#endif
#define INT
#define DOUBLE

#define Real double
#define Int int
#define Int_max INT_MAX
#define CHOLMOD(name) cholmod_ ## name
#define ITYPE CHOLMOD_INT
#define DTYPE CHOLMOD_DOUBLE

#endif

/* -------------------------------------------------------------------------- */

#define BLAS_OK cm->blas_ok
#include "cholmod_blas.h"

#define EMPTY (-1)
#define TRUE 1
#define FALSE 0
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define IMPLIES(p,q) (!(p) || (q))

#if defined(WIN32) || defined(_WIN32)
#define ISNAN(x) (((x) != (x)) || (((x) < (x))))
#else
#define ISNAN(x) ((x) != (x))
#endif

#define MY_RAND_MAX 32767

#define MAXERR(maxerr,err,anorm) \
{ \
    if (ISNAN (maxerr)) \
    { \
	/* do nothing */ ; \
    } \
    else if (ISNAN (err)) \
    { \
	maxerr = err ; \
    } \
    else if (anorm > 0) \
    { \
	if ((err/anorm) > maxerr) maxerr = (err/anorm) ; \
    } \
    else \
    { \
	if (err > maxerr) maxerr = err ; \
    } \
    /* printf ("MAXERR: %7.2e %7.2e %7.2e  in %d : %s\n", \
	    maxerr, err, (double) anorm, __LINE__, __FILE__ ) ; */ \
}

#define OKP(p) Assert ((p) != NULL, __FILE__, __LINE__)
#define OK(truth) Assert (truth, __FILE__, __LINE__)
#define NOT(truth) Assert (!(truth), __FILE__, __LINE__)
#define NOP(p) Assert ((p) == NULL, __FILE__, __LINE__)

#define NSMALL 200
#define NLARGE 1000

#define ERROR(status,message) \
    CHOLMOD(error) (status, __FILE__, __LINE__, message, cm)

/* -------------------------------------------------------------------------- */
/* global variables */
/* -------------------------------------------------------------------------- */

#ifndef EXTERN
#define EXTERN extern
#endif

EXTERN double zero [2], one [2], minusone [2] ;
EXTERN cholmod_common Common, *cm ;
EXTERN cholmod_dense *M1 ;
EXTERN Int my_tries ;
EXTERN double Zero [2] ;

/* -------------------------------------------------------------------------- */
/* prototypes */
/* -------------------------------------------------------------------------- */

void null_test (cholmod_common *) ;
void null_test2 (void) ;
void Assert (int truth, char *file, int line) ;
Int nrand (Int n) ;
Int *prand (Int n) ;
cholmod_triplet *read_triplet (FILE *f) ;
double test_ops (cholmod_sparse *A) ;
cholmod_dense *xtrue (Int nrow, Int ncol, Int d, Int xtype) ;
double resid (cholmod_sparse *A, cholmod_dense *X, cholmod_dense *B) ;
double solve (cholmod_sparse *A) ;
double aug (cholmod_sparse *A) ;
double do_matrix (cholmod_sparse *A) ;
cholmod_dense *rhs (cholmod_sparse *A, Int nrhs, Int d) ;
void prune_row (cholmod_sparse *A, Int k) ;
double pnorm (cholmod_dense *X, Int *P, cholmod_dense *B, Int inv) ;
double test_solver (cholmod_sparse *A) ;
Int *rand_set (Int len, Int n) ;
void my_handler  (int status, const char *file, int line, const char *msg) ;
void my_handler2 (int status, const char *file, int line, const char *msg) ;
double resid3 (cholmod_sparse *A1, cholmod_sparse *A2, cholmod_sparse *A3,
    cholmod_dense *X, cholmod_dense *B) ;
double xrand (double range) ;
double lp_resid (cholmod_sparse *A, Int *rflag, Int *fset, Int fsize,
    double beta [2], cholmod_dense *X, cholmod_dense *B) ;
double lpdemo (cholmod_triplet *T) ;
cholmod_sparse *lp_prune ( cholmod_sparse *A, Int *rflag, Int *fset, Int fsize);
void null2 (cholmod_triplet *Tok, int do_nantests) ;
void *my_malloc2 (size_t size) ;
void *my_calloc2 (size_t n, size_t size) ;
void *my_realloc2 (void *p, size_t size) ;
void my_free2 (void *p) ;
void memory_tests (cholmod_triplet *T) ;
void progress (Int force, char s) ;
void test_memory_handler ( void ) ;
void normal_memory_handler ( void ) ;
cholmod_sparse *unpack (cholmod_sparse *A) ;
Int nzdiag (cholmod_sparse *A) ;
Int check_partition (cholmod_sparse *A, Int *Part) ;
double raw_factor (cholmod_sparse *A, Int errors) ;
double raw_factor2 (cholmod_sparse *A, double alpha, int domask) ;
cholmod_sparse *get_row (cholmod_sparse *A, Int i, Int *rflag, Int *fset,
    Int fsize, double beta [2]) ;
Int my_rand (void) ;
void my_srand (unsigned seed) ;
unsigned long my_seed (void) ;
void cctest (cholmod_sparse *A) ;
Int check_constraints (Int *P, Int *Cmember, Int n) ;
void ctest (cholmod_sparse *A) ;
void amdtest (cholmod_sparse *A) ;
double resid_sparse (cholmod_sparse *A, cholmod_sparse *X, cholmod_sparse *B) ;
cholmod_dense *zeros (Int nrow, Int ncol, Int d, Int xtype) ;

/* -------------------------------------------------------------------------- */
/* AMD, COLAMD, and CCOLAMD */
/* -------------------------------------------------------------------------- */

#ifdef LONG

#define ID "%ld"

#define AMD_order amd_l_order
#define AMD_defaults amd_l_defaults
#define AMD_control amd_l_control
#define AMD_info amd_l_info
#define AMD_1 amd_l1
#define AMD_2 amd_l2
#define AMD_valid amd_l_valid
#define AMD_aat amd_l_aat
#define AMD_postorder amd_l_postorder
#define AMD_post_tree amd_l_post_tree
#define AMD_dump amd_l_dump
#define AMD_debug amd_l_debug
#define AMD_debug_init amd_l_debug_init
#define AMD_preprocess amd_l_preprocess

#define CAMD_order camd_l_order
#define CAMD_defaults camd_l_defaults
#define CAMD_control camd_l_control
#define CAMD_info camd_l_info
#define CAMD_1 camd_l1
#define CAMD_2 camd_l2
#define CAMD_valid camd_l_valid
#define CAMD_cvalid camd_l_cvalid
#define CAMD_aat camd_l_aat
#define CAMD_postorder camd_l_postorder
#define CAMD_dump camd_l_dump
#define CAMD_debug camd_l_debug
#define CAMD_debug_init camd_l_debug_init
#define CAMD_preprocess camd_l_preprocess

#define CCOLAMD_recommended ccolamd_l_recommended
#define CCOLAMD_set_defaults ccolamd_l_set_defaults
#define CCOLAMD_2 ccolamd2_l
#define CCOLAMD_MAIN ccolamd_l
#define CCOLAMD_apply_order ccolamd_l_apply_order
#define CCOLAMD_postorder ccolamd_l_postorder
#define CCOLAMD_post_tree ccolamd_l_post_tree
#define CCOLAMD_fsize ccolamd_l_fsize
#define CSYMAMD_MAIN csymamd_l
#define CCOLAMD_report ccolamd_l_report
#define CSYMAMD_report csymamd_l_report

#define COLAMD_recommended colamd_l_recommended
#define COLAMD_set_defaults colamd_l_set_defaults
#define COLAMD_MAIN colamd_l
#define SYMAMD_MAIN symamd_l
#define COLAMD_report colamd_l_report
#define SYMAMD_report symamd_l_report

#else

#define ID "%d"

#define AMD_order amd_order
#define AMD_defaults amd_defaults
#define AMD_control amd_control
#define AMD_info amd_info
#define AMD_1 amd_1
#define AMD_2 amd_2
#define AMD_valid amd_valid
#define AMD_aat amd_aat
#define AMD_postorder amd_postorder
#define AMD_post_tree amd_post_tree
#define AMD_dump amd_dump
#define AMD_debug amd_debug
#define AMD_debug_init amd_debug_init
#define AMD_preprocess amd_preprocess

#define CAMD_order camd_order
#define CAMD_defaults camd_defaults
#define CAMD_control camd_control
#define CAMD_info camd_info
#define CAMD_1 camd_1
#define CAMD_2 camd_2
#define CAMD_valid camd_valid
#define CAMD_cvalid camd_cvalid
#define CAMD_aat camd_aat
#define CAMD_postorder camd_postorder
#define CAMD_dump camd_dump
#define CAMD_debug camd_debug
#define CAMD_debug_init camd_debug_init
#define CAMD_preprocess camd_preprocess

#define CCOLAMD_recommended ccolamd_recommended
#define CCOLAMD_set_defaults ccolamd_set_defaults
#define CCOLAMD_2 ccolamd2
#define CCOLAMD_MAIN ccolamd
#define CCOLAMD_apply_order ccolamd_apply_order
#define CCOLAMD_postorder ccolamd_postorder
#define CCOLAMD_post_tree ccolamd_post_tree
#define CCOLAMD_fsize ccolamd_fsize
#define CSYMAMD_MAIN csymamd
#define CCOLAMD_report ccolamd_report
#define CSYMAMD_report csymamd_report

#define COLAMD_recommended colamd_recommended
#define COLAMD_set_defaults colamd_set_defaults
#define COLAMD_MAIN colamd
#define SYMAMD_MAIN symamd
#define COLAMD_report colamd_report
#define SYMAMD_report symamd_report

#endif
