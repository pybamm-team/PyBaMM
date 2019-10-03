/* ========================================================================== */
/* === Tcov/memory ========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Extensive memory-failure testing for CHOLMOD.
 *
 * my_malloc2, my_calloc2, and my_realloc2 pretend to fail if my_tries goes to
 * zero, to test CHOLMOD's memory error handling.   No failure occurs if
 * my_tries is negative.
 */

#include "cm.h"


/* ========================================================================== */
/* === my_tries ============================================================= */
/* ========================================================================== */

Int my_tries = -1 ; /* a global variable */


/* ========================================================================== */
/* === my_malloc2 =========================================================== */
/* ========================================================================== */

void *my_malloc2 (size_t size)
{
    void *p ;
    if (my_tries == 0)
    {
	/* pretend to fail */
	/* printf ("p 0 (pretend to fail)\n") ; */
	return (NULL) ;
    }
    if (my_tries > 0)
    {
	my_tries-- ;
    }
    p = malloc (size) ;
    /* printf ("p %p\n", p) ; */
    return (p) ;
}


/* ========================================================================== */
/* === my_calloc2 =========================================================== */
/* ========================================================================== */

void *my_calloc2 (size_t n, size_t size)
{
    void *p ;
    if (my_tries == 0)
    {
	/* pretend to fail */
	/* printf ("p 0 (pretend to fail)\n") ; */
	return (NULL) ;
    }
    if (my_tries > 0)
    {
	my_tries-- ;
    }
    p = calloc (n, size) ;
    /* printf ("p %p\n", p) ; */
    return (p) ;
}


/* ========================================================================== */
/* === my_realloc2 ========================================================== */
/* ========================================================================== */

void *my_realloc2 (void *p, size_t size)
{
    void *p2 ;
    if (my_tries == 0)
    {
	/* pretend to fail */
	/* printf ("p2 0 (pretend to fail)\n") ; */
	return (NULL) ;
    }
    if (my_tries > 0)
    {
	my_tries-- ;
    }
    p2 = realloc (p, size) ;
    /* printf ("p2 %p\n", p2) ; */
    return (p2) ;
}


/* ========================================================================== */
/* === my_free2 ============================================================= */
/* ========================================================================== */

void my_free2 (void *p)
{
    free (p) ;
}


/* ========================================================================== */
/* === normal_memory_handler ================================================ */
/* ========================================================================== */

void normal_memory_handler ( void )
{

    SuiteSparse_config.malloc_func = malloc ;
    SuiteSparse_config.calloc_func = calloc ;
    SuiteSparse_config.realloc_func = realloc ;
    SuiteSparse_config.free_func = free ;

    cm->error_handler = my_handler ;
    CHOLMOD(free_work) (cm) ;
}


/* ========================================================================== */
/* === test_memory_handler ================================================== */
/* ========================================================================== */

void test_memory_handler ( void )
{
    SuiteSparse_config.malloc_func = my_malloc2 ;
    SuiteSparse_config.calloc_func = my_calloc2 ;
    SuiteSparse_config.realloc_func = my_realloc2 ;
    SuiteSparse_config.free_func = my_free2 ;

    cm->error_handler = NULL ;
    CHOLMOD(free_work) (cm) ;
    my_tries = 0 ;
}


/* ========================================================================== */
/* === memory tests ========================================================= */
/* ========================================================================== */

void memory_tests (cholmod_triplet *T)
{
    double err ;
    cholmod_sparse *A ;
    Int trial ;
    size_t count, inuse ;

    test_memory_handler ( ) ;
    inuse = cm->memory_inuse ;

    cm->nmethods = 8 ;
    cm->print = 0 ;
    cm->final_resymbol = TRUE ;

    cm->final_asis = FALSE ;
    cm->final_super = FALSE ;
    cm->final_ll = FALSE ;
    cm->final_pack = FALSE ;
    cm->final_monotonic = FALSE ;

    /* ---------------------------------------------------------------------- */
    /* test raw factorizations */
    /* ---------------------------------------------------------------------- */

    printf ("==================================== fac memory test\n") ;
    count = cm->malloc_count ;
    my_tries = -1 ;
    for (trial = 0 ; my_tries <= 0 ; trial++)
    {
	cm->print = 0 ;
	fflush (stdout) ;
	my_tries = trial ;
	A = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
	my_srand (trial+1) ;					/* RAND reset */
	err = raw_factor (A, FALSE) ;				/* RAND */
	CHOLMOD(free_sparse) (&A, cm) ;
	OK (CHOLMOD(print_common) ("cm", cm)) ;
	CHOLMOD(free_work) (cm) ;
	OK (count == cm->malloc_count) ;
	OK (inuse == cm->memory_inuse) ;
    }
    CHOLMOD(free_work) (cm) ;
    printf ("memory test: fac error %.1g trials "ID"\n", err, trial) ;
    printf ("initial count: "ID" final count "ID"\n",
	    (Int) count, (Int) cm->malloc_count) ;
    printf ("initial inuse: "ID" final inuse "ID"\n",
	    (Int) inuse, (Int) cm->memory_inuse) ;
    OK (count == cm->malloc_count) ;
    OK (inuse == cm->memory_inuse) ;

    /* ---------------------------------------------------------------------- */
    /* test raw factorizations (rowfac_mask) */
    /* ---------------------------------------------------------------------- */

    printf ("==================================== fac memory test2\n") ;
    count = cm->malloc_count ;
    my_tries = -1 ;
    for (trial = 0 ; my_tries <= 0 ; trial++)
    {
	cm->print = 0 ;
	fflush (stdout) ;
	my_tries = trial ;
	A = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
	my_srand (trial+1) ;					/* RAND reset */
	err = raw_factor2 (A, 0., 0) ;				/* RAND */
	CHOLMOD(free_sparse) (&A, cm) ;
	OK (CHOLMOD(print_common) ("cm", cm)) ;
	CHOLMOD(free_work) (cm) ;
	OK (count == cm->malloc_count) ;
	OK (inuse == cm->memory_inuse) ;
    }
    CHOLMOD(free_work) (cm) ;
    printf ("memory test: fac error %.1g trials "ID"\n", err, trial) ;
    printf ("initial count: "ID" final count "ID"\n",
	    (Int) count, (Int) cm->malloc_count) ;
    printf ("initial inuse: "ID" final inuse "ID"\n",
	    (Int) inuse, (Int) cm->memory_inuse) ;
    OK (count == cm->malloc_count) ;
    OK (inuse == cm->memory_inuse) ;

    /* ---------------------------------------------------------------------- */
    /* test augmented system solver */
    /* ---------------------------------------------------------------------- */

    printf ("==================================== aug memory test\n") ;
    count = cm->malloc_count ;
    my_tries = -1 ;
    for (trial = 0 ; my_tries <= 0 ; trial++)
    {
	cm->print = 0 ;
	fflush (stdout) ;
	my_tries = trial ;
	A = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
	err = aug (A) ;				/* no random number use */
	CHOLMOD(free_sparse) (&A, cm) ;
	OK (CHOLMOD(print_common) ("cm", cm)) ;
	CHOLMOD(free_work) (cm) ;
	OK (count == cm->malloc_count) ;
	OK (inuse == cm->memory_inuse) ;
    }
    CHOLMOD(free_work) (cm) ;
    printf ("memory test: aug error %.1g trials "ID"\n", err, trial) ;
    printf ("initial count: "ID" final count "ID"\n",
	    (Int) count, (Int) cm->malloc_count) ;
    printf ("initial inuse: "ID" final inuse "ID"\n",
	    (Int) inuse, (Int) cm->memory_inuse) ;
    OK (count == cm->malloc_count) ;
    OK (inuse == cm->memory_inuse) ;

    /* ---------------------------------------------------------------------- */
    /* test ops */
    /* ---------------------------------------------------------------------- */

    printf ("==================================== test_ops memory test\n") ;
    count = cm->malloc_count ;
    my_tries = -1 ;
    for (trial = 0 ; my_tries <= 0 ; trial++)
    {
	cm->print = 0 ;
	fflush (stdout) ;
	my_tries = trial ;
	A = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
	my_srand (trial+1) ;					/* RAND reset */
	err = test_ops (A) ;					/* RAND */
	CHOLMOD(free_sparse) (&A, cm) ;
	OK (CHOLMOD(print_common) ("cm", cm)) ;
	CHOLMOD(free_work) (cm) ;
	printf ("inuse "ID" "ID"\n", (Int) inuse, (Int) (cm->memory_inuse)) ;
	OK (count == cm->malloc_count) ;
	OK (inuse == cm->memory_inuse) ;
    }
    printf ("memory test: testops error %.1g trials "ID"\n", err, trial) ;
    printf ("initial count: "ID" final count "ID"\n",
	    (Int) count, (Int) cm->malloc_count) ;
    printf ("initial inuse: "ID" final inuse "ID"\n",
	    (Int) inuse, (Int) cm->memory_inuse) ;
    OK (count == cm->malloc_count) ;
    OK (inuse == cm->memory_inuse) ;

    /* ---------------------------------------------------------------------- */
    /* test lpdemo */
    /* ---------------------------------------------------------------------- */

    if (T == NULL || T->nrow != T->ncol)
    {
	printf ("==================================== lpdemo memory test\n") ;
	count = cm->malloc_count ;
	my_tries = -1 ;
	for (trial = 0 ; my_tries <= 0 ; trial++)
	{
	    cm->print = 0 ;
	    fflush (stdout) ;
	    my_tries = trial ;
	    my_srand (trial+1) ;				/* RAND reset */
	    err = lpdemo (T) ;					/* RAND */
	    OK (CHOLMOD(print_common) ("cm", cm)) ;
	    CHOLMOD(free_work) (cm) ;
	    OK (count == cm->malloc_count) ;
	    OK (inuse == cm->memory_inuse) ;
	}
	CHOLMOD(free_work) (cm) ;
	printf ("memory test: lpdemo error %.1g trials "ID"\n", err, trial) ;
	printf ("initial count: "ID" final count "ID"\n",
	    (Int) count, (Int) cm->malloc_count) ;
	printf ("initial inuse: "ID" final inuse "ID"\n",
	    (Int) inuse, (Int) cm->memory_inuse) ;
	OK (count == cm->malloc_count) ;
	OK (inuse == cm->memory_inuse) ;
    }

    /* ---------------------------------------------------------------------- */
    /* test solver */
    /* ---------------------------------------------------------------------- */

    printf ("==================================== solve memory test\n") ;
    count = cm->malloc_count ;
    my_tries = -1 ;
    for (trial = 0 ; my_tries <= 0 ; trial++)
    {
	CHOLMOD(defaults) (cm) ; cm->useGPU = 0 ;
	cm->supernodal = CHOLMOD_SUPERNODAL ;
	cm->metis_memory = 2.0 ;
	cm->nmethods = 4 ;
	cm->print = 0 ;
	fflush (stdout) ;
	my_tries = trial ;
	A = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
	my_srand (trial+1) ;					/* RAND reset */
	err = solve (A) ;					/* RAND */
	CHOLMOD(free_sparse) (&A, cm) ;
	OK (CHOLMOD(print_common) ("cm", cm)) ;
	CHOLMOD(free_work) (cm) ;
	OK (count == cm->malloc_count) ;
	OK (inuse == cm->memory_inuse) ;
    }
    CHOLMOD(free_work) (cm) ;
    printf ("memory test: solve error %.1g trials "ID"\n", err, trial) ;
    printf ("initial count: "ID" final count "ID"\n",
	    (Int) count, (Int) cm->malloc_count) ;
    printf ("initial inuse: "ID" final inuse "ID"\n",
	    (Int) inuse, (Int) cm->memory_inuse) ;
    OK (count == cm->malloc_count) ;
    OK (inuse == cm->memory_inuse) ;
    cm->supernodal = CHOLMOD_AUTO ;
    progress (1, '|') ;

    /* ---------------------------------------------------------------------- */
    /* restore original memory handler */
    /* ---------------------------------------------------------------------- */

    normal_memory_handler ( ) ;
    cm->print = 1 ;

    printf ("All memory tests OK, no error\n") ;
}
