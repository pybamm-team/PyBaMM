/* ========================================================================== */
/* === Demo/cholmod_demo.h ================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Demo Module.  Copyright (C) 2005-2013, Timothy A. Davis
 * -------------------------------------------------------------------------- */

#include "cholmod.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define TRUE 1
#define FALSE 0

#define CPUTIME (SuiteSparse_time ( ))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(a)   (((a) >= (0)) ? (a) : -(a))

#include "cholmod_function.h"
