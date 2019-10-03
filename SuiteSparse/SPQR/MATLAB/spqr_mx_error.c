/* ========================================================================== */
/* === spqr_mx_error ======================================================== */
/* ========================================================================== */

/* Compile with gcc, not g++.  This is called by the CHOLMOD error handler,
 * which is itself in C.  A global variable is used for spumoni because the
 * parameter signature of this function cannot be changed; it is passed as
 * a function pointer to the CHOLMOD error handler.
 *
 *  errors:
 *
 *      CHOLMOD_TOO_LARGE       MATLAB:pmaxsize     problem too large
 *      CHOLMOD_OUT_OF_MEMORY   MATLAB:nomem        out of memory
 *      CHOLMOD_INVALID         MATLAB:internal     invalid option
 *      CHOLMOD_NOT_INSTALLED   MATLAB:internal     internal error
 *
 *  warnings:  these are not used by SuiteSparseQR.  They can only come from
 *  CHOLMOD, but they do not apply to SuiteSparseQR.
 *
 *      CHOLMOD_NOT_POSDEF      matrix not positive definite (for chol)
 *      CHOLMOD_DSMALL          diagonal too small (for LDL')
 */

#include "mex.h"
#include "cholmod.h"

int spqr_spumoni = 0 ;

void spqr_mx_error (int status, const char *file, int line, const char *msg)
{

    if (spqr_spumoni > 0 ||
        !(status == CHOLMOD_OUT_OF_MEMORY || status == CHOLMOD_TOO_LARGE))
    {
        mexPrintf ("ERROR: %s line %d, status %d: %s\n",
            file, line, status, msg) ;
    }

    if (status < CHOLMOD_OK)
    {
        switch (status)
        {
            case CHOLMOD_OUT_OF_MEMORY:
	        mexErrMsgIdAndTxt ("MATLAB:nomem",
                "Out of memory. Type HELP MEMORY for your options.") ;

            case CHOLMOD_TOO_LARGE:
	        mexErrMsgIdAndTxt ("MATLAB:pmaxsize", 
                "Maximum variable size allowed by the program is exceeded.") ;
                break ;

            default:
                /* CHOLMOD_NOT_INSTALLED and CHOLMOD_INVALID:
                   These errors should be caught by the mexFunction interface
                   to SuiteSparseQR, not by the CHOLMOD or SuiteSparseQR
                   internal code itself */
	        mexErrMsgIdAndTxt ("MATLAB:internal", "Internal error") ;
                break ;
        }            
    }
    else
    {
        /* A CHOMOD warning is not used by SuiteSparseQR at all.  Thus, it is
           reported here as an internal error rather than as a warning. */
	/* mexWarnMsgTxt (msg) ; */
        mexErrMsgIdAndTxt ("MATLAB:internal", "Internal error") ;
    }
}
