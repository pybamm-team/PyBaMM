/* ========================================================================== */
/* === RBio/RBio/RBerror.c: mexFunction error-handling ====================== */
/* ========================================================================== */

/* Copyright 2009, Timothy A. Davis, All Rights Reserved.
   Refer to RBio/Doc/license.txt for the RBio license. */

#include "RBio.h"

void RBerror (int status)
{
    switch (status)
    {
        case RBIO_OK:
            break ;

        case RBIO_CP_INVALID:
            mexErrMsgTxt ("column pointers are invalid") ;
            break ;

        case RBIO_ROW_INVALID:
            mexErrMsgTxt ("row indices are out of range") ;
            break ;

        case RBIO_DUPLICATE:
            mexErrMsgTxt ("duplicate entry") ;
            break ;

        case RBIO_EXTRANEOUS:
            mexErrMsgTxt
                ("entries in upper triangular part of symmetric matrix") ;
            break ;

        case RBIO_TYPE_INVALID:
            mexErrMsgTxt ("matrix type invalid") ;
            break ;

        case RBIO_DIM_INVALID:
            mexErrMsgTxt ("matrix dimensions invalid") ;
            break ;

        case RBIO_JUMBLED:
            mexErrMsgTxt ("matrix contains unsorted columns") ;
            break ;

        case RBIO_ARG_ERROR:
            mexErrMsgTxt ("input arguments invalid") ;
            break ;

        case RBIO_OUT_OF_MEMORY:
            mexErrMsgTxt ("out of memory") ;
            break ;

        case RBIO_MKIND_INVALID:
            mexErrMsgTxt ("mkind is invalid") ;
            break ;

        case RBIO_UNSUPPORTED:
            mexErrMsgTxt ("finite-element form unsupported") ;
            break ;

        case RBIO_HEADER_IOERROR:
            mexErrMsgTxt ("header I/O error") ;
            break ;

        case RBIO_CP_IOERROR:
            mexErrMsgTxt ("column pointers I/O error") ;
            break ;
            
        case RBIO_ROW_IOERROR:
            mexErrMsgTxt ("row indices I/O error") ;
            break ;

        case RBIO_VALUE_IOERROR:
            mexErrMsgTxt ("numerical values I/O error") ;
            break ;

        default:
            mexErrMsgTxt ("unknown error") ;
            break ;
    }
}
