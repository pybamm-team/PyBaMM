// =============================================================================
// === spqr_parallel ===========================================================
// =============================================================================

// Factorize all the tasks in parallel with TBB.
// The GPU is not used.

#ifdef HAVE_TBB
#include "spqr.hpp"
#include <tbb/task_scheduler_init.h>
#include <tbb/task.h>

using namespace tbb ;

// =============================================================================
// === spqr_zippy ==============================================================
// =============================================================================

template <typename Entry> class spqr_zippy: public task
{
  public:

    // -------------------------------------------------------------------------
    // spqr_zippy state
    // -------------------------------------------------------------------------

    const Long id ;
    spqr_blob <Entry> *Blob ;

    // -------------------------------------------------------------------------
    // spqr_zippy constructor
    // -------------------------------------------------------------------------

    spqr_zippy (Long id_, spqr_blob <Entry> *Blob_) : id (id_), Blob (Blob_) { }

    // -------------------------------------------------------------------------
    // spqr_zippy task
    // -------------------------------------------------------------------------

    task* execute ( )
    {

        // ---------------------------------------------------------------------
        // spawn my children
        // ---------------------------------------------------------------------

        Long *TaskChildp = Blob->QRsym->TaskChildp ;
        Long *TaskChild  = Blob->QRsym->TaskChild ;
        Long pfirst = TaskChildp [id] ;
        Long plast  = TaskChildp [id+1] ;
        Long nchildren = plast - pfirst ;

        if (nchildren > 0)
        {
            // create a list of TBB tasks, one for each child
            task_list TasksToDo ;
            for (Long i = 0 ; i < nchildren ; i++)
            {
                Long child = TaskChild [pfirst+i] ;
                TasksToDo.push_back (*new (allocate_child ( ))
                    spqr_zippy (child, Blob)) ;
            }
            // spawn all children and wait for all of them to finish
            set_ref_count (nchildren + 1) ;
            spawn_and_wait_for_all (TasksToDo) ;
        }

        // ---------------------------------------------------------------------
        // chilren are done, do my own task
        // ---------------------------------------------------------------------

        spqr_kernel (id, Blob) ;

        return (NULL) ;
    }
} ;


// =============================================================================
// === spqr_parallel ===========================================================
// =============================================================================

template <typename Entry> void spqr_parallel
(
    Long ntasks,
    int nthreads,
    spqr_blob <Entry> *Blob
)
{
    // fire up TBB on the task tree, starting at the root id = ntasks-1
    task_scheduler_init
        init (nthreads <= 0 ? (task_scheduler_init::automatic) : nthreads) ;
    spqr_zippy <Entry> & a = *new (task::allocate_root ( ))
        spqr_zippy <Entry> (ntasks-1, Blob) ;
    task::spawn_root_and_wait (a) ;
}

// =============================================================================

template void spqr_parallel <double>
(
    Long ntasks,
    int nthreads,
    spqr_blob <double> *Blob
) ;

template void spqr_parallel <Complex>
(
    Long ntasks,
    int nthreads,
    spqr_blob <Complex> *Blob
) ;

#endif
