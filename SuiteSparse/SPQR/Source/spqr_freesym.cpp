// =============================================================================
// === spqr_freesym ============================================================
// =============================================================================

// Frees the contents of the QR Symbolic object

#include "spqr.hpp"

void spqr_freesym
(
    spqr_symbolic **QRsym_handle,

    // workspace and parameters
    cholmod_common *cc
)
{
    spqr_symbolic *QRsym ;
    spqr_gpu *QRgpu ;
    Long m, n, anz, nf, rjsize, ns, ntasks ;

    if (QRsym_handle == NULL || *QRsym_handle == NULL)
    {
        // nothing to do; caller probably ran out of memory
        return ;
    }
    QRsym = *QRsym_handle  ;

    m = QRsym->m ;
    n = QRsym->n ;
    nf = QRsym->nf ;
    anz = QRsym->anz ;
    rjsize = QRsym->rjsize ;

    cholmod_l_free (n,      sizeof (Long), QRsym->Qfill, cc) ;
    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Super, cc) ;
    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Rp, cc) ;
    cholmod_l_free (rjsize, sizeof (Long), QRsym->Rj, cc) ;
    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Parent, cc) ;
    cholmod_l_free (nf+2,   sizeof (Long), QRsym->Childp, cc) ;
    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Child, cc) ;
    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Post, cc) ;
    cholmod_l_free (m,      sizeof (Long), QRsym->PLinv, cc) ;
    cholmod_l_free (n+2,    sizeof (Long), QRsym->Sleft, cc) ;
    cholmod_l_free (m+1,    sizeof (Long), QRsym->Sp, cc) ;
    cholmod_l_free (anz,    sizeof (Long), QRsym->Sj, cc) ;

    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Hip, cc) ;

    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Fm, cc) ;
    cholmod_l_free (nf+1,   sizeof (Long), QRsym->Cm, cc) ;

    cholmod_l_free (n,      sizeof (Long), QRsym->ColCount, cc) ;

    // gpu metadata
    QRgpu = QRsym->QRgpu;
    if(QRgpu)
    {
        cholmod_l_free (nf,   sizeof (Long),   QRgpu->RimapOffsets, cc) ;
        cholmod_l_free (nf,   sizeof (Long),   QRgpu->RjmapOffsets, cc) ;
        cholmod_l_free (nf+2, sizeof (Long),   QRgpu->Stagingp, cc) ;
        cholmod_l_free (nf,   sizeof (Long),   QRgpu->StageMap, cc) ;
        cholmod_l_free (nf+1, sizeof (size_t), QRgpu->FSize, cc) ;
        cholmod_l_free (nf+1, sizeof (size_t), QRgpu->RSize, cc) ;
        cholmod_l_free (nf+1, sizeof (size_t), QRgpu->SSize, cc) ;
        cholmod_l_free (nf,   sizeof (Long),   QRgpu->FOffsets, cc) ;
        cholmod_l_free (nf,   sizeof (Long),   QRgpu->ROffsets, cc) ;
        cholmod_l_free (nf,   sizeof (Long),   QRgpu->SOffsets, cc) ;

        cholmod_l_free (1, sizeof (spqr_gpu), QRgpu, cc) ;
    }

    // parallel analysis
    ntasks = QRsym->ntasks ;
    cholmod_l_free (ntasks+2, sizeof (Long), QRsym->TaskChildp, cc) ;
    cholmod_l_free (ntasks+1, sizeof (Long), QRsym->TaskChild, cc) ;
    cholmod_l_free (nf+1,     sizeof (Long), QRsym->TaskFront, cc) ;
    cholmod_l_free (ntasks+2, sizeof (Long), QRsym->TaskFrontp, cc) ;
    cholmod_l_free (ntasks+1, sizeof (Long), QRsym->TaskStack, cc) ;
    cholmod_l_free (nf+1,     sizeof (Long), QRsym->On_stack, cc) ;

    ns = QRsym->ns ;
    cholmod_l_free (ns+2,     sizeof (Long), QRsym->Stack_maxstack, cc) ;

    cholmod_l_free (1, sizeof (spqr_symbolic), QRsym, cc) ;

    *QRsym_handle = NULL ;
}
