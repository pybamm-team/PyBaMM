// =============================================================================
// === spqrgpu_buildAssemblyMaps ===============================================
// =============================================================================

#ifdef GPU_BLAS
#include "spqr.hpp"

void spqrgpu_buildAssemblyMaps
(
    Long numFronts,
    Long n,
    Long *Fmap,
    Long *Post,
    Long *Super,
    Long *Rp,
    Long *Rj,
    Long *Sleft,
    Long *Sp,
    Long *Sj,
    double *Sx,
    Long *Fm,
    Long *Cm,
    Long *Childp,
    Long *Child,
    Long *CompleteStair,
    int *CompleteRjmap,
    Long *RjmapOffsets,
    int *CompleteRimap,
    Long *RimapOffsets,
    SEntry *cpuS
)
{
    PR (("GPU: building assembly maps:\n")) ;

    /* Use Fmap and Stair to map a front's local rows to global rows. */
    Long sindex = 0;

    for(Long pf=0; pf<numFronts; pf++) // iterate in post-order
    {
        Long f = Post[pf];

        /* Build Fmap for front f. */
        Long pstart = Rp[f], pend = Rp[f+1];
        for (Long p=pstart; p<pend ; p++)
        {
            Fmap[Rj[p]] = p - pstart;
        }

        /* Get workspaces for offset front members */
        Long *Stair = CompleteStair + Rp[f];

        // ---------------------------------------------------------------------
        // initialize the staircase for front F
        // ---------------------------------------------------------------------

        // initialize the staircase with original rows of S
        Long col1 = Super[f], col2 = Super[f+1];
        Long fp = col2 - col1;
        Long fn = Rp[f+1] - Rp[f];

        for (Long j = 0 ; j < fp ; j++)
        {
            // global column j+col1 is the jth pivot column of front F
            Long col = j + col1 ;
            Stair[j] = Sleft [col+1] - Sleft [col] ;
            PR (("GPU init rows, j: %ld count %ld\n", j, Stair[j])) ;
        }

        // contribution blocks from children will be added here
        for (Long j = fp ; j < fn ; j++){ Stair[j] = 0; }

        /* Build Rjmap and Stair for each child of the current front. */
        for(Long p=Childp[f]; p<Childp[f+1]; p++)
        {
            Long c = Child[p];
            PR (("child %ld\n", c)) ;
            ASSERT (c >= 0 && c < f) ;
            int *Rjmap = CompleteRjmap + RjmapOffsets[c];
            Long pc = Rp [c] ;
            Long pcend = Rp [c+1] ;
            Long fnc = pcend - pc ;              // total # cols in child F
            Long fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
            Long cm = Cm [c] ;                   // # of rows in child C
            Long cn = fnc - fpc ;                // # of cols in child C

            // Create the relative column indices of the child's contributions
            // to the parent.  The global column 'col' in the child is a
            // contribution to the jth column of its parent front.  j is
            // a relative column index.
            for (Long pp=0 ; pp<cn ; pp++)
            {
                Long col = Rj [pc+fpc+pp] ;    // global column index
                Long j = Fmap [col] ;          // relative column index
                Rjmap [pp] = j ;               // Save this column
            }

            ASSERT (cm >= 0 && cm <= cn) ;
            pc += fpc ;                        // pointer to column indices in C
            ASSERT (pc + cn == Rp [c+1]) ;
            PR (("  cm %ld cn %ld\n", cm, cn)) ;

            // add the child rows to the staircase
            for (Long ci = 0 ; ci < cm ; ci++)
            {
                Long col = Rj [pc + ci] ;      // leftmost col of this row of C
                Long j = Fmap [col] ;          // global col is jth col of F
                PR (("  child col %ld j %ld\n", col, j)) ;
                ASSERT (j >= 0 && j < fn) ;
                Stair[j]++ ;                   // add this row to jth staircase
            }
        }

        // ---------------------------------------------------------------------
        // replace Stair with cumsum ([0 Stair]), and find # rows of F
        // ---------------------------------------------------------------------

        Long fm = 0 ;
        for (Long j = 0 ; j < fn ; j++)
        {
            Long t = fm ;
            fm += Stair[j] ;
            Stair[j] = t ;
        }
        PR (("fm %ld %ld\n", fm, Stair[fn-1])) ;

        // ---------------------------------------------------------------------
        // pack all the S values into the cpuS workspace & advance scalar stair
        // ---------------------------------------------------------------------

        Long Scount = MAX(0, Sp[Sleft[Super[f+1]]] - Sp[Sleft[Super[f]]]);
        if(Scount > 0)
        {
            for(Long k=0 ; k<fp ; k++)
            {
                /* pack all rows whose leftmost global column index is k+col1 */
                Long leftcol = k + col1 ;

                /* for each row of S whose leftmost column is leftcol */
                for (Long row = Sleft [leftcol]; row < Sleft [leftcol+1]; row++)
                {
                    // get the location of this row in F & advance the staircase
                    Long i = Stair[k]++;

                    /* Pack into S */
                    for (Long p=Sp[row] ; p<Sp[row+1] ; p++)
                    {
                        Long j = Sj[p];
                        cpuS[sindex].findex = i*fn + j;
                        cpuS[sindex].value = Sx[p];
                        sindex++;
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // build Rimap
        // ---------------------------------------------------------------------

        for (Long p = Childp [f] ; p < Childp [f+1] ; p++)
        {
            /* Get child details */
            Long c = Child [p] ;                 // get the child c of front F
            int *Rimap = CompleteRimap + RimapOffsets[c];
            int *Rjmap = CompleteRjmap + RjmapOffsets[c];
            Long pc = Rp [c] ;
            // Long pcend = Rp [c+1] ;
            // Long fnc = pcend - pc ;           // total # cols in child F
            Long fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
            Long cm = Cm [c] ;                   // # of rows in child C
            // Long cn = (fnc - fpc) ;           // cn =# of cols in child C
            // ASSERT (cm >= 0 && cm <= cn) ;
            pc += fpc ;                        // pointer to column indices in C
            // ASSERT (pc + cn == Rp [c+1]) ;

            /* -------------------------------------------------------------- */
            /* construct the Rimap                                            */
            /* -------------------------------------------------------------- */

            for (Long ci = 0 ; ci < cm ; ci++)
            {
                Long j = Rjmap[ci] ;          // global col is jth col of F
                ASSERT (j >= 0 && j < fn) ;
                Long i = Stair[j]++ ;         // add row F(i,:) to jth staircase
                ASSERT (i >= 0 && i < fm) ;
                Rimap[ci] = i ;               // keep track of the mapping
            }
        }
    }
}
#endif
