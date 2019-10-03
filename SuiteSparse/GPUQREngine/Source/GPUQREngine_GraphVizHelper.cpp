// =============================================================================
// === GPUQREngine/Source/GPUQREngine_GraphVisHelper.cpp =======================
// =============================================================================
// === This is used for development and debugging only. ========================
// =============================================================================
//
// This file contains logic to render the current state of the BucketList data
// structure, including the current arrangement of row tiles into column
// buckets and bundles. The output language is DOT for use with either
// GraphViz's dot or sfdp package.
//
// Bundles are colored by task type:
//   Apply         : Yellow
//   Factorize     : Red
//   ApplyFactorize: Orange
//
// =============================================================================

#include "GPUQREngine_Internal.hpp"

#ifdef GPUQRENGINE_RENDER

#include "GPUQREngine_BucketList.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void GraphVizHelper_ComputeBundleLabel
(
    LLBundle& bundle,   // C++ Reference to the bundle
    char *label         // (output) The label to use for the bundle
);

static int DotSID = 1;

void GPUQREngine_RenderBuckets(BucketList *buckets)
{
    // if(!RENDER_DENSE_FACTORIZATION) return;

    LLBundle *bundles = buckets->Bundles;

    int numBundles = buckets->numBundles;
    char bundleNames[numBundles][64];
    char bundleLabel[numBundles][64];

    Int *head = buckets->head;
    Int *next = buckets->next;

    char filename[64];
    sprintf(filename, "out_%d.dot", DotSID++);

    FILE *output = fopen(filename, "w");

    bool RenderOutline = false;
    bool UseFancyRendering = false;

    /* If we want to render the outline, do it first. */
    if (RenderOutline)
    {
        fprintf(output, "graph O\n");
        fprintf(output, "{\n");
        fprintf(output, "node [shape=circle, style=filled, color=black];\n");
        for (int colBucket = 0; colBucket < buckets->numBuckets; colBucket++)
        {
            /* Shade the point depending on the region. */
            const char *fillColor = "gray";
            if (colBucket < buckets->Wavefront) fillColor = "green";
            else if (colBucket <= buckets->LastBucket) fillColor = "red";
            fprintf(output, "O_%d [fillcolor=%s];\n", colBucket, fillColor);
        }
        fprintf(output, "}\n");
    }

    fprintf(output, "digraph D\n");
    fprintf(output, "{\n");
    fprintf(output, "node [shape=record, style=filled];\n");

    int start = (UseFancyRendering ? MAX(0, buckets->Wavefront - 1) : 0);
    int end = (UseFancyRendering ?
        MIN(buckets->numBuckets, buckets->LastBucket + 1)
        : buckets->numBuckets);

    for (int colBucket = start; colBucket < end; colBucket++)
    {
//      const char *colBucketName = "CB_" + colBucket;
//      const char *colBucketLabel = "ColBucket" + colBucket;
            // +" (" + buckets->bundleCount[colBucket] + " bundles)";
//      const char *colBucketEmptyName = "CBE_" + colBucket;
//      const char *colBucketEmptyLabel = "Empty";
        const char *wavefrontColor = (colBucket < buckets->Wavefront ?
            "green" : "gray");

        fprintf(output, "CB_%d [label=\"ColBucket%d\", fillcolor=%s];\n",
            colBucket, colBucket, wavefrontColor);
        fprintf(output, "CBE_%d [label=\"Empty\"];\n", colBucket);

        /* Render the idle tiles. */
        int node;
        char lastTile[32]; strcpy(lastTile, "");
        if ((node = head[colBucket]) != EMPTY)
        {
            sprintf(lastTile, "IdleTile_%d", node);

            const char *nodeShape = (buckets->triu[node] ?
                "triangle" : "ellipse");
            fprintf(output,
                "IdleTile_%d [shape=%s, fillcolor=%s, label=\"%d\"];\n",
                node, nodeShape, wavefrontColor, node);
            fprintf(output, "CB_%d -> IdleTile_%d;\n", colBucket, node);
            int last = node;
            while ((node = next[node]) != EMPTY)
            {
                sprintf(lastTile, "IdleTile_%d", node);

                nodeShape = (buckets->triu[node] ? "triangle" : "ellipse");
                fprintf(output, "IdleTile_%d [shape=%s, label=\"%d\"];\n",
                    node, nodeShape, node);
                fprintf(output, "IdleTile_%d -> IdleTile_%d;\n", last, node);
                last = node;
            }
        }

        /* Now render the bundles native to this bucket. */

        /* Write the nodes */
        int bbc = 0;
        for(int i=0; i<numBundles; i++)
        {
            strcpy(bundleNames[i], "");
            strcpy(bundleLabel[i], "");
        }
        for(int i=0; i<numBundles; i++)
        {
            LLBundle& bundle = bundles[i];
            if (bundle.NativeBucket == colBucket)
            {
                sprintf(bundleNames[bbc], "CB_%d_HB_%d", colBucket, bbc);

                const char *taskColor = "white";
                bool isApply = (bundle.CurrentTask == TASKTYPE_GenericApply);
                bool isFactorize =
                    (bundle.CurrentTask == TASKTYPE_GenericFactorize);
                taskColor =
                    (isApply ? "yellow" : isFactorize ? "red" : "orange");

                GraphVizHelper_ComputeBundleLabel(bundle, bundleLabel[bbc]);
                fprintf(output, "%s [fillcolor=\"%s\", label=\"%s\"];\n",
                    bundleNames[bbc], taskColor, bundleLabel[bbc]);
                bbc++;
            }
        }

        /* Print connectivity. */
        if (bbc > 0)
        {
            if(!strcmp(lastTile, ""))
            {
                fprintf(output, "CB_%d -> %s;\n", colBucket, bundleNames[0]);
            }
            else
            {
                fprintf(output, "%s -> %s;\n", lastTile, bundleNames[0]);
            }

            for (int i = 1; i < bbc; i++)
            {
                fprintf(output, "%s -> %s;\n",
                    bundleNames[i-1], bundleNames[i]);
            }
            fprintf(output, "%s -> CBE_%d;\n", bundleNames[bbc-1], colBucket);
        }
        else if (head[colBucket] != EMPTY)
        {
            fprintf(output, "%s -> CBE_%d;\n", lastTile, colBucket);
        }
        else
        {
            fprintf(output, "CB_%d -> CBE_%d;\n", colBucket, colBucket);
        }
    }

    fprintf(output, "}\n");
    fclose(output);
}

void GraphVizHelper_ComputeBundleLabel
(
    LLBundle& bundle,   // C++ Reference to the bundle
    char *label         // (output) The label to use for the bundle
)
{
    Int *next = bundle.Buckets->next;

    char temp[16];
    strcpy(temp, "");

    strcpy(label, "");
    if (bundle.Shadow != EMPTY)
    {
        sprintf(temp, "s%ld%s", bundle.Shadow,
            (bundle.First != EMPTY ? "|" : ""));
        strcat(label, temp);
    }

    for (int i = bundle.First; i!=EMPTY; i=next[i])
    {
        sprintf(temp, "%d%s", i, (next[i] != EMPTY ? "|" : ""));
        strcat(label, temp);
    }
    for (int i = bundle.Delta; i!=EMPTY; i=next[i])
    {
        sprintf(temp, "|%s%d", (i == bundle.Delta ? "D" : "d"), i);
        strcat(label, temp);
    }
}
#endif
