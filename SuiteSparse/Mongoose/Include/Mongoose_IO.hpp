/* ========================================================================== */
/* === Include/Mongoose_IO.hpp ============================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/**
 * Simplified I/O functions for reading matrices and graphs
 *
 * For reading Matrix Market files into Mongoose, read_graph and read_matrix
 * are provided (depending on if a Graph class instance or CSparse matrix
 * instance is needed). The filename can be specified as either a const char*
 * (easier for C programmers) or std::string (easier from C++).
 */

// #pragma once
#ifndef MONGOOSE_IO_HPP
#define MONGOOSE_IO_HPP

#include "Mongoose_CSparse.hpp"
#include "Mongoose_Graph.hpp"
#include "Mongoose_Internal.hpp"
#include <string>

extern "C"
{
#include "mmio.h"
}

namespace Mongoose
{

/**
 * Generate a Graph from a Matrix Market file.
 *
 * Generate a Graph class instance from a Matrix Market file. The matrix
 * contained in the file must be sparse, real, and square. If the matrix
 * is not symmetric, it will be made symmetric with (A+A')/2. If the matrix has
 * more than one connected component, the largest will be found and the rest
 * discarded. If a diagonal is present, it will be removed.
 *
 * @param filename the filename or path to the Matrix Market File.
 */
Graph *read_graph(const std::string &filename);

/**
 * Generate a CSparse matrix from a Matrix Market file.
 *
 * Generate a cs struct instance from a Matrix Market file. The matrix
 * contained in the file must be sparse, real, and square. If the matrix
 * is not symmetric, it will be made symmetric with (A+A')/2. If the matrix has
 * more than one connected component, the largest will be found and the rest
 * discarded. If a diagonal is present, it will be removed.
 *
 * @param filename the filename or path to the Matrix Market File.
 * @param matcode the four character Matrix Market type code.
 */
cs *read_matrix(const std::string &filename, MM_typecode &matcode);

/**
 * Generate a Graph from a Matrix Market file.
 *
 * Generate a Graph class instance from a Matrix Market file. The matrix
 * contained in the file must be sparse, real, and square. If the matrix
 * is not symmetric, it will be made symmetric with (A+A')/2. If the matrix has
 * more than one connected component, the largest will be found and the rest
 * discarded. If a diagonal is present, it will be removed.
 *
 * @param filename the filename or path to the Matrix Market File.
 */
Graph *read_graph(const char *filename);

/**
 * Generate a CSparse matrix from a Matrix Market file.
 *
 * Generate a cs struct instance from a Matrix Market file. The matrix
 * contained in the file must be sparse, real, and square. If the matrix
 * is not symmetric, it will be made symmetric with (A+A')/2. If the matrix has
 * more than one connected component, the largest will be found and the rest
 * discarded. If a diagonal is present, it will be removed.
 *
 * @param filename the filename or path to the Matrix Market File.
 * @param matcode the four character Matrix Market type code.
 */
cs *read_matrix(const char *filename, MM_typecode &matcode);

} // end namespace Mongoose

#endif
