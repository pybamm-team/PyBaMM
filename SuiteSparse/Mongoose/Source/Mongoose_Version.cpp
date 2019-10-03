/* ========================================================================== */
/* === Source/Mongoose_Version.cpp ========================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_Version.hpp"
#include <sstream>

namespace Mongoose
{

int major_version()
{
    return Mongoose_VERSION_MAJOR;
}

int minor_version()
{
    return Mongoose_VERSION_MINOR;
}

int patch_version()
{
    return Mongoose_VERSION_PATCH;
}

std::string mongoose_version()
{
    std::ostringstream stringStream;
    stringStream << major_version() << "." << minor_version() << "."
                 << patch_version() << " " << Mongoose_DATE;
    return stringStream.str();
}

} // end namespace Mongoose
