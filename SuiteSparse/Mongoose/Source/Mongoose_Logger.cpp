/* ========================================================================== */
/* === Source/Mongoose_Logger.cpp =========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/**
 * Centralized debug and timing manager
 *
 * For debug and timing information to be displayed via stdout. This system
 * allows this information to be displayed (or not) without recompilation.
 * Timing inforation for different *portions of the library are also managed
 * here with a tic/toc pattern.
 */

#include "Mongoose_Logger.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include <iomanip>

namespace Mongoose
{

int Logger::debugLevel = None;
bool Logger::timingOn  = false;
clock_t Logger::clocks[6];
float Logger::times[6];

void Logger::setDebugLevel(int debugType)
{
    debugLevel = debugType;
}

void Logger::setTimingFlag(bool tFlag)
{
    timingOn = tFlag;
}

void Logger::printTimingInfo()
{
    std::cout << " Matching:   " << std::setprecision(4)
              << times[MatchingTiming] << "s\n";
    std::cout << " Coarsening: " << std::setprecision(4)
              << times[CoarseningTiming] << "s\n";
    std::cout << " Refinement: " << std::setprecision(4)
              << times[RefinementTiming] << "s\n";
    std::cout << " FM:         " << std::setprecision(4) << times[FMTiming]
              << "s\n";
    std::cout << " QP:         " << std::setprecision(4) << times[QPTiming]
              << "s\n";
    std::cout << " IO:         " << std::setprecision(4) << times[IOTiming]
              << "s\n";
}

} // end namespace Mongoose
