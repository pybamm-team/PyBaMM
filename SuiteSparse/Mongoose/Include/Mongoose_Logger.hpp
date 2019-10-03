/* ========================================================================== */
/* === Include/Mongoose_Logger.hpp ========================================== */
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

// #pragma once
#ifndef MONGOOSE_LOGGER_HPP
#define MONGOOSE_LOGGER_HPP

#include <iostream>
#include <string>
#include <time.h>

// Default Logging Levels
#ifndef LOG_ERROR
#define LOG_ERROR 1
#endif

#ifndef LOG_WARN
#define LOG_WARN 0
#endif

#ifndef LOG_INFO
#define LOG_INFO 0
#endif

#ifndef LOG_TEST
#define LOG_TEST 0
#endif

// Main Logging Macros
#define LogError(msg)                                                          \
    do                                                                         \
    {                                                                          \
        if (LOG_ERROR)                                                         \
            (std::cout << __FILE__ << ":" << __LINE__ << ": " << msg);         \
    } while (0)
#define LogWarn(msg)                                                           \
    do                                                                         \
    {                                                                          \
        if (LOG_WARN)                                                          \
            (std::cout << __FILE__ << ":" << __LINE__ << ": " << msg);         \
    } while (0)
#define LogInfo(msg)                                                           \
    do                                                                         \
    {                                                                          \
        if (LOG_INFO)                                                          \
            (std::cout << msg);                                                \
    } while (0)
#define LogTest(msg)                                                           \
    do                                                                         \
    {                                                                          \
        if (LOG_TEST)                                                          \
            (std::cout << msg);                                                \
    } while (0)

namespace Mongoose
{

typedef enum DebugType
{
    None  = 0,
    Error = 1,
    Warn  = 2,
    Info  = 4,
    Test  = 8,
    All   = 15
} DebugType;

typedef enum TimingType
{
    MatchingTiming   = 0,
    CoarseningTiming = 1,
    RefinementTiming = 2,
    FMTiming         = 3,
    QPTiming         = 4,
    IOTiming         = 5
} TimingType;

class Logger
{
private:
    static int debugLevel;
    static bool timingOn;
    static clock_t clocks[6];
    static float times[6];

public:
    static inline void tic(TimingType timingType);
    static inline void toc(TimingType timingType);
    static inline float getTime(TimingType timingType);
    static inline int getDebugLevel();
    static void setDebugLevel(int debugType);
    static void setTimingFlag(bool tFlag);
    static void printTimingInfo();
};

/**
 * Start a timer for a given type/part of the code.
 *
 * Given a timingType (MatchingTiming, CoarseningTiming, RefinementTiming,
 * FMTiming, QPTiming, or IOTiming), a clock is started for that computation.
 * The general structure is to call tic(IOTiming) at the beginning of an I/O
 * operation, then call toc(IOTiming) at the end of the I/O operation.
 *
 * Note that problems can occur and timing results may be inaccurate if a tic
 * is followed by another tic (or a toc is followed by another toc).
 *
 * @param timingType The portion of the library being timed (MatchingTiming,
 *   CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 */
inline void Logger::tic(TimingType timingType)
{
    if (timingOn)
    {
        clocks[timingType] = clock();
    }
}

/**
 * Stop a timer for a given type/part of the code.
 *
 * Given a timingType (MatchingTiming, CoarseningTiming, RefinementTiming,
 * FMTiming, QPTiming, or IOTiming), a clock is stopped for that computation.
 * The general structure is to call tic(IOTiming) at the beginning of an I/O
 * operation, then call toc(IOTiming) at the end of the I/O operation.
 *
 * Note that problems can occur and timing results may be inaccurate if a tic
 * is followed by another tic (or a toc is followed by another toc).
 *
 * @param timingType The portion of the library being timed (MatchingTiming,
 *   CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 */
inline void Logger::toc(TimingType timingType)
{
    if (timingOn)
    {
        times[timingType]
            += ((float)(clock() - clocks[timingType])) / CLOCKS_PER_SEC;
    }
}

/**
 * Get the time recorded for a given timing type.
 *
 * Retreive the total clock time for a given timing type (MatchingTiming,
 * CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 *
 * @param timingType The portion of the library being timed (MatchingTiming,
 *   CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 */
inline float Logger::getTime(TimingType timingType)
{
    return times[timingType];
}

inline int Logger::getDebugLevel()
{
    return debugLevel;
}

} // end namespace Mongoose

#endif
