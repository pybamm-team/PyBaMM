// =============================================================================
// === GPUQREngine/Include/GPUQREngine_timing.hpp ==============================
// =============================================================================
//
// Contains timing macros that wrap GPU timing logic, using cudaEvents.
//
// =============================================================================


#ifndef GPUQRENGINE_TIMING_HPP
#define GPUQRENGINE_TIMING_HPP

#ifdef TIMING

// create the timer
#define TIMER_INIT()                            \
cudaEvent_t start, stop ;                       \
cudaEventCreate(&start) ;                       \
cudaEventCreate(&stop) ;                        \

// start the timer
#define TIMER_START()                           \
cudaEventRecord(start, 0);                      \

// stop the timer and get the time since the last tic
// t is the time since the last TIMER_START()
#define TIMER_STOP(t)                           \
cudaEventRecord(stop, 0);                       \
cudaThreadSynchronize();                        \
cudaEventElapsedTime(&(t), start, stop);        \

// destroy the timer
#define TIMER_FINISH()                          \
cudaEventDestroy(start);                        \
cudaEventDestroy(stop);                         \

#else

// compile with no timing
#define TIMER_INIT() ;
#define TIMER_START() ;
#define TIMER_STOP(t) ;
#define TIMER_FINISH() ;

#endif
#endif
