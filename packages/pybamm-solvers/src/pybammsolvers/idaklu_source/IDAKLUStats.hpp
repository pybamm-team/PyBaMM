#ifndef PYBAMM_IDAKLU_STATS_HPP
#define PYBAMM_IDAKLU_STATS_HPP

/**
 * @brief Struct to hold IDA solver statistics
 *
 * This struct encapsulates all statistics retrieved from the IDA solver.
 * It supports accumulation via operator+= to correctly track stats across
 * solver reinitializations.
 */
struct IDAKLUStats
{
  // Integrator stats
  long nsteps = 0;     // Number of steps taken
  long nrevals = 0;    // Number of residual evaluations
  long nlinsetups = 0; // Number of linear solver setups
  long netfails = 0;   // Number of error test failures

  // Nonlinear solver stats
  long nniters = 0;  // Number of nonlinear iterations
  long nncfails = 0; // Number of nonlinear convergence failures

  // Preconditioner stats (BBD)
  long ngevalsBBDP = 0; // Number of g evaluations for BBD preconditioner

  /**
   * @brief Reset all statistics to zero
   */
  void reset()
  {
    nsteps = 0;
    nrevals = 0;
    nlinsetups = 0;
    netfails = 0;
    nniters = 0;
    nncfails = 0;
    ngevalsBBDP = 0;
  }

  /**
   * @brief Accumulate statistics from another IDAKLUStats object
   */
  IDAKLUStats& operator+=(IDAKLUStats const& other)
  {
    nsteps += other.nsteps;
    nrevals += other.nrevals;
    nlinsetups += other.nlinsetups;
    netfails += other.netfails;
    nniters += other.nniters;
    nncfails += other.nncfails;
    ngevalsBBDP += other.ngevalsBBDP;
    return *this;
  }
};

#endif // PYBAMM_IDAKLU_STATS_HPP
