#ifndef PYBAMM_IDAKLU_SOLVER_GROUP_HPP
#define PYBAMM_IDAKLU_SOLVER_GROUP_HPP

#include "IDAKLUSolver.hpp"
#include "common.hpp"

/**
 * @brief class for a group of solvers.
 */
class IDAKLUSolverGroup
{
public:

  /**
   * @brief Default constructor
   */
  IDAKLUSolverGroup(std::vector<std::unique_ptr<IDAKLUSolver>> solvers, int number_of_states, int number_of_parameters):
    m_solvers(std::move(solvers)),
    number_of_states(number_of_states),
    number_of_parameters(number_of_parameters)
    {}

  // no copy constructor (unique_ptr cannot be copied)
  IDAKLUSolverGroup(IDAKLUSolverGroup &) = delete;

  /**
   * @brief Default destructor
   */
  ~IDAKLUSolverGroup() = default;

  /**
   * @brief solver method that returns a vector of Solutions
   */
  std::vector<Solution> solve(
    np_array t_eval_np,
    np_array t_interp_np,
    np_array y0_np,
    np_array yp0_np,
    np_array inputs,
    py::object logger = py::none());


  private:
    /**
     * @brief Emit each solver's buffered diagnostics (GIL held, serial only)
     */
    void flush_logs();

    /**
     * Copying a py::object touches Python reference counts, so this MUST run in
     * a serial section, before any OpenMP region.
     * @brief Give every solver the logger to write debug output through
     */
    void set_loggers(py::object logger);

    std::vector<std::unique_ptr<IDAKLUSolver>> m_solvers;
    int number_of_states;
    int number_of_parameters;
};

#endif // PYBAMM_IDAKLU_SOLVER_GROUP_HPP
