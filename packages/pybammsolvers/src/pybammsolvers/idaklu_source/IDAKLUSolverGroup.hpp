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
   *
   * `number_of_inputs` is the width each solver copies from an inputs row. `rust_owner` keeps
   * the Python object owning the expression sets' memory alive with the group (none for
   * CasADi); pybind11 destroys the group with the GIL held.
   */
  IDAKLUSolverGroup(std::vector<std::unique_ptr<IDAKLUSolver>> solvers, int number_of_states, int number_of_parameters,
                    int number_of_inputs, py::object rust_owner):
    m_rust_owner(std::move(rust_owner)),
    m_solvers(std::move(solvers)),
    number_of_states(number_of_states),
    number_of_parameters(number_of_parameters),
    number_of_inputs(number_of_inputs)
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
    py::object logger = py::none(),
    np_array pbar = np_array());


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

    // Declared first so it is destroyed last, after the solvers that borrow its memory.
    py::object m_rust_owner;
    std::vector<std::unique_ptr<IDAKLUSolver>> m_solvers;
    int number_of_states;
    int number_of_parameters;
    int number_of_inputs;
};

#endif // PYBAMM_IDAKLU_SOLVER_GROUP_HPP
