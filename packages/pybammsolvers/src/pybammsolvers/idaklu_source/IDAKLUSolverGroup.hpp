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
   * `rust_owner` keeps whatever Python object owns the memory the solvers'
   * expression sets point at (the Rust evaluator pool) alive for at least as
   * long as the group, instead of relying on the caller to outlive us.
   * pybind11 destroys the group with the GIL held, so its destructor is safe.
   * It is required rather than defaulted, so a new construction site has to say
   * what owns its memory; CasADi's expression sets own theirs and pass none.
   */
  IDAKLUSolverGroup(std::vector<std::unique_ptr<IDAKLUSolver>> solvers, int number_of_states, int number_of_parameters,
                    py::object rust_owner):
    m_solvers(std::move(solvers)),
    number_of_states(number_of_states),
    number_of_parameters(number_of_parameters),
    m_rust_owner(std::move(rust_owner))
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
    np_array pbar = np_array(),
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
    py::object m_rust_owner;
};

#endif // PYBAMM_IDAKLU_SOLVER_GROUP_HPP
