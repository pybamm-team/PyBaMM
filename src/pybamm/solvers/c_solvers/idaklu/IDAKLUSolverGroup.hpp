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
  IDAKLUSolverGroup(std::vector<std::unique_ptr<IDAKLUSolver>> solvers, int number_of_states, int number_of_parameters, int length_of_return_vector, bool is_output_variables):
    m_solvers(std::move(solvers)),
    number_of_states(number_of_states),
    number_of_parameters(number_of_parameters),
    length_of_return_vector(length_of_return_vector),
    is_output_variables(is_output_variables)
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
    np_array_dense inputs);


  private:
    std::vector<std::unique_ptr<IDAKLUSolver>> m_solvers;
    int number_of_states;
    int number_of_parameters;
    int length_of_return_vector;
    bool is_output_variables;
};

#endif // PYBAMM_IDAKLU_SOLVER_GROUP_HPP
