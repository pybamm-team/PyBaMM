#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_GROUP_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_GROUP_HPP

#include "CasadiSolver.hpp"
#include "common.hpp"

/**
 * @brief class for a group of solvers.
 */
class CasadiSolverGroup
{
public:

  /**
   * @brief Default constructor
   */
  CasadiSolverGroup(std::vector<std::unique_ptr<CasadiSolver>> solvers, int number_of_states, int number_of_parameters, int length_of_return_vector, bool is_output_variables):
    m_solvers(std::move(solvers)),
    number_of_states(number_of_states),
    number_of_parameters(number_of_parameters),
    length_of_return_vector(length_of_return_vector),
    is_output_variables(is_output_variables)
    {}

  // no copy constructor (unique_ptr cannot be copied)
  CasadiSolverGroup(CasadiSolverGroup &) = delete;

  /**
   * @brief Default destructor
   */
  ~CasadiSolverGroup() = default;

  /**
   * @brief solver method that returns a vector of Solutions
   */
  std::vector<Solution> solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array inputs);

  private:
    std::vector<std::unique_ptr<CasadiSolver>> m_solvers;
    int number_of_states;
    int number_of_parameters;
    int length_of_return_vector;
    bool is_output_variables;
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_GROUP_HPP
