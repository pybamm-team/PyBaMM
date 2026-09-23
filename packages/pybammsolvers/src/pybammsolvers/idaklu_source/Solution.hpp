#ifndef PYBAMM_IDAKLU_SOLUTION_HPP
#define PYBAMM_IDAKLU_SOLUTION_HPP

#include "common.hpp"
#include "IDAKLUStats.hpp"

/**
 * @brief Solution class
 */
class Solution
{
public:
  /**
   * @brief Default Constructor
   */
  Solution() = default;

  /**
   * @brief Constructor
   */
  Solution(int retval, np_array t_np, np_array y_np, np_array yp_np,
           np_array yS_np, np_array ypS_np, np_array y_term_np,
           IDAKLUStats stats)
      : flag(retval), t(t_np), y(y_np), yp(yp_np), yS(yS_np), ypS(ypS_np),
        y_term(y_term_np), stats(stats)
  {
  }

  /**
   * @brief Default copy from another Solution
   */
  Solution(const Solution &solution) = default;

  int flag;
  np_array t;
  np_array y;
  np_array yp;
  np_array yS;
  np_array ypS;
  np_array y_term;
  IDAKLUStats stats;
};

#endif // PYBAMM_IDAKLU_SOLUTION_HPP
