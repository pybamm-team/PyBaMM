#ifndef PYBAMM_IDAKLU_SOLUTION_HPP
#define PYBAMM_IDAKLU_SOLUTION_HPP

#include "common.hpp"

/**
 * @brief Solution class
 */
class Solution
{
public:
  /**
   * @brief Constructor
   */
  Solution(int retval, np_array t_np, np_array y_np, np_array yS_np)
      : flag(retval), t(t_np), y(y_np), yS(yS_np)
  {
  }

  int flag;
  np_array t;
  np_array y;
  np_array yS;
};

#endif // PYBAMM_IDAKLU_COMMON_HPP
