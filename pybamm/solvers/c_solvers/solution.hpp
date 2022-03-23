#ifndef PYBAMM_SOLUTION_HPP
#define PYBAMM_SOLUTION_HPP

#include "idaklu_python.hpp"

class Solution
{
public:
  Solution(int retval, np_array t_np, np_array y_np, np_array yS_np)
      : flag(retval), t(t_np), y(y_np), yS(yS_np)
  {
  }

  int flag;
  np_array t;
  np_array y;
  np_array yS;
};

#endif // PYBAMM_SOLUTION_HPP
