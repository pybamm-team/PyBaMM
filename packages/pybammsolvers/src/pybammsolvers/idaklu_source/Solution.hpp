#ifndef PYBAMM_IDAKLU_SOLUTION_HPP
#define PYBAMM_IDAKLU_SOLUTION_HPP

#include "common.hpp"
#include "IDAKLUStats.hpp"
#include <utility>

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
           np_array t_additive_np, np_array yS_np, np_array ypS_np, np_array y_init_np,
           np_array y_term_np, np_array yS_init_np, np_array yS_term_np,
           const IDAKLUStats &stats)
      : flag(retval), t(std::move(t_np)), y(std::move(y_np)),
        yp(std::move(yp_np)), t_additive(std::move(t_additive_np)), yS(std::move(yS_np)), ypS(std::move(ypS_np)),
        y_init(std::move(y_init_np)), y_term(std::move(y_term_np)),
        yS_init(std::move(yS_init_np)), yS_term(std::move(yS_term_np)),
        stats(stats)
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
  np_array t_additive;
  np_array yS;
  np_array ypS;
  np_array y_init;
  np_array y_term;
  np_array yS_init;
  np_array yS_term;
  IDAKLUStats stats;
};

#endif // PYBAMM_IDAKLU_SOLUTION_HPP
