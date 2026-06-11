#ifndef PYBAMM_IDAKLU_NOPROGRESS_GUARD_HPP
#define PYBAMM_IDAKLU_NOPROGRESS_GUARD_HPP

#include "common.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

/**
 * @brief Utility for checking lack-of-progress over a fixed-size sliding window
 */
class NoProgressGuard {
public:
  NoProgressGuard(size_t window_size, sunrealtype threshold_sec)
    : window_size_(window_size), threshold_sec_(threshold_sec), idx_(0) {
    if (!Disabled()) {
      dt_window_.assign(window_size_, threshold_sec_);
    }
  }

  inline bool Disabled() const {
    return window_size_ == 0 || threshold_sec_ == SUN_RCONST(0.0);
  }

  // initialize with a full window of threshold values to avoid immediate triggering
  inline void Initialize() {
    if (Disabled()) {
      return;
    }
    idx_ = 0;
    dt_window_.assign(window_size_, threshold_sec_);
  }

  // insert a new dt into the circular buffer
  inline void AddDt(sunrealtype dt) {
    if (Disabled()) {
        return;
    }
    
    dt_window_[idx_] = dt;
    idx_ = (idx_ + 1) % window_size_;
  }

  // violation if the running sum across the window remains below the threshold
  // early exit: as soon as we reach/exceed threshold, we are not violated
  inline bool Violated() const {
    if (Disabled()) {
        return false;
    }
    
    sunrealtype sum_dt = SUN_RCONST(0.0);
    for (const auto &dt : dt_window_) {
      sum_dt += dt;
      if (sum_dt >= threshold_sec_) {
        return false;
      }
    }
    return true;
  }

private:
  const size_t window_size_;
  const sunrealtype threshold_sec_;
  std::vector<sunrealtype> dt_window_;
  size_t idx_;
};

#endif // PYBAMM_IDAKLU_NOPROGRESS_GUARD_HPP


