#ifndef PYBAMM_IDAKLU_SOLUTION_DATA_HPP
#define PYBAMM_IDAKLU_SOLUTION_DATA_HPP

#include <vector>
#include "common.hpp"
#include "Solution.hpp"

namespace {
// Helper: wrap a vector as a numpy array (zero-copy via capsule)
// MUST be called with GIL held!
inline np_array vector_to_numpy(std::vector<sunrealtype>&& vec) {
    auto* holder = new std::vector<sunrealtype>(std::move(vec));
    py::capsule capsule(holder, [](void* v) {
        delete reinterpret_cast<std::vector<sunrealtype>*>(v);
    });
    return np_array(holder->size(), holder->data(), capsule);
}

// Helper: wrap a vector as a 3D numpy array (zero-copy via capsule)
// MUST be called with GIL held!
inline np_array vector_to_numpy_3d(std::vector<sunrealtype>&& vec, 
                                    ptrdiff_t d0, ptrdiff_t d1, ptrdiff_t d2) {
    auto* holder = new std::vector<sunrealtype>(std::move(vec));
    py::capsule capsule(holder, [](void* v) {
        delete reinterpret_cast<std::vector<sunrealtype>*>(v);
    });
    return np_array(std::vector<ptrdiff_t>{d0, d1, d2}, holder->data(), capsule);
}
} // anonymous namespace

/**
 * @brief SolutionData class - holds raw C++ vectors from solve().
 * Numpy arrays are created only in generate_solution() when GIL is held.
 */
class SolutionData
{
  public:
    SolutionData() = default;
    
    SolutionData(
      int flag,
      std::vector<sunrealtype>&& t,
      std::vector<sunrealtype>&& y,
      std::vector<sunrealtype>&& yp,
      std::vector<sunrealtype>&& yS,
      std::vector<sunrealtype>&& ypS,
      std::vector<sunrealtype>&& yterm,
      ptrdiff_t arg_sens0,
      ptrdiff_t arg_sens1,
      ptrdiff_t arg_sens2,
      bool save_hermite)
      : flag(flag),
        t_vec(std::move(t)),
        y_vec(std::move(y)),
        yp_vec(std::move(yp)),
        yS_vec(std::move(yS)),
        ypS_vec(std::move(ypS)),
        yterm_vec(std::move(yterm)),
        arg_sens0(arg_sens0),
        arg_sens1(arg_sens1),
        arg_sens2(arg_sens2),
        save_hermite(save_hermite)
    {}

    ~SolutionData() = default;
    SolutionData(const SolutionData&) = delete;
    SolutionData& operator=(const SolutionData&) = delete;
    SolutionData(SolutionData&&) noexcept = default;
    SolutionData& operator=(SolutionData&&) noexcept = default;

    /**
     * @brief Convert raw vectors to numpy arrays and create Solution.
     * MUST be called with GIL held (i.e., in serial section).
     */
    Solution generate_solution() {
      return Solution(
        flag,
        vector_to_numpy(std::move(t_vec)),
        vector_to_numpy(std::move(y_vec)),
        vector_to_numpy(std::move(yp_vec)),
        vector_to_numpy_3d(std::move(yS_vec), arg_sens0, arg_sens1, arg_sens2),
        vector_to_numpy_3d(std::move(ypS_vec), 
                           save_hermite ? arg_sens0 : 0, arg_sens1, arg_sens2),
        vector_to_numpy(std::move(yterm_vec))
      );
    }

private:
    int flag = 0;
    std::vector<sunrealtype> t_vec;
    std::vector<sunrealtype> y_vec;
    std::vector<sunrealtype> yp_vec;
    std::vector<sunrealtype> yS_vec;
    std::vector<sunrealtype> ypS_vec;
    std::vector<sunrealtype> yterm_vec;
    ptrdiff_t arg_sens0 = 0;
    ptrdiff_t arg_sens1 = 0;
    ptrdiff_t arg_sens2 = 0;
    bool save_hermite = false;
};

#endif // PYBAMM_IDAKLU_SOLUTION_DATA_HPP
