#include "SolutionData.hpp"

Solution SolutionData::generate_solution() {
  // Transfer ownership from unique_ptr to Python capsules
  sunrealtype* t_ptr = t_return.release();
  py::capsule free_t_when_done(
    t_ptr,
    [](void *f) {
      delete[] reinterpret_cast<sunrealtype*>(f);
    }
  );

  np_array t_ret = np_array(
    number_of_timesteps,
    t_ptr,
    free_t_when_done
  );

  sunrealtype* y_ptr = y_return.release();
  py::capsule free_y_when_done(
    y_ptr,
    [](void *f) {
      delete[] reinterpret_cast<sunrealtype*>(f);
    }
  );

  np_array y_ret = np_array(
    number_of_timesteps * length_of_return_vector,
    y_ptr,
    free_y_when_done
  );

  sunrealtype* yp_ptr = yp_return.release();
  py::capsule free_yp_when_done(
    yp_ptr,
    [](void *f) {
      delete[] reinterpret_cast<sunrealtype*>(f);
    }
  );

  np_array yp_ret = np_array(
    (save_hermite ? 1 : 0) * number_of_timesteps * length_of_return_vector,
    yp_ptr,
    free_yp_when_done
  );

  sunrealtype* yS_ptr = yS_return.release();
  py::capsule free_yS_when_done(
    yS_ptr,
    [](void *f) {
      delete[] reinterpret_cast<sunrealtype*>(f);
    }
  );

  np_array yS_ret = np_array(
    std::vector<ptrdiff_t> {
      arg_sens0,
      arg_sens1,
      arg_sens2
    },
    yS_ptr,
    free_yS_when_done
  );

  sunrealtype* ypS_ptr = ypS_return.release();
  py::capsule free_ypS_when_done(
    ypS_ptr,
    [](void *f) {
      delete[] reinterpret_cast<sunrealtype*>(f);
    }
  );

  np_array ypS_ret = np_array(
    std::vector<ptrdiff_t> {
      (save_hermite ? 1 : 0) * arg_sens0,
      arg_sens1,
      arg_sens2
    },
    ypS_ptr,
    free_ypS_when_done
  );

  sunrealtype* yterm_ptr = yterm_return.release();
  py::capsule free_yterm_when_done(
    yterm_ptr,
    [](void *f) {
      delete[] reinterpret_cast<sunrealtype*>(f);
    }
  );

  np_array y_term = np_array(
    length_of_final_sv_slice,
    yterm_ptr,
    free_yterm_when_done
  );

  return Solution(flag, t_ret, y_ret, yp_ret, yS_ret, ypS_ret, y_term);
}
