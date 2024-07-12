#include "CasadiSolverGroup.hpp"
#include <omp.h>

std::vector<Solution> CasadiSolverGroup::solve(np_array t_np, np_array y0_np, np_array yp0_np, np_array inputs) {
  auto n_coeffs = number_of_states + number_of_parameters * number_of_states;

  if (y0_np.ndim() != 2)
    throw std::domain_error("y0 has wrong number of dimensions. Expected 2 but got " + std::to_string(y0_np.ndim()));
  if (yp0_np.ndim() != 2)
    throw std::domain_error("yp0 has wrong number of dimensions. Expected 2 but got " + std::to_string(yp0_np.ndim()));
  if (inputs.ndim() != 2)
    throw std::domain_error("inputs has wrong number of dimensions. Expected 2 but got " + std::to_string(inputs.ndim()));

  auto n_groups = y0_np.shape()[0];

  if (y0_np.shape()[1] != n_coeffs)
    throw std::domain_error(
      "y0 has wrong number of cols. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(y0_np.shape()[1]));

  if (yp0_np.shape()[1] != n_coeffs)
    throw std::domain_error(
      "yp0 has wrong number of cols. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(yp0_np.shape()[1]));

  if (yp0_np.shape()[0] != n_groups)
    throw std::domain_error(
      "yp0 has wrong number of rows. Expected " + std::to_string(n_groups) +
      " but got " + std::to_string(yp0_np.shape()[0]));

  if (inputs.shape()[0] != n_groups)
    throw std::domain_error(
      "inputs has wrong number of rows. Expected " + std::to_string(n_groups) +
      " but got " + std::to_string(inputs.shape()[0]));


  const int number_of_timesteps = t_np.shape(0);

  // set return vectors
  std::vector<int> retval_returns(n_groups);
  std::vector<int> t_i_returns(n_groups);
  std::vector<realtype *> t_returns(n_groups);
  std::vector<realtype *> y_returns(n_groups);
  std::vector<realtype *> yS_returns(n_groups);
  for (int i = 0; i < n_groups; i++) {

    t_returns[i] = new realtype[number_of_timesteps];
    y_returns[i] = new realtype[number_of_timesteps *
                                length_of_return_vector];
    yS_returns[i] = new realtype[number_of_parameters *
                                 number_of_timesteps *
                                 length_of_return_vector];
  }


  const std::size_t solves_per_thread = n_groups / m_solvers.size();
  const std::size_t remainder_solves = n_groups % m_solvers.size();

  const realtype *t = t_np.data();
  const realtype *y0 = y0_np.data();
  const realtype *yp0 = yp0_np.data();
  const realtype *inputs_data = inputs.data();

  omp_set_num_threads(m_solvers.size());
  #pragma omp parallel for
  for (int i = 0; i < m_solvers.size(); i++) {
    for (int j = 0; j < solves_per_thread; j++) {
      const std::size_t index = i * solves_per_thread + j;
      const realtype *y = y0 + index * y0_np.shape(1);
      const realtype *yp = yp0 + index * yp0_np.shape(1);
      const realtype *input = inputs_data + index * inputs.shape(1);
      realtype *y_return = y_returns[index];
      realtype *yS_return = yS_returns[index];
      realtype *t_return = t_returns[index];
      int &t_i = t_i_returns[index];
      int &retval = retval_returns[index];
      m_solvers[i]->solve(t, number_of_timesteps, y, yp, input, length_of_return_vector, y_return, yS_return, t_return, t_i, retval);
    }
  }

  for (int i = 0; i < remainder_solves; i++) {
    const std::size_t index = n_groups - remainder_solves + i;
    const realtype *y = y0 + index * y0_np.shape(1);
    const realtype *yp = yp0 + index * yp0_np.shape(1);
    const realtype *input = inputs_data + index * inputs.shape(1);
    realtype *y_return = y_returns[index];
    realtype *yS_return = yS_returns[index];
    realtype *t_return = t_returns[index];
    int &t_i = t_i_returns[index];
    int &retval = retval_returns[index];
    m_solvers[i]->solve(t, number_of_timesteps, y, yp, input, length_of_return_vector, y_return, yS_return, t_return, t_i, retval);
  }

  // create solutions
  std::vector<Solution> solutions(n_groups);
  for (int i = 0; i < n_groups; i++) {
    int t_i = t_i_returns[i];
    int retval = retval_returns[i];
    realtype *t_return = t_returns[i];
    realtype *y_return = y_returns[i];
    realtype *yS_return = yS_returns[i];

    py::capsule free_t_when_done(
      t_return,
      [](void *f) {
        realtype *vect = reinterpret_cast<realtype *>(f);
        delete[] vect;
      }
    );
    py::capsule free_y_when_done(
      y_return,
      [](void *f) {
        realtype *vect = reinterpret_cast<realtype *>(f);
        delete[] vect;
      }
    );
    py::capsule free_yS_when_done(
      yS_return,
      [](void *f) {
        realtype *vect = reinterpret_cast<realtype *>(f);
        delete[] vect;
      }
    );

    np_array t_ret = np_array(
        t_i,
        &t_return[0],
        free_t_when_done
      );
      np_array y_ret = np_array(
        t_i * length_of_return_vector,
        &y_return[0],
        free_y_when_done
      );
      // Note: Ordering of vector is differnet if computing variables vs returning
      // the complete state vector
      np_array yS_ret;
      if (is_output_variables) {
        yS_ret = np_array(
          std::vector<ptrdiff_t> {
            number_of_timesteps,
            length_of_return_vector,
            number_of_parameters
          },
          &yS_return[0],
          free_yS_when_done
        );
      } else {
        yS_ret = np_array(
          std::vector<ptrdiff_t> {
            number_of_parameters,
            number_of_timesteps,
            length_of_return_vector
          },
          &yS_return[0],
          free_yS_when_done
        );
      }
      solutions[i] = Solution(retval, t_ret, y_ret, yS_ret);
  }

  return solutions;
}
