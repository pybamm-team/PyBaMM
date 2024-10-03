#include "IDAKLUSolverGroup.hpp"
#include <omp.h>
#include <optional>

std::vector<Solution> IDAKLUSolverGroup::solve(
    np_array t_eval_np,
    np_array t_interp_np,
    np_array y0_np,
    np_array yp0_np,
    np_array inputs) {
  DEBUG("IDAKLUSolverGroup::solve");

  // If t_interp is empty, save all adaptive steps
  bool save_adaptive_steps =  t_interp_np.size() == 0;

  const realtype* t_eval_begin = t_eval_np.data();
  const realtype* t_eval_end = t_eval_begin + t_eval_np.size();
  const realtype* t_interp_begin = t_interp_np.data();
  const realtype* t_interp_end = t_interp_begin + t_interp_np.size();

  // Process the time inputs
  // 1. Get the sorted and unique t_eval vector
  auto const t_eval = makeSortedUnique(t_eval_begin, t_eval_end);

  // 2.1. Get the sorted and unique t_interp vector
  auto const t_interp_unique_sorted = makeSortedUnique(t_interp_begin, t_interp_end);

  // 2.2 Remove the t_eval values from t_interp
  auto const t_interp_setdiff = setDiff(t_interp_unique_sorted.begin(), t_interp_unique_sorted.end(), t_eval_begin, t_eval_end);

  // 2.3 Finally, get the sorted and unique t_interp vector with t_eval values removed
  auto const t_interp = makeSortedUnique(t_interp_setdiff.begin(), t_interp_setdiff.end());

  int const number_of_evals = t_eval.size();
  int const number_of_interps = t_interp.size();

  // setDiff removes entries of t_interp that overlap with
  // t_eval, so we need to check if we need to interpolate any unique points.
  // This is not the same as save_adaptive_steps since some entries of t_interp
  // may be removed by setDiff
  bool save_interp_steps = number_of_interps > 0;

  // 3. Check if the timestepping entries are valid
  if (number_of_evals < 2) {
    throw std::invalid_argument(
      "t_eval must have at least 2 entries"
    );
  } else if (save_interp_steps) {
    if (t_interp.front() < t_eval.front()) {
      throw std::invalid_argument(
        "t_interp values must be greater than the smallest t_eval value: "
        + std::to_string(t_eval.front())
      );
    } else if (t_interp.back() > t_eval.back()) {
      throw std::invalid_argument(
        "t_interp values must be less than the greatest t_eval value: "
        + std::to_string(t_eval.back())
      );
    }
  }

  auto n_coeffs = number_of_states + number_of_parameters * number_of_states;

  // check y0 and yp0 and inputs have the correct dimensions
  if (y0_np.ndim() != 2)
    throw std::domain_error("y0 has wrong number of dimensions. Expected 2 but got " + std::to_string(y0_np.ndim()));
  if (yp0_np.ndim() != 2)
    throw std::domain_error("yp0 has wrong number of dimensions. Expected 2 but got " + std::to_string(yp0_np.ndim()));
  if (inputs.ndim() != 2)
    throw std::domain_error("inputs has wrong number of dimensions. Expected 2 but got " + std::to_string(inputs.ndim()));

  auto number_of_groups = y0_np.shape()[0];

  // check y0 and yp0 and inputs have the correct shape
  if (y0_np.shape()[1] != n_coeffs)
    throw std::domain_error(
      "y0 has wrong number of cols. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(y0_np.shape()[1]));

  if (yp0_np.shape()[1] != n_coeffs)
    throw std::domain_error(
      "yp0 has wrong number of cols. Expected " + std::to_string(n_coeffs) +
      " but got " + std::to_string(yp0_np.shape()[1]));

  if (yp0_np.shape()[0] != number_of_groups)
    throw std::domain_error(
      "yp0 has wrong number of rows. Expected " + std::to_string(number_of_groups) +
      " but got " + std::to_string(yp0_np.shape()[0]));

  if (inputs.shape()[0] != number_of_groups)
    throw std::domain_error(
      "inputs has wrong number of rows. Expected " + std::to_string(number_of_groups) +
      " but got " + std::to_string(inputs.shape()[0]));

  const std::size_t solves_per_thread = number_of_groups / m_solvers.size();
  const std::size_t remainder_solves = number_of_groups % m_solvers.size();

  const realtype *y0 = y0_np.data();
  const realtype *yp0 = yp0_np.data();
  const realtype *inputs_data = inputs.data();

  std::vector<SolutionData> results(number_of_groups);

  std::optional<std::exception> exception;

  omp_set_num_threads(m_solvers.size());
  #pragma omp parallel for
  for (int i = 0; i < m_solvers.size(); i++) {
    try {
      for (int j = 0; j < solves_per_thread; j++) {
        const std::size_t index = i * solves_per_thread + j;
        const realtype *y = y0 + index * y0_np.shape(1);
        const realtype *yp = yp0 + index * yp0_np.shape(1);
        const realtype *input = inputs_data + index * inputs.shape(1);
        results[index] = m_solvers[i]->solve(t_eval, t_interp, y, yp, input, save_adaptive_steps, save_interp_steps);
      }
    } catch (std::exception &e) {
      // If an exception is thrown, we need to catch it and rethrow it outside the parallel region
      #pragma omp critical
      {
        exception = e;
      }
    }
  }

  if (exception.has_value()) {
    py::set_error(PyExc_ValueError, exception->what());
    throw py::error_already_set();
  }

  for (int i = 0; i < remainder_solves; i++) {
    const std::size_t index = number_of_groups - remainder_solves + i;
    const realtype *y = y0 + index * y0_np.shape(1);
    const realtype *yp = yp0 + index * yp0_np.shape(1);
    const realtype *input = inputs_data + index * inputs.shape(1);
    results[index] = m_solvers[i]->solve(t_eval, t_interp, y, yp, input, save_adaptive_steps, save_interp_steps);
  }

  // create solutions (needs to be serial as we're using the Python GIL)
  std::vector<Solution> solutions(number_of_groups);
  for (int i = 0; i < number_of_groups; i++) {
    solutions[i] = results[i].generate_solution();
  }
  return solutions;
}
