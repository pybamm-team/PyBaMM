#include "IDAKLUSolverGroup.hpp"
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <optional>

std::vector<Solution> IDAKLUSolverGroup::solve(
    np_array t_eval_np,
    np_array t_interp_np,
    np_array y0_np,
    np_array yp0_np,
    np_array inputs,
    py::object logger) {
  DEBUG("IDAKLUSolverGroup::solve");

  // If t_interp is empty, save all adaptive steps
  bool save_adaptive_steps =  t_interp_np.size() == 0;

  const sunrealtype* t_eval_begin = t_eval_np.data();
  const sunrealtype* t_eval_end = t_eval_begin + t_eval_np.size();
  const sunrealtype* t_interp_begin = t_interp_np.data();
  const sunrealtype* t_interp_end = t_interp_begin + t_interp_np.size();

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
  }
  if (save_interp_steps) {
    if (t_interp.front() < t_eval.front()) {
      throw std::invalid_argument(
        "t_interp values must be greater than the smallest t_eval value: "
        + std::to_string(t_eval.front())
      );
    }
    if (t_interp.back() > t_eval.back()) {
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

  const sunrealtype *y0 = y0_np.data();
  const sunrealtype *yp0 = yp0_np.data();
  const sunrealtype *inputs_data = inputs.data();

  std::vector<SolutionData> results(number_of_groups);

  // One slot per input set, so the rethrow below can name every set that failed
  std::vector<std::optional<std::string>> errors(number_of_groups);
  // Python exceptions carry their own type, which the string path below loses
  std::exception_ptr python_exception;

  // Also records this thread as the GIL holder, so each solver knows whether it
  // may log directly or must buffer until flush_logs().
  set_loggers(logger);

  // num_threads() scopes the team to this region, unlike the process-wide
  // omp_set_num_threads; the runtime may still start fewer threads than asked.
  // cppcheck-suppress unreadVariable
  const int team_size = std::max<int>(1, std::min<int>(m_solvers.size(), number_of_groups));
  // A plain atomic, not schedule(dynamic): the macOS wheels' libomp lacks
  // __kmpc_dispatch_deinit, and MSVC needs -openmp:llvm for omp atomic capture.
  std::atomic<int> next_group{1};
  std::atomic<bool> interrupted{false};
  #pragma omp parallel num_threads(team_size)
  {
    // Thread 0 is the calling thread, which holds the GIL, so it always takes a
    // set and streams that set's diagnostics instead of buffering them.
    const int thread = omp_get_thread_num();
    int i = thread == 0 ? 0 : next_group.fetch_add(1, std::memory_order_relaxed);
    while (i < number_of_groups && !interrupted.load(std::memory_order_relaxed)) {
      const sunrealtype *y = y0 + i * y0_np.shape(1);
      const sunrealtype *yp = yp0 + i * yp0_np.shape(1);
      const sunrealtype *input = inputs_data + i * inputs.shape(1);
      try {
        results[i] = m_solvers[thread]->solve(
          t_eval, t_interp, y, yp, input, save_adaptive_steps, save_interp_steps);
      } catch (py::error_already_set &) {
        // A Python exception such as KeyboardInterrupt stops the whole sweep
        interrupted.store(true, std::memory_order_relaxed);
        #pragma omp critical
        {
          if (!python_exception) {
            python_exception = std::current_exception();
          }
        }
      } catch (std::exception &e) {
        errors[i] = e.what();
      }
      i = next_group.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Drain before the rethrow below, so a solve that throws still emits its log
  flush_logs();

  if (python_exception) {
    std::rethrow_exception(python_exception);
  }

  std::string failures;
  for (int i = 0; i < number_of_groups; i++) {
    if (!errors[i].has_value()) {
      continue;
    }
    if (!failures.empty()) {
      failures += "; ";
    }
    failures += "input set " + std::to_string(i) + ": " + *errors[i];
  }
  if (!failures.empty()) {
    py::set_error(PyExc_ValueError, failures.c_str());
    throw py::error_already_set();
  }

  // create solutions (needs to be serial as we're using the Python GIL)
  std::vector<Solution> solutions(number_of_groups);
  for (int i = 0; i < number_of_groups; i++) {
    solutions[i] = results[i].generate_solution();
  }
  return solutions;
}

void IDAKLUSolverGroup::flush_logs() {
  for (const auto& solver : m_solvers) {
    solver->flush_log();
  }
}

void IDAKLUSolverGroup::set_loggers(py::object logger) {
  for (const auto& solver : m_solvers) {
    solver->set_logger(logger);
  }
}
