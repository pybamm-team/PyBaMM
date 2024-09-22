#ifndef PYBAMM_CREATE_OBSERVE_HPP
#define PYBAMM_CREATE_OBSERVE_HPP

#include "IDAKLUSolverOpenMP_solvers.hpp"
#include <idas/idas.h>
#include <memory>
#include <vector>
#include "common.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // For numpy support in pybind11
#include <casadi/core/function.hpp>

namespace py = pybind11;

void apply_copy(
    std::vector<double>& out,
    const py::detail::unchecked_reference<double, 2>& y,
    const size_t j
);

/**
 * @brief Loops over the solution and generates the observable output
 */
void process_time_series(
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<const casadi::Function*>& funcs,
    double* out,
    const bool is_f_contiguous,
    const int len
);

void hermite_interp(
    std::vector<double>& out,
    const double t_interp,
    const py::detail::unchecked_reference<double, 1>& t,
    const py::detail::unchecked_reference<double, 2>& y,
    const py::detail::unchecked_reference<double, 2>& yp,
    const size_t j
);

const double hermite_interp_scalar(
    const double t_interp,
    const double t_j,
    const double t_jp1,
    const double y_j,
    const double y_jp1,
    const double yp_j,
    const double yp_jp1
);

void compute_c_d(
    std::vector<double>& c_out,
    std::vector<double>& d_out,
    const py::detail::unchecked_reference<double, 1>& t,
    const py::detail::unchecked_reference<double, 2>& y,
    const py::detail::unchecked_reference<double, 2>& yp,
    const size_t j
);

void apply_hermite_interp(
    std::vector<double>& out,
    const double t_interp,
    const double t_j,
    const py::detail::unchecked_reference<double, 2>& y,
    const py::detail::unchecked_reference<double, 2>& yp,
    const std::vector<double>& c,
    const std::vector<double>& d,
    const size_t j
);

/**
 * @brief Loops over the solution and generates the observable output
 */
void process_and_interp_sorted_time_series(
    const np_array_realtype& t_interp_np,
    const vector<np_array_realtype>& ts_data_np,
    const vector<np_array_realtype>& ys_data_np,
    const vector<np_array_realtype>& yps_data_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<const casadi::Function*>& funcs,
    double* out,
    const int len
);

const int _setup_observables(const vector<int>& sizes);


/**
 * @brief Observe and Hermite interpolate ND variables
 */
const py::array_t<double> observe_hermite_interp_ND(
    const np_array_realtype& t_interp_np,
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& yps_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<const casadi::Function*>& funcs,
    const vector<int> sizes
);


/**
 * @brief Observe ND variables
 */
const py::array_t<double> observe_ND(
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<const casadi::Function*>& funcs,
    const bool is_f_contiguous,
    const vector<int> sizes
);

#endif // PYBAMM_CREATE_OBSERVE_HPP
