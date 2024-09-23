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

/**
 * @brief Observe and Hermite interpolate ND variables
 */
const py::array_t<double> observe_hermite_interp_ND(
    const np_array_realtype& t_interp,
    const vector<np_array_realtype>& ts,
    const vector<np_array_realtype>& ys,
    const vector<np_array_realtype>& yps,
    const vector<np_array_realtype>& inputs,
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

int setup_observable(const vector<int>& sizes);

#endif // PYBAMM_CREATE_OBSERVE_HPP
