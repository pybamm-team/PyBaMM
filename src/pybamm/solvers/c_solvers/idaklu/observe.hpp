#ifndef PYBAMM_CREATE_OBSERVE_HPP
#define PYBAMM_CREATE_OBSERVE_HPP

#include "IDAKLUSolverOpenMP_solvers.hpp"
#include <idas/idas.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include "common.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // For numpy support in pybind11
#include <casadi/core/function.hpp>

namespace py = pybind11;

/**
 * @brief Observe and Hermite interpolate ND variables
 */
const np_array_realtype observe_hermite_interp_ND(
    const np_array_realtype& t_interp,
    const vector<np_array_realtype>& ts,
    const vector<np_array_realtype>& ys,
    const vector<np_array_realtype>& yps,
    const vector<np_array_realtype>& inputs,
    const vector<std::string>& strings,
    const vector<int> sizes
);


/**
 * @brief Observe ND variables
 */
const np_array_realtype observe_ND(
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<std::string>& strings,
    const bool is_f_contiguous,
    const vector<int> sizes
);

const std::vector<std::shared_ptr<const casadi::Function>> setup_casadi_funcs(const std::vector<std::string>& strings);

int setup_observable(const vector<int>& sizes);

#endif // PYBAMM_CREATE_OBSERVE_HPP
