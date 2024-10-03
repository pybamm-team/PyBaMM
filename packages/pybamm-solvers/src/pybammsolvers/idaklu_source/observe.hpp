#ifndef PYBAMM_CREATE_OBSERVE_HPP
#define PYBAMM_CREATE_OBSERVE_HPP

#include <memory>
#include <unordered_map>
#include <string>
#include "common.hpp"
#include <casadi/core/function.hpp>
#include <vector>
using std::vector;

#if defined(_MSC_VER)
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
#endif

/**
 * @brief Observe and Hermite interpolate ND variables
 */
const np_array_realtype observe_hermite_interp(
    const np_array_realtype& t_interp,
    const vector<np_array_realtype>& ts,
    const vector<np_array_realtype>& ys,
    const vector<np_array_realtype>& yps,
    const vector<np_array_realtype>& inputs,
    const vector<std::string>& strings,
    const vector<int>& shape
);


/**
 * @brief Observe ND variables
 */
const np_array_realtype observe(
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<std::string>& strings,
    const bool is_f_contiguous,
    const vector<int>& shape
);

const vector<std::shared_ptr<const casadi::Function>> setup_casadi_funcs(const vector<std::string>& strings);

int _setup_len_spatial(const vector<int>& shape);

#endif // PYBAMM_CREATE_OBSERVE_HPP
