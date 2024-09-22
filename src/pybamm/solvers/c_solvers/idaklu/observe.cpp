#include "observe.hpp"

void hermite_interp(
    std::vector<double>& out,
    const double t_interp,
    const py::detail::unchecked_reference<double, 1>& t,
    const py::detail::unchecked_reference<double, 2>& y,
    const py::detail::unchecked_reference<double, 2>& yp,
    const size_t j
) {
    const double h = t_interp - t(j);
    const double h2 = h * h;
    const double h3 = h2 * h;

    const double h_full = t(j + 1) - t(j);
    const double inv_h = 1 / h_full;
    const double inv_h2 = inv_h * inv_h;
    const double inv_h3 = inv_h2 * inv_h;

    double c, d, y_ij, yp_ij, y_ijp1, yp_ijp1;

    for (size_t i = 0; i < out.size(); ++i) {
        y_ij = y(i, j);
        yp_ij = yp(i, j);
        y_ijp1 = y(i, j + 1);
        yp_ijp1 = yp(i, j + 1);

        c = 3 * (y_ijp1 - y_ij) * inv_h2 - (2 * yp_ij + yp_ijp1) * inv_h;
        d = 2 * (y_ij - y_ijp1) * inv_h3 + (yp_ij + yp_ijp1) * inv_h2;

        out[i] = y_ij + yp_ij * h + c * h2 + d * h3;
    }
}

void apply_copy(
    std::vector<double>& out,
    const py::detail::unchecked_reference<double, 2>& y,
    const size_t j
) {
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = y(i, j);
    }
}

void hermite_interp_no_y(
    std::vector<double>& out,
    const double t_interp,
    const py::detail::unchecked_reference<double, 1>& t,
    const py::detail::unchecked_reference<double, 2>& y,
    const py::detail::unchecked_reference<double, 2>& yp,
    const size_t j
) {
    // This begins from a copy of y, so we don't need to copy y
    // once again
    const double h = t_interp - t(j);
    const double h2 = h * h;
    const double h3 = h2 * h;

    const double h_full = t(j + 1) - t(j);
    const double inv_h = 1 / h_full;
    const double inv_h2 = inv_h * inv_h;
    const double inv_h3 = inv_h2 * inv_h;

    double c, d, y_ij, yp_ij, y_ijp1, yp_ijp1;

    for (size_t i = 0; i < out.size(); ++i) {
        y_ij = y(i, j);
        yp_ij = yp(i, j);
        y_ijp1 = y(i, j + 1);
        yp_ijp1 = yp(i, j + 1);

        c = 3 * (y_ijp1 - y_ij) * inv_h2 - (2 * yp_ij + yp_ijp1) * inv_h;
        d = 2 * (y_ij - y_ijp1) * inv_h3 + (yp_ij + yp_ijp1) * inv_h2;

        out[i] += yp_ij * h + c * h2 + d * h3;
    }
}

void compute_c_d(
    std::vector<double>& c_out,
    std::vector<double>& d_out,
    const py::detail::unchecked_reference<double, 1>& t,
    const py::detail::unchecked_reference<double, 2>& y,
    const py::detail::unchecked_reference<double, 2>& yp,
    const size_t j
) {
    const double h_full = t(j + 1) - t(j);
    const double inv_h = 1.0 / h_full;
    const double inv_h2 = inv_h * inv_h;
    const double inv_h3 = inv_h2 * inv_h;

    for (size_t i = 0; i < y.shape(0); ++i) {
        double y_ij = y(i, j);
        double yp_ij = yp(i, j);
        double y_ijp1 = y(i, j + 1);
        double yp_ijp1 = yp(i, j + 1);

        c_out[i] = 3.0 * (y_ijp1 - y_ij) * inv_h2 - (2.0 * yp_ij + yp_ijp1) * inv_h;
        d_out[i] = 2.0 * (y_ij - y_ijp1) * inv_h3 + (yp_ij + yp_ijp1) * inv_h2;
    }
}

void apply_hermite_interp(
    std::vector<double>& out,
    const double t_interp,
    const double t_j,
    const py::detail::unchecked_reference<double, 2>& y,
    const py::detail::unchecked_reference<double, 2>& yp,
    const std::vector<double>& c,
    const std::vector<double>& d,
    const size_t j
) {
    const double h = t_interp - t_j;  // h = t_interp - t(j)
    const double h2 = h * h;
    const double h3 = h2 * h;

    for (size_t i = 0; i < out.size(); ++i) {
        double y_ij = y(i, j);
        double yp_ij = yp(i, j);

        out[i] = y_ij + yp_ij * h + c[i] * h2 + d[i] * h3;
    }
}

const int _setup_observables(const vector<int>& sizes) {
    // Create a numpy array to manage the output
    if (sizes.size() == 0) {
        throw std::invalid_argument("sizes must have at least one element");
    }

    // Create a numpy array to manage the output
    int size_tot = 1;
    for (const auto& size : sizes) {
        size_tot *= size;
    }

    if (size_tot == 0) {
        throw std::invalid_argument("sizes must have at least one element");
    }

    return size_tot;
}


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
) {
    // Buffer for non-f-contiguous arrays
    vector<double> y_buffer;

    int count = 0;
    for (size_t i = 0; i < ts_np.size(); i++) {
        const auto& t_i = ts_np[i].unchecked<1>();
        const auto& y_i = ys_np[i].unchecked<2>();  // y_i is 2D
        const auto inputs_data_i = inputs_np[i].data();
        const auto func_i = *funcs[i];

        int M = y_i.shape(0);
        if (!is_f_contiguous && y_buffer.size() < M) {
            y_buffer.resize(M); // Resize the buffer
        }

        for (size_t j = 0; j < t_i.size(); j++) {
            const double t_ij = t_i(j);

            // Use a view of y_i
            if (!is_f_contiguous) {
                apply_copy(y_buffer, y_i, j);
            }
            const double* y_ij = is_f_contiguous ? &y_i(0, j) : y_buffer.data();

            // Prepare CasADi function arguments
            vector<const double*> args = { &t_ij, y_ij, inputs_data_i };
            vector<double*> results = { &out[count] };
            // Call the CasADi function with proper arguments
            (func_i)(args, results);

            count += len;
        }
    }
}

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
) {
    // y cache
    vector<double> y_interp;

    auto t_interp = t_interp_np.unchecked<1>();
    ssize_t i_interp = 0;
    int count = 0;

    auto t_interp_next = t_interp(0);
    vector<const double*> args;
    vector<double*> results;

    ssize_t N_data = 0;
    const ssize_t N_interp = t_interp.size();

    for (size_t i = 0; i < ts_data_np.size(); i++) {
        N_data += ts_data_np[i].size();
    }

    const bool cache_hermite_interp = N_interp > N_data;
    vector<double> hermite_c;
    vector<double> hermite_d;

    for (size_t i = 0; i < ts_data_np.size(); i++) {
        const auto& t_data_i = ts_data_np[i].unchecked<1>();
        const auto& y_data_i = ys_data_np[i].unchecked<2>();  // y_data_i is 2D
        const auto& yp_data_i = yps_data_np[i].unchecked<2>();  // yp_data_i is 2D
        const auto inputs_i = inputs_np[i].data();
        const auto func_i = *funcs[i];
        const double t_data_final = t_data_i(t_data_i.size() - 1);  // Access last element

        // Resize y_interp buffer to match the number of rows in y_data_i
        int M = y_data_i.shape(0);
        if (y_interp.size() < M) {
            y_interp.resize(M);
            if (cache_hermite_interp) {
                hermite_c.resize(M);
                hermite_d.resize(M);
            }
        }

        ssize_t j = 0;
        t_interp_next = t_interp(i_interp);
        while (i_interp < N_interp && t_interp(i_interp) <= t_data_final) {
            // Find the correct index j
            for (; j < t_data_i.size() - 2; ++j) {
                if (t_data_i(j) <= t_interp_next && t_interp_next <= t_data_i(j + 1)) {
                    break;
                }
            }
            const double t_data_start = t_data_i(j);
            const double t_data_next = t_data_i(j + 1);

            if (cache_hermite_interp) {
                compute_c_d(hermite_c, hermite_d, t_data_i, y_data_i, yp_data_i, j);
            }

            args = { &t_interp_next, y_interp.data(), inputs_i };

            // Perform Hermite interpolation for all valid t_interp values
            for (ssize_t k = 0; t_interp_next <= t_data_next; ++k) {
                if (k == 0 && t_interp_next == t_data_start) {
                    apply_copy(y_interp, y_data_i, j);
                } else if (cache_hermite_interp) {
                    apply_hermite_interp(y_interp, t_interp_next, t_data_start, y_data_i, yp_data_i, hermite_c, hermite_d, j);
                } else {
                    hermite_interp(y_interp, t_interp_next, t_data_i, y_data_i, yp_data_i, j);
                }

                // Prepare CasADi function arguments
                results = { &out[count] };

                // Call the CasADi function with the proper arguments
                func_i(args, results);

                count += len;
                ++i_interp;  // Move to the next time step for interpolation
                if (i_interp >= N_interp) {
                    return;
                }
                t_interp_next = t_interp(i_interp);
            }
        }
    }

    // Return if there are no more data points to interpolate
    if (i_interp == N_interp) {
        return;
    }

    // Extrapolate to the right
    const auto& t_data_i = ts_data_np[ts_data_np.size() - 1].unchecked<1>();
    const auto& y_data_i = ys_data_np[ys_data_np.size() - 1].unchecked<2>();  // y_data_i is 2D
    const auto& yp_data_i = yps_data_np[yps_data_np.size() - 1].unchecked<2>();  // yp_data_i is 2D
    const auto inputs_i = inputs_np[inputs_np.size() - 1].data();
    const auto func_i = *funcs[funcs.size() - 1];

    const ssize_t j = t_data_i.size() - 2;
    const double t_data_start = t_data_i(j);
    const double t_data_final = t_data_i(j + 1);

    // Resize y_interp buffer to match the number of rows in y_data_i
    int M = y_data_i.shape(0);
    if (y_interp.size() < M) {
        y_interp.resize(M);
        if (cache_hermite_interp) {
            hermite_c.resize(M);
            hermite_d.resize(M);
        }
    }

    // Find the number of steps within this interval
    args = { &t_interp_next, y_interp.data(), inputs_i };

    // Perform Hermite interpolation for all valid t_interp values
    for (; i_interp < N_interp; ++i_interp) {
        t_interp_next = t_interp(i_interp);

        if (cache_hermite_interp) {
            compute_c_d(hermite_c, hermite_d, t_data_i, y_data_i, yp_data_i, j);
            apply_hermite_interp(y_interp, t_interp_next, t_data_start, y_data_i, yp_data_i, hermite_c, hermite_d, j);
        } else {
            hermite_interp(y_interp, t_interp_next, t_data_i, y_data_i, yp_data_i, j);
        }

        // Prepare CasADi function arguments
        results = { &out[count] };

        // Call the CasADi function with the proper arguments
        func_i(args, results);

        count += len;
    }
}

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
) {
    const int size_tot = _setup_observables(sizes);

    py::array_t<double, py::array::f_style> out_array(sizes);
    auto out = out_array.mutable_data();

    process_and_interp_sorted_time_series(
        t_interp_np, ts_np, ys_np, yps_np, inputs_np, funcs, out, size_tot / sizes.back()
    );

    return out_array;
}

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
) {
    const int size_tot = _setup_observables(sizes);

    py::array_t<double, py::array::f_style> out_array(sizes);
    auto out = out_array.mutable_data();

    process_time_series(
        ts_np, ys_np, inputs_np, funcs, out, is_f_contiguous, size_tot / sizes.back()
    );

    return out_array;
}
