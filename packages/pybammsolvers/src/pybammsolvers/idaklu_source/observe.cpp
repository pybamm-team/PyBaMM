#include "observe.hpp"

int _setup_len_spatial(const std::vector<int>& shape) {
    // Calculate the product of all dimensions except the last (spatial dimensions)
    int size_spatial = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        size_spatial *= shape[i];
    }

    if (size_spatial == 0 || shape.back() == 0) {
        throw std::invalid_argument("output array must have at least one element");
    }

    return size_spatial;
}

// Coupled observe and Hermite interpolation of variables
class HermiteInterpolator {
public:
    HermiteInterpolator(const py::detail::unchecked_reference<sunrealtype, 1>& t,
                       const py::detail::unchecked_reference<sunrealtype, 2>& y,
                       const py::detail::unchecked_reference<sunrealtype, 2>& yp)
        : t(t), y(y), yp(yp) {}

    void compute_knots(const size_t j,
                       vector<sunrealtype>& c,
                       vector<sunrealtype>& d,
                       const vector<casadi_int>& active_y) const {
        // Must be called after compute_knots
        const sunrealtype h_full = t(j + 1) - t(j);
        const sunrealtype inv_h = 1.0 / h_full;
        const sunrealtype inv_h2 = inv_h * inv_h;
        const sunrealtype inv_h3 = inv_h2 * inv_h;

        for (size_t k = 0; k < active_y.size(); ++k) {
            const auto i = active_y[k];
            sunrealtype y_ij = y(i, j);
            sunrealtype yp_ij = yp(i, j);
            sunrealtype y_ijp1 = y(i, j + 1);
            sunrealtype yp_ijp1 = yp(i, j + 1);

            c[k] = 3.0 * (y_ijp1 - y_ij) * inv_h2 - (2.0 * yp_ij + yp_ijp1) * inv_h;
            d[k] = 2.0 * (y_ij - y_ijp1) * inv_h3 + (yp_ij + yp_ijp1) * inv_h2;
        }
    }

    void interpolate(vector<sunrealtype>& entries,
                     sunrealtype t_interp,
                     const size_t j,
                     vector<sunrealtype>& c,
                     vector<sunrealtype>& d,
                     const vector<casadi_int>& active_y) const {
        // Must be called after compute_knots
        const sunrealtype h = t_interp - t(j);
        const sunrealtype h2 = h * h;
        const sunrealtype h3 = h2 * h;

        for (size_t k = 0; k < active_y.size(); ++k) {
            const auto i = active_y[k];
            sunrealtype y_ij = y(i, j);
            sunrealtype yp_ij = yp(i, j);
            entries[i] = y_ij + yp_ij * h + c[k] * h2 + d[k] * h3;
        }
    }

private:
    const py::detail::unchecked_reference<sunrealtype, 1>& t;
    const py::detail::unchecked_reference<sunrealtype, 2>& y;
    const py::detail::unchecked_reference<sunrealtype, 2>& yp;
};

class TimeSeriesInterpolator {
public:
    TimeSeriesInterpolator(const np_array_realtype& _t_interp,
                           const vector<np_array_realtype>& _ts_data,
                           const vector<np_array_realtype>& _ys_data,
                           const vector<np_array_realtype>& _yps_data,
                           const vector<np_array_realtype>& _inputs,
                           const vector<CasadiFuncData>& _func_data,
                           sunrealtype* _entries,
                           const int _size_spatial)
        : t_interp_np(_t_interp), ts_data_np(_ts_data), ys_data_np(_ys_data),
          yps_data_np(_yps_data), inputs_np(_inputs), func_data(_func_data),
          entries(_entries), size_spatial(_size_spatial) {}

    void process() {
        auto t_interp = t_interp_np.unchecked<1>();
        ssize_t i_interp = 0;
        int i_entries = 0;
        const ssize_t N_interp = t_interp.size();

        // Main processing within bounds
        process_within_bounds(i_interp, i_entries, t_interp, N_interp);

        // Extrapolation for remaining points
        if (i_interp < N_interp) {
            extrapolate_remaining(i_interp, i_entries, t_interp, N_interp);
        }
    }

    void process_within_bounds(
            ssize_t& i_interp,
            int& i_entries,
            const py::detail::unchecked_reference<sunrealtype, 1>& t_interp,
            const ssize_t N_interp
        ) {
        for (size_t i = 0; i < ts_data_np.size(); i++) {
            const auto& t_data = ts_data_np[i].unchecked<1>();
            // Continue if there is no data
            if (t_data.size() == 0) {
                continue;
            }

            const sunrealtype t_data_final = t_data(t_data.size() - 1);
            sunrealtype t_interp_next = t_interp(i_interp);
            // Continue if the next interpolation point is beyond the final data point
            if (t_interp_next > t_data_final) {
                continue;
            }

            const auto& y_data = ys_data_np[i].unchecked<2>();
            const auto& yp_data = yps_data_np[i].unchecked<2>();
            const auto input = inputs_np[i].data();
            const auto& fd = func_data[i];
            const auto func = *fd.func;
            const auto& active_y = fd.active_y;

            resize_arrays(y_data.shape(0), fd.func, active_y.size());
            args[1] = y_buffer.data();
            args[2] = input;

            ssize_t j = 0;
            ssize_t j_prev = -1;
            const auto itp = HermiteInterpolator(t_data, y_data, yp_data);
            while (t_interp_next <= t_data_final) {
                for (; j < t_data.size() - 2; ++j) {
                    if (t_data(j) <= t_interp_next && t_interp_next <= t_data(j + 1)) {
                        break;
                    }
                }

                if (j != j_prev) {
                    // Compute c and d for the new interval
                    itp.compute_knots(j, c, d, active_y);
                }

                itp.interpolate(y_buffer, t_interp(i_interp), j, c, d, active_y);

                args[0] = &t_interp(i_interp);
                results[0] = &entries[i_entries];
                func(args.data(), results.data(), iw.data(), w.data(), 0);

                ++i_interp;
                if (i_interp == N_interp) {
                    return;
                }
                t_interp_next = t_interp(i_interp);
                i_entries += size_spatial;
                j_prev = j;
            }
        }
    }

    void extrapolate_remaining(
            ssize_t& i_interp,
            int& i_entries,
            const py::detail::unchecked_reference<sunrealtype, 1>& t_interp,
            const ssize_t N_interp
        ) {
        const auto& t_data = ts_data_np.back().unchecked<1>();
        const auto& y_data = ys_data_np.back().unchecked<2>();
        const auto& yp_data = yps_data_np.back().unchecked<2>();
        const auto input = inputs_np.back().data();
        const auto& fd = func_data.back();
        const auto func = *fd.func;
        const auto& active_y = fd.active_y;
        const ssize_t j = t_data.size() - 2;

        resize_arrays(y_data.shape(0), fd.func, active_y.size());
        args[1] = y_buffer.data();
        args[2] = input;

        const auto itp = HermiteInterpolator(t_data, y_data, yp_data);
        itp.compute_knots(j, c, d, active_y);

        for (; i_interp < N_interp; ++i_interp) {
            const sunrealtype t_interp_next = t_interp(i_interp);
            itp.interpolate(y_buffer, t_interp_next, j, c, d, active_y);

            args[0] = &t_interp_next;
            results[0] = &entries[i_entries];
            func(args.data(), results.data(), iw.data(), w.data(), 0);

            i_entries += size_spatial;
        }
    }

    void resize_arrays(const int M,
                       std::shared_ptr<const casadi::Function> func,
                       const size_t n_active) {
        args.resize(func->sz_arg());
        results.resize(func->sz_res());
        iw.resize(func->sz_iw());
        w.resize(func->sz_w());
        if (static_cast<int>(y_buffer.size()) < M) {
            y_buffer.resize(M);
        }
        if (c.size() < n_active) {
            c.resize(n_active);
            d.resize(n_active);
        }
    }

private:
    const np_array_realtype& t_interp_np;
    const vector<np_array_realtype>& ts_data_np;
    const vector<np_array_realtype>& ys_data_np;
    const vector<np_array_realtype>& yps_data_np;
    const vector<np_array_realtype>& inputs_np;
    const vector<CasadiFuncData>& func_data;
    sunrealtype* entries;
    const int size_spatial;
    vector<sunrealtype> c;
    vector<sunrealtype> d;
    vector<sunrealtype> y_buffer;
    vector<const sunrealtype*> args;
    vector<sunrealtype*> results;
    vector<casadi_int> iw;
    vector<sunrealtype> w;
};

// Observe the raw data
class TimeSeriesProcessor {
public:
    TimeSeriesProcessor(const vector<np_array_realtype>& _ts,
                        const vector<np_array_realtype>& _ys,
                        const vector<np_array_realtype>& _inputs,
                        const vector<CasadiFuncData>& _func_data,
                        sunrealtype* _entries,
                        const bool _is_f_contiguous,
                        const int _size_spatial)
        : ts(_ts), ys(_ys), inputs(_inputs), func_data(_func_data),
          entries(_entries), is_f_contiguous(_is_f_contiguous), size_spatial(_size_spatial) {}

    void process() {
        int i_entries = 0;
        for (size_t i = 0; i < ts.size(); i++) {
            const auto& t = ts[i].unchecked<1>();
            if (t.size() == 0) {
                continue;
            }
            const auto& y = ys[i].unchecked<2>();
            const auto input = inputs[i].data();
            const auto& fd = func_data[i];
            const auto func = *fd.func;
            const auto& active_y = fd.active_y;

            resize_arrays(y.shape(0), fd.func, active_y);
            args[2] = input;

            for (size_t j = 0; j < t.size(); j++) {
                const sunrealtype t_val = t(j);
                const sunrealtype* y_val;
                if (is_f_contiguous) {
                    y_val = &y(0, j);
                } else {
                    copy_to_buffer(y_buffer, y, j, active_y);
                    y_val = y_buffer.data();
                }

                args[0] = &t_val;
                args[1] = y_val;
                results[0] = &entries[i_entries];

                func(args.data(), results.data(), iw.data(), w.data(), 0);

                i_entries += size_spatial;
            }
        }
    }

private:
    void copy_to_buffer(
        vector<sunrealtype>& buffer,
        const py::detail::unchecked_reference<sunrealtype, 2>& y,
        size_t j,
        const vector<casadi_int>& active_y) {
        for (const auto i : active_y) {
            buffer[i] = y(i, j);
        }
    }

    void resize_arrays(const int M,
                       std::shared_ptr<const casadi::Function> func,
                       const vector<casadi_int>& active_y) {
        args.resize(func->sz_arg());
        results.resize(func->sz_res());
        iw.resize(func->sz_iw());
        w.resize(func->sz_w());
        if (!is_f_contiguous && static_cast<int>(y_buffer.size()) < M) {
            y_buffer.resize(M);
        }
    }

    const vector<np_array_realtype>& ts;
    const vector<np_array_realtype>& ys;
    const vector<np_array_realtype>& inputs;
    const vector<CasadiFuncData>& func_data;
    sunrealtype* entries;
    const bool is_f_contiguous;
    int size_spatial;
    vector<sunrealtype> y_buffer;
    vector<const sunrealtype*> args;
    vector<sunrealtype*> results;
    vector<casadi_int> iw;
    vector<sunrealtype> w;
};

const np_array_realtype observe_hermite_interp(
    const np_array_realtype& t_interp_np,
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& yps_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<std::string>& strings,
    const vector<int>& shape
) {
    const int size_spatial = _setup_len_spatial(shape);
    const auto& func_data = setup_casadi_funcs(strings);
    py::array_t<sunrealtype, py::array::f_style> out_array(shape);
    auto entries = out_array.mutable_data();

    TimeSeriesInterpolator(t_interp_np, ts_np, ys_np, yps_np, inputs_np, func_data, entries, size_spatial).process();

    return out_array;
}

const np_array_realtype observe(
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<std::string>& strings,
    const bool is_f_contiguous,
    const vector<int>& shape
) {
    const int size_spatial = _setup_len_spatial(shape);
    const auto& func_data = setup_casadi_funcs(strings);
    py::array_t<sunrealtype, py::array::f_style> out_array(shape);
    auto entries = out_array.mutable_data();

    TimeSeriesProcessor(ts_np, ys_np, inputs_np, func_data, entries, is_f_contiguous, size_spatial).process();

    return out_array;
}

static vector<casadi_int> all_y_indices(casadi_int ny) {
    vector<casadi_int> all(ny);
    std::iota(all.begin(), all.end(), 0);
    return all;
}

static vector<casadi_int> compute_active_y_indices(const casadi::Function& func) {
    // Find the dependence of the function on the y variables (if any)
    // so we only need to compute the Hermite knots at the active y indices
    const casadi_int ny = func.size1_in(1);
    // No y variables
    if (ny == 0) {
        return {};
    }

    const casadi_int y_arg_index = 1;

    // No y derivatives available -- assume dense y behavior
    if (!func.is_diff_in(y_arg_index)) {
        return all_y_indices(ny);
    }

    // Get the y-dependent indices
    casadi::Sparsity sp = func.jac_sparsity(0, y_arg_index, true);
    vector<casadi_int> cols = sp.get_col();
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

    return cols;
}

const vector<CasadiFuncData> setup_casadi_funcs(const vector<std::string>& strings) {
    std::unordered_map<std::string, size_t> function_cache;
    vector<CasadiFuncData> func_data(strings.size());

    for (size_t i = 0; i < strings.size(); ++i) {
        const std::string& str = strings[i];

        auto it = function_cache.find(str);
        if (it != function_cache.end()) {
            func_data[i] = func_data[it->second];
        } else {
            function_cache[str] = i;

            auto func = std::make_shared<casadi::Function>(
                casadi::Function::deserialize(str));

            vector<casadi_int> active_y = compute_active_y_indices(*func);

            func_data[i] = {std::move(func), std::move(active_y)};
        }
    }

    return func_data;
}
