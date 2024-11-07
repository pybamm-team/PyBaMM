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
    HermiteInterpolator(const py::detail::unchecked_reference<realtype, 1>& t,
                       const py::detail::unchecked_reference<realtype, 2>& y,
                       const py::detail::unchecked_reference<realtype, 2>& yp)
        : t(t), y(y), yp(yp) {}

    void compute_knots(const size_t j, vector<realtype>& c, vector<realtype>& d) const {
        // Called at the start of each interval
        const realtype h_full = t(j + 1) - t(j);
        const realtype inv_h = 1.0 / h_full;
        const realtype inv_h2 = inv_h * inv_h;
        const realtype inv_h3 = inv_h2 * inv_h;

        for (size_t i = 0; i < y.shape(0); ++i) {
            realtype y_ij = y(i, j);
            realtype yp_ij = yp(i, j);
            realtype y_ijp1 = y(i, j + 1);
            realtype yp_ijp1 = yp(i, j + 1);

            c[i] = 3.0 * (y_ijp1 - y_ij) * inv_h2 - (2.0 * yp_ij + yp_ijp1) * inv_h;
            d[i] = 2.0 * (y_ij - y_ijp1) * inv_h3 + (yp_ij + yp_ijp1) * inv_h2;
        }
    }

    void interpolate(vector<realtype>& entries,
    realtype t_interp,
    const size_t j,
    vector<realtype>& c,
    vector<realtype>& d) const {
        // Must be called after compute_knots
        const realtype h = t_interp - t(j);
        const realtype h2 = h * h;
        const realtype h3 = h2 * h;

        for (size_t i = 0; i < entries.size(); ++i) {
            realtype y_ij = y(i, j);
            realtype yp_ij = yp(i, j);
            entries[i] = y_ij + yp_ij * h + c[i] * h2 + d[i] * h3;
        }
    }

private:
    const py::detail::unchecked_reference<realtype, 1>& t;
    const py::detail::unchecked_reference<realtype, 2>& y;
    const py::detail::unchecked_reference<realtype, 2>& yp;
};

class TimeSeriesInterpolator {
public:
    TimeSeriesInterpolator(const np_array_realtype& _t_interp,
                           const vector<np_array_realtype>& _ts_data,
                           const vector<np_array_realtype>& _ys_data,
                           const vector<np_array_realtype>& _yps_data,
                           const vector<np_array_realtype>& _inputs,
                           const vector<std::shared_ptr<const casadi::Function>>& _funcs,
                           realtype* _entries,
                           const int _size_spatial)
        : t_interp_np(_t_interp), ts_data_np(_ts_data), ys_data_np(_ys_data),
          yps_data_np(_yps_data), inputs_np(_inputs), funcs(_funcs),
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
            const py::detail::unchecked_reference<realtype, 1>& t_interp,
            const ssize_t N_interp
        ) {
        for (size_t i = 0; i < ts_data_np.size(); i++) {
            const auto& t_data = ts_data_np[i].unchecked<1>();
            // Continue if there is no data
            if (t_data.size() == 0) {
                continue;
            }

            const realtype t_data_final = t_data(t_data.size() - 1);
            realtype t_interp_next = t_interp(i_interp);
            // Continue if the next interpolation point is beyond the final data point
            if (t_interp_next > t_data_final) {
                continue;
            }

            const auto& y_data = ys_data_np[i].unchecked<2>();
            const auto& yp_data = yps_data_np[i].unchecked<2>();
            const auto input = inputs_np[i].data();
            const auto func = *funcs[i];

            resize_arrays(y_data.shape(0), funcs[i]);
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
                    itp.compute_knots(j, c, d);
                }

                itp.interpolate(y_buffer, t_interp(i_interp), j, c, d);

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
            const py::detail::unchecked_reference<realtype, 1>& t_interp,
            const ssize_t N_interp
        ) {
        const auto& t_data = ts_data_np.back().unchecked<1>();
        const auto& y_data = ys_data_np.back().unchecked<2>();
        const auto& yp_data = yps_data_np.back().unchecked<2>();
        const auto input = inputs_np.back().data();
        const auto func = *funcs.back();
        const ssize_t j = t_data.size() - 2;

        resize_arrays(y_data.shape(0), funcs.back());
        args[1] = y_buffer.data();
        args[2] = input;

        const auto itp = HermiteInterpolator(t_data, y_data, yp_data);
        itp.compute_knots(j, c, d);

        for (; i_interp < N_interp; ++i_interp) {
            const realtype t_interp_next = t_interp(i_interp);
            itp.interpolate(y_buffer, t_interp_next, j, c, d);

            args[0] = &t_interp_next;
            results[0] = &entries[i_entries];
            func(args.data(), results.data(), iw.data(), w.data(), 0);

            i_entries += size_spatial;
        }
    }

    void resize_arrays(const int M, std::shared_ptr<const casadi::Function> func) {
        args.resize(func->sz_arg());
        results.resize(func->sz_res());
        iw.resize(func->sz_iw());
        w.resize(func->sz_w());
        if (y_buffer.size() < M) {
            y_buffer.resize(M);
            c.resize(M);
            d.resize(M);
        }
    }

private:
    const np_array_realtype& t_interp_np;
    const vector<np_array_realtype>& ts_data_np;
    const vector<np_array_realtype>& ys_data_np;
    const vector<np_array_realtype>& yps_data_np;
    const vector<np_array_realtype>& inputs_np;
    const vector<std::shared_ptr<const casadi::Function>>& funcs;
    realtype* entries;
    const int size_spatial;
    vector<realtype> c;
    vector<realtype> d;
    vector<realtype> y_buffer;
    vector<const realtype*> args;
    vector<realtype*> results;
    vector<casadi_int> iw;
    vector<realtype> w;
};

// Observe the raw data
class TimeSeriesProcessor {
public:
    TimeSeriesProcessor(const vector<np_array_realtype>& _ts,
                        const vector<np_array_realtype>& _ys,
                        const vector<np_array_realtype>& _inputs,
                        const vector<std::shared_ptr<const casadi::Function>>& _funcs,
                        realtype* _entries,
                        const bool _is_f_contiguous,
                        const int _size_spatial)
        : ts(_ts), ys(_ys), inputs(_inputs), funcs(_funcs),
          entries(_entries), is_f_contiguous(_is_f_contiguous), size_spatial(_size_spatial) {}

    void process() {
        int i_entries = 0;
        for (size_t i = 0; i < ts.size(); i++) {
            const auto& t = ts[i].unchecked<1>();
            // Continue if there is no data
            if (t.size() == 0) {
                continue;
            }
            const auto& y = ys[i].unchecked<2>();
            const auto input = inputs[i].data();
            const auto func = *funcs[i];

            resize_arrays(y.shape(0), funcs[i]);
            args[2] = input;

            for (size_t j = 0; j < t.size(); j++) {
                const realtype t_val = t(j);
                const realtype* y_val = is_f_contiguous ? &y(0, j) : copy_to_buffer(y_buffer, y, j);

                args[0] = &t_val;
                args[1] = y_val;
                results[0] = &entries[i_entries];

                func(args.data(), results.data(), iw.data(), w.data(), 0);

                i_entries += size_spatial;
            }
        }
    }

private:
    const realtype* copy_to_buffer(
        vector<realtype>& entries,
        const py::detail::unchecked_reference<realtype, 2>& y,
        size_t j) {
        for (size_t i = 0; i < entries.size(); ++i) {
            entries[i] = y(i, j);
        }

        return entries.data();
    }

    void resize_arrays(const int M, std::shared_ptr<const casadi::Function> func) {
        args.resize(func->sz_arg());
        results.resize(func->sz_res());
        iw.resize(func->sz_iw());
        w.resize(func->sz_w());
        if (!is_f_contiguous && y_buffer.size() < M) {
            y_buffer.resize(M);
        }
    }

    const vector<np_array_realtype>& ts;
    const vector<np_array_realtype>& ys;
    const vector<np_array_realtype>& inputs;
    const vector<std::shared_ptr<const casadi::Function>>& funcs;
    realtype* entries;
    const bool is_f_contiguous;
    int size_spatial;
    vector<realtype> y_buffer;
    vector<const realtype*> args;
    vector<realtype*> results;
    vector<casadi_int> iw;
    vector<realtype> w;
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
    const auto& funcs = setup_casadi_funcs(strings);
    py::array_t<realtype, py::array::f_style> out_array(shape);
    auto entries = out_array.mutable_data();

    TimeSeriesInterpolator(t_interp_np, ts_np, ys_np, yps_np, inputs_np, funcs, entries, size_spatial).process();

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
    const auto& funcs = setup_casadi_funcs(strings);
    py::array_t<realtype, py::array::f_style> out_array(shape);
    auto entries = out_array.mutable_data();

    TimeSeriesProcessor(ts_np, ys_np, inputs_np, funcs, entries, is_f_contiguous, size_spatial).process();

    return out_array;
}

const vector<std::shared_ptr<const casadi::Function>> setup_casadi_funcs(const vector<std::string>& strings) {
    std::unordered_map<std::string, std::shared_ptr<casadi::Function>> function_cache;
    vector<std::shared_ptr<const casadi::Function>> funcs(strings.size());

    for (size_t i = 0; i < strings.size(); ++i) {
        const std::string& str = strings[i];

        // Check if function is already in the local cache
        if (function_cache.find(str) == function_cache.end()) {
            // If not in the cache, create a new casadi::Function::deserialize and store it
            function_cache[str] = std::make_shared<casadi::Function>(casadi::Function::deserialize(str));
        }

        // Retrieve the function from the cache as a shared pointer
        funcs[i] = function_cache[str];
    }

    return funcs;
}
