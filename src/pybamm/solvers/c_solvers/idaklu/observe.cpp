#include "observe.hpp"
#include <iostream>
#include <chrono>

class HermiteInterpolator {
public:
    HermiteInterpolator(const py::detail::unchecked_reference<realtype, 1>& t,
                       const py::detail::unchecked_reference<realtype, 2>& y,
                       const py::detail::unchecked_reference<realtype, 2>& yp)
        : t(t), y(y), yp(yp) {}

    void compute_c_d(size_t j, vector<realtype>& c, vector<realtype>& d) const {
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

    void interpolate(vector<realtype>& out, realtype t_interp, size_t j, vector<realtype>& c, vector<realtype>& d) const {
        // Must be called after compute_c_d
        const realtype h = t_interp - t(j);
        const realtype h2 = h * h;
        const realtype h3 = h2 * h;

        for (size_t i = 0; i < out.size(); ++i) {
            realtype y_ij = y(i, j);
            realtype yp_ij = yp(i, j);
            out[i] = y_ij + yp_ij * h + c[i] * h2 + d[i] * h3;
        }
    }

private:
    const py::detail::unchecked_reference<realtype, 1>& t;
    const py::detail::unchecked_reference<realtype, 2>& y;
    const py::detail::unchecked_reference<realtype, 2>& yp;
};

int setup_observable(const vector<int>& sizes) {
    if (sizes.empty()) {
        throw std::invalid_argument("sizes must have at least one element");
    }

    int size_tot = 1;
    for (const auto& size : sizes) {
        size_tot *= size;
    }

    if (size_tot == 0) {
        throw std::invalid_argument("sizes must have at least one element");
    }

    return size_tot;
}

class TimeSeriesProcessor {
public:
    TimeSeriesProcessor(const vector<np_array_realtype>& _ts,
                        const vector<np_array_realtype>& _ys,
                        const vector<np_array_realtype>& _inputs,
                        const std::vector<std::shared_ptr<const casadi::Function>>& _funcs,
                        realtype* _out,
                        bool _is_f_contiguous,
                        const int _len)
        : ts(_ts), ys(_ys), inputs(_inputs), funcs(_funcs),
          out(_out), is_f_contiguous(_is_f_contiguous), len(_len) {}

    void process() {
        vector<realtype> y_buffer;
        vector<const realtype*> args;
        vector<realtype*> results;

        int count = 0;
        for (size_t i = 0; i < ts.size(); i++) {
            const auto& t = ts[i].unchecked<1>();
            const auto& y = ys[i].unchecked<2>();
            const auto input = inputs[i].data();
            const auto func = *funcs[i];

            std::vector<casadi_int> iw(funcs[i]->sz_iw());
            std::vector<realtype> w(funcs[i]->sz_w());
            args.resize(funcs[i]->sz_arg());
            args[2] = input;

            // Output buffer
            results.resize(funcs[i]->sz_res());

            int M = y.shape(0);
            if (!is_f_contiguous && y_buffer.size() < M) {
                y_buffer.resize(M);
            }

            for (size_t j = 0; j < t.size(); j++) {
                const realtype t_val = t(j);
                const realtype* y_val = is_f_contiguous ? &y(0, j) : copy_to_buffer(y_buffer, y, j);

                args[0] = &t_val;
                args[1] = y_val;
                results[0] = &out[count];

                func(casadi::get_ptr(args), casadi::get_ptr(results), casadi::get_ptr(iw), casadi::get_ptr(w), 0);

                count += len;
            }
        }
    }

private:
    const realtype* copy_to_buffer(vector<realtype>& out, const py::detail::unchecked_reference<realtype, 2>& y, size_t j) {
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = y(i, j);
        }

        return out.data();
    }

    const vector<np_array_realtype>& ts;
    const vector<np_array_realtype>& ys;
    const vector<np_array_realtype>& inputs;
    const std::vector<std::shared_ptr<const casadi::Function>>& funcs;
    realtype* out;
    bool is_f_contiguous;
    int len;
};

class TimeSeriesInterpolator {
public:
    TimeSeriesInterpolator(const np_array_realtype& _t_interp_np,
                           const vector<np_array_realtype>& _ts_data,
                           const vector<np_array_realtype>& _ys_data,
                           const vector<np_array_realtype>& _yps_data,
                           const vector<np_array_realtype>& _inputs,
                           const std::vector<std::shared_ptr<const casadi::Function>>& _funcs,
                           realtype* _out,
                           int _len)
        : t_interp_np(_t_interp_np), ts_data_np(_ts_data), ys_data_np(_ys_data),
          yps_data_np(_yps_data), inputs_np(_inputs), funcs(_funcs),
          out(_out), len(_len) {}

    void process() {
        auto t_interp = t_interp_np.unchecked<1>();
        ssize_t i_interp = 0;
        int count = 0;
        ssize_t N_data = 0;
        const ssize_t N_interp = t_interp.size();

        for (const auto& ts : ts_data_np) {
            N_data += ts.size();
        }

        // Preallocate vectors
        vector<realtype> c, d, y_interp;

        vector<const realtype*> args;
        vector<realtype*> results;
        vector<casadi_int> iw;
        vector<realtype> w;

        // Main processing within bounds
        process_within_bounds(i_interp, count, t_interp, N_interp, args, results, iw, w, y_interp, c, d);

        // Extrapolation for remaining points
        if (i_interp < N_interp) {
            extrapolate_remaining(i_interp, count, t_interp, N_interp, args, results, iw, w, y_interp, c, d);
        }
    }

    void process_within_bounds(
            ssize_t& i_interp,
            int& count,
            const py::detail::unchecked_reference<realtype, 1>& t_interp,
            const ssize_t N_interp,
            vector<const realtype*>& args,
            vector<realtype*>& results,
            vector<casadi_int>& iw,
            vector<realtype>& w,
            vector<realtype>& y_interp,
            vector<realtype>& c,
            vector<realtype>& d
        ) {
        for (size_t i = 0; i < ts_data_np.size(); i++) {
            const auto& t_data = ts_data_np[i].unchecked<1>();
            const auto& y_data = ys_data_np[i].unchecked<2>();
            const auto& yp_data = yps_data_np[i].unchecked<2>();
            const auto input = inputs_np[i].data();
            const auto func = *funcs[i];
            const realtype t_data_final = t_data(t_data.size() - 1);

            resize_arrays(y_interp, c, d, y_data.shape(0));

            iw.resize(funcs[i]->sz_iw());
            w.resize(funcs[i]->sz_w());
            args.resize(funcs[i]->sz_arg());

            args[1] = y_interp.data();
            args[2] = input;

            // Output buffer
            results.resize(funcs[i]->sz_res());

            ssize_t j = 0;
            ssize_t j_prev = -1;
            auto itp = HermiteInterpolator(t_data, y_data, yp_data);
            while (i_interp < N_interp && t_interp(i_interp) <= t_data_final) {
                for (; j < t_data.size() - 2; ++j) {
                    if (t_data(j) <= t_interp(i_interp) && t_interp(i_interp) <= t_data(j + 1)) {
                        break;
                    }
                }

                if (j != j_prev) {
                    // Compute c and d for the new interval
                    itp.compute_c_d(j, c, d);
                }

                itp.interpolate(y_interp, t_interp(i_interp), j, c, d);

                args[0] = &t_interp(i_interp);
                results[0] = &out[count];
                func(casadi::get_ptr(args), casadi::get_ptr(results), casadi::get_ptr(iw), casadi::get_ptr(w), 0);

                count += len;
                ++i_interp;
                j_prev = j;
            }
        }
    }

    void extrapolate_remaining(
            ssize_t& i_interp,
            int& count,
            const py::detail::unchecked_reference<realtype, 1>& t_interp,
            const ssize_t N_interp,
            vector<const realtype*>& args,
            vector<realtype*>& results,
            vector<casadi_int>& iw,
            vector<realtype>& w,
            vector<realtype>& y_interp,
            vector<realtype>& c,
            vector<realtype>& d
        ) {
        const auto& t_data = ts_data_np.back().unchecked<1>();
        const auto& y_data = ys_data_np.back().unchecked<2>();
        const auto& yp_data = yps_data_np.back().unchecked<2>();
        const auto inputs_data = inputs_np.back().data();
        const auto func = *funcs.back();
        const ssize_t j = t_data.size() - 2;

        resize_arrays(y_interp, c, d, y_data.shape(0));
        iw.resize(funcs.back()->sz_iw());
        w.resize(funcs.back()->sz_w());
        args.resize(funcs.back()->sz_arg());

        auto itp = HermiteInterpolator(t_data, y_data, yp_data);
        itp.compute_c_d(j, c, d);

        for (; i_interp < N_interp; ++i_interp) {
            const realtype t_interp_next = t_interp(i_interp);
            itp.interpolate(y_interp, t_interp_next, j, c, d);

            args[0] = &t_interp_next;
            results[0] = &out[count];
            func(casadi::get_ptr(args), casadi::get_ptr(results), casadi::get_ptr(iw), casadi::get_ptr(w), 0);

            count += len;
        }
    }

    void resize_arrays(vector<realtype>& y_interp, vector<realtype>& c, vector<realtype>& d, const int M) {
        if (y_interp.size() < M) {
            y_interp.resize(M);
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
    const std::vector<std::shared_ptr<const casadi::Function>>& funcs;
    realtype* out;
    int len;
};

const np_array_realtype observe_hermite_interp_ND(
    const np_array_realtype& t_interp_np,
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& yps_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<std::string>& strings,
    const vector<int> sizes
) {
    const int size_tot = setup_observable(sizes);
    const auto& funcs = setup_casadi_funcs(strings);
    py::array_t<realtype, py::array::f_style> out_array(sizes);
    auto out = out_array.mutable_data();

    TimeSeriesInterpolator(t_interp_np, ts_np, ys_np, yps_np, inputs_np, funcs, out, size_tot / sizes.back()).process();

    return out_array;
}

const np_array_realtype observe_ND(
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<std::string>& strings,
    const bool is_f_contiguous,
    const vector<int> sizes
) {
    const int size_tot = setup_observable(sizes);
    const auto& funcs = setup_casadi_funcs(strings);
    py::array_t<realtype, py::array::f_style> out_array(sizes);
    auto out = out_array.mutable_data();

    TimeSeriesProcessor(ts_np, ys_np, inputs_np, funcs, out, is_f_contiguous, size_tot / sizes.back()).process();

    return out_array;
}

const std::vector<std::shared_ptr<const casadi::Function>> setup_casadi_funcs(const std::vector<std::string>& strings) {
    std::unordered_map<std::string, std::shared_ptr<casadi::Function>> function_cache;
    std::vector<std::shared_ptr<const casadi::Function>> funcs(strings.size());

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
