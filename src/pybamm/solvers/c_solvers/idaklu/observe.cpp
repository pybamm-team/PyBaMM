#include "observe.hpp"

class HermiteInterpolator {
public:
    HermiteInterpolator(const py::detail::unchecked_reference<double, 1>& t,
                       const py::detail::unchecked_reference<double, 2>& y,
                       const py::detail::unchecked_reference<double, 2>& yp)
        : t(t), y(y), yp(yp) {}

    void compute_c_d(size_t j, vector<double>& c, vector<double>& d) const {
        const double h_full = t(j + 1) - t(j);
        const double inv_h = 1.0 / h_full;
        const double inv_h2 = inv_h * inv_h;
        const double inv_h3 = inv_h2 * inv_h;

        for (size_t i = 0; i < y.shape(0); ++i) {
            double y_ij = y(i, j);
            double yp_ij = yp(i, j);
            double y_ijp1 = y(i, j + 1);
            double yp_ijp1 = yp(i, j + 1);

            c[i] = 3.0 * (y_ijp1 - y_ij) * inv_h2 - (2.0 * yp_ij + yp_ijp1) * inv_h;
            d[i] = 2.0 * (y_ij - y_ijp1) * inv_h3 + (yp_ij + yp_ijp1) * inv_h2;
        }
    }

    void interpolate(vector<double>& out, double t_interp, size_t j, vector<double>& c, vector<double>& d) const {
        const double h = t_interp - t(j);
        const double h2 = h * h;
        const double h3 = h2 * h;

        for (size_t i = 0; i < out.size(); ++i) {
            double y_ij = y(i, j);
            double yp_ij = yp(i, j);
            out[i] = y_ij + yp_ij * h + c[i] * h2 + d[i] * h3;
        }
    }

private:
    const py::detail::unchecked_reference<double, 1>& t;
    const py::detail::unchecked_reference<double, 2>& y;
    const py::detail::unchecked_reference<double, 2>& yp;
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
                        const vector<const casadi::Function*>& _funcs,
                        double* _out,
                        bool _is_f_contiguous,
                        const int _len)
        : ts(_ts), ys(_ys), inputs(_inputs), funcs(_funcs),
          out(_out), is_f_contiguous(_is_f_contiguous), len(_len) {}

    void process() {
        vector<double> y_buffer;
        vector<const double*> args;
        vector<double*> results;

        int count = 0;
        for (size_t i = 0; i < ts.size(); i++) {
            const auto& t = ts[i].unchecked<1>();
            const auto& y = ys[i].unchecked<2>();
            const auto input = inputs[i].data();
            const auto func = *funcs[i];

            int M = y.shape(0);
            if (!is_f_contiguous && y_buffer.size() < M) {
                y_buffer.resize(M);
            }

            for (size_t j = 0; j < t.size(); j++) {
                const double t_val = t(j);
                const double* y_val = is_f_contiguous ? &y(0, j) : copy_to_buffer(y_buffer, y, j);

                args = { &t_val, y_val, input };
                results = { &out[count] };
                func(args, results);

                count += len;
            }
        }
    }

private:
    const double* copy_to_buffer(vector<double>& out, const py::detail::unchecked_reference<double, 2>& y, size_t j) {
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = y(i, j);
        }

        return out.data();
    }

    const vector<np_array_realtype>& ts;
    const vector<np_array_realtype>& ys;
    const vector<np_array_realtype>& inputs;
    const vector<const casadi::Function*>& funcs;
    double* out;
    bool is_f_contiguous;
    int len;
};

class TimeSeriesInterpolator {
public:
    TimeSeriesInterpolator(const np_array_realtype& t_interp_np,
                           const vector<np_array_realtype>& ts_data,
                           const vector<np_array_realtype>& ys_data,
                           const vector<np_array_realtype>& yps_data,
                           const vector<np_array_realtype>& inputs,
                           const vector<const casadi::Function*>& funcs,
                           double* out,
                           int len)
        : t_interp_np(t_interp_np), ts_data_np(ts_data), ys_data_np(ys_data),
          yps_data_np(yps_data), inputs_np(inputs), funcs(funcs),
          out(out), len(len) {}

    void process() {
        vector<double> y_interp;
        auto t_interp = t_interp_np.unchecked<1>();
        ssize_t i_interp = 0;
        int count = 0;
        ssize_t N_data = 0;
        const ssize_t N_interp = t_interp.size();

        for (const auto& ts : ts_data_np) {
            N_data += ts.size();
        }

        // Preallocate c and d vectors
        vector<double> c, d;

        // Main processing within bounds
        process_within_bounds(i_interp, count, y_interp, c, d, t_interp, N_interp);

        // Extrapolation for remaining points
        if (i_interp < N_interp) {
            extrapolate_remaining(i_interp, count, y_interp, c, d, t_interp, N_interp);
        }
    }

    void process_within_bounds(ssize_t& i_interp, int& count, vector<double>& y_interp,
                                vector<double>& c, vector<double>& d,
                                const py::detail::unchecked_reference<double, 1>& t_interp,
                                const ssize_t N_interp) {
        vector<const double*> args;
        vector<double*> results;
        for (size_t i = 0; i < ts_data_np.size(); i++) {
            const auto& t_data = ts_data_np[i].unchecked<1>();
            const auto& y_data = ys_data_np[i].unchecked<2>();
            const auto& yp_data = yps_data_np[i].unchecked<2>();
            const auto inputs_data = inputs_np[i].data();
            const auto func = *funcs[i];
            const double t_data_final = t_data(t_data.size() - 1);

            resize_arrays(y_interp, c, d, y_data.shape(0));

            args = { &t_interp(i_interp), y_interp.data(), inputs_data };

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
                results = { &out[count] };
                args[0] = &t_interp(i_interp);
                func(args, results);

                count += len;
                ++i_interp;
                j_prev = j;
            }
        }
    }

    void extrapolate_remaining(ssize_t& i_interp, int& count, vector<double>& y_interp,
                               vector<double>& c, vector<double>& d,
                               const py::detail::unchecked_reference<double, 1>& t_interp,
                               const ssize_t N_interp) {
        const auto& t_data = ts_data_np.back().unchecked<1>();
        const auto& y_data = ys_data_np.back().unchecked<2>();
        const auto& yp_data = yps_data_np.back().unchecked<2>();
        const auto inputs_data = inputs_np.back().data();
        const auto func = *funcs.back();
        const ssize_t j = t_data.size() - 2;

        resize_arrays(y_interp, c, d, y_data.shape(0));

        auto itp = HermiteInterpolator(t_data, y_data, yp_data);
        itp.compute_c_d(j, c, d);

        for (; i_interp < N_interp; ++i_interp) {
            const double t_interp_next = t_interp(i_interp);
            itp.interpolate(y_interp, t_interp_next, j, c, d);

            vector<const double*> args = { &t_interp_next, y_interp.data(), inputs_data };
            vector<double*> results = { &out[count] };
            func(args, results);

            count += len;
        }
    }

    void resize_arrays(vector<double>& y_interp, vector<double>& c, vector<double>& d, const int M) {
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
    const vector<const casadi::Function*>& funcs;
    double* out;
    int len;
};

const py::array_t<double> observe_hermite_interp_ND(
    const np_array_realtype& t_interp_np,
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& yps_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<const casadi::Function*>& funcs,
    const vector<int> sizes
) {
    const int size_tot = setup_observable(sizes);
    py::array_t<double, py::array::f_style> out_array(sizes);
    auto out = out_array.mutable_data();

    TimeSeriesInterpolator(t_interp_np, ts_np, ys_np, yps_np, inputs_np, funcs, out, size_tot / sizes.back()).process();

    return out_array;
}

const py::array_t<double> observe_ND(
    const vector<np_array_realtype>& ts_np,
    const vector<np_array_realtype>& ys_np,
    const vector<np_array_realtype>& inputs_np,
    const vector<const casadi::Function*>& funcs,
    const bool is_f_contiguous,
    const vector<int> sizes
) {
    const int size_tot = setup_observable(sizes);
    py::array_t<double, py::array::f_style> out_array(sizes);
    auto out = out_array.mutable_data();

    TimeSeriesProcessor(ts_np, ys_np, inputs_np, funcs, out, is_f_contiguous, size_tot / sizes.back()).process();

    return out_array;
}
