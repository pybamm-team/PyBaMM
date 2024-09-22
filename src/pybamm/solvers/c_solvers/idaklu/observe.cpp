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

        // c[i] = (3 * (y_ptr[i + 1] - y_ptr[i]) * inv_h_sq) - (2 * yp_ptr[i] + yp_ptr[i + 1]) * inv_h;
        // d[i] = (2 * (y_ptr[i] - y_ptr[i + 1]) * inv_h_sq * inv_h) + (yp_ptr[i] + yp_ptr[i + 1]) * inv_h_sq;

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

const double hermite_interp_scalar(
    const double t_interp,
    const double t_j,
    const double t_jp1,
    const double y_j,
    const double y_jp1,
    const double yp_j,
    const double yp_jp1
) {
    // const double h = t_interp - t(j);
    // const double h2 = h * h;
    // const double h3 = h2 * h;

    // const double h_full = t(j + 1) - t(j);
    // const double inv_h = 1 / h_full;
    // const double inv_h2 = inv_h * inv_h;
    // const double inv_h3 = inv_h2 * inv_h;

    // double c, d, y_ij, yp_ij, y_ijp1, yp_ijp1;

    // for (size_t i = 0; i < out.size(); ++i) {
    //     y_ij = y(i, j);
    //     yp_ij = yp(i, j);
    //     y_ijp1 = y(i, j + 1);
    //     yp_ijp1 = yp(i, j + 1);

    //     // c[i] = (3 * (y_ptr[i + 1] - y_ptr[i]) * inv_h_sq) - (2 * yp_ptr[i] + yp_ptr[i + 1]) * inv_h;
    //     // d[i] = (2 * (y_ptr[i] - y_ptr[i + 1]) * inv_h_sq * inv_h) + (yp_ptr[i] + yp_ptr[i + 1]) * inv_h_sq;

    //     c = 3 * (y_ijp1 - y_ij) * inv_h2 - (2 * yp_ij + yp_ijp1) * inv_h;
    //     d = 2 * (y_ij - y_ijp1) * inv_h3 + (yp_ij + yp_ijp1) * inv_h2;

    //     out[i] = y_ij + yp_ij * h + c * h2 + d * h3;
    // }

    const double h = t_interp - t_j;
    const double h2 = h * h;
    const double h3 = h2 * h;

    const double hinv = 1 / (t_jp1 - t_j);
    const double h2inv = hinv * hinv;
    const double h3inv = h2inv * hinv;

    const double c = 3 * (y_jp1 - y_j) * h2inv - (2 * yp_j + yp_jp1) * hinv;
    const double d = 2 * (y_j - y_jp1) * h3inv + (yp_j + yp_jp1) * h2inv;

    return y_j + yp_j * h + c * h2 + d * h3;
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
