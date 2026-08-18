#ifndef PYBAMM_IDAKLU_REDUCE_HPP
#define PYBAMM_IDAKLU_REDUCE_HPP

#include "common.hpp"
#include "HermiteKnotReducer.hpp"
#include "SolutionData.hpp"  // for vector_to_numpy()
#include <cstring>

/**
 * @brief Post-hoc streaming knot reduction on multi-segment solution data.
 *
 * Accepts vectors of per-segment arrays in the flat time-major layout
 * used by pybamm.Solution from IDAKLUSolver:
 *   ts[i]:  1D, shape (M_i,)
 *   ys[i]:  1D, shape (M_i * N_i,), time-major
 *   yps[i]: 1D, shape (M_i * N_i,), same layout
 *   atols[i]: 1D, shape (N_i,), per-state absolute tolerance
 *   t_evals[i]: 1D, shape (K_i,), breakpoint times for this segment
 *
 * n_states per segment is inferred as ys[i].size() / ts[i].size().
 *
 * Points matching a t_eval time are marked as breakpoints, except the last
 * point of each segment (which is handled by Finalize(), matching the online
 * solver where will_exit_loop suppresses the breakpoint flag).
 *
 * @return py::tuple of three VectorRealtypeNdArray (reduced ts, ys, yps).
 */
inline py::object reduce_knots(
    const std::vector<np_array_realtype>& ts,
    const std::vector<np_array_realtype>& ys,
    const std::vector<np_array_realtype>& yps,
    const std::vector<np_array_realtype>& atols,
    const std::vector<np_array_realtype>& t_evals,
    double rtol,
    double multiplier)
{
    const size_t n_seg = ts.size();

    std::vector<np_array_realtype> out_ts;
    std::vector<np_array_realtype> out_ys;
    std::vector<np_array_realtype> out_yps;
    out_ts.reserve(n_seg);
    out_ys.reserve(n_seg);
    out_yps.reserve(n_seg);

    for (size_t seg = 0; seg < n_seg; ++seg) {
        const sunrealtype* t_ptr = ts[seg].data();
        const sunrealtype* y_ptr = ys[seg].data();
        const sunrealtype* yp_ptr = yps[seg].data();
        const sunrealtype* atol_ptr = atols[seg].data();

        const int M = static_cast<int>(ts[seg].size());
        const int N = M > 0 ? static_cast<int>(ys[seg].size()) / M : 0;

        const sunrealtype* te_ptr = t_evals[seg].data();
        const int K = static_cast<int>(t_evals[seg].size());

        std::vector<sunrealtype> out_t;
        std::vector<sunrealtype> out_y;
        std::vector<sunrealtype> out_yp;
        out_t.reserve(M);
        out_y.reserve(static_cast<size_t>(M) * N);
        out_yp.reserve(static_cast<size_t>(M) * N);

        HermiteKnotReducer reducer(N, rtol, atol_ptr,
                                     multiplier,
                                     out_t, out_y, out_yp);

        int j = 0;  // index into t_evals
        for (int i = 0; i < M; ++i) {
            // Advance t_eval cursor past any entries before t_ptr[i]
            while (j < K && te_ptr[j] < t_ptr[i]) ++j;

            // check if we match a t_eval time
            bool is_breakpoint = (
                i < M - 1 &&
                j < K && te_ptr[j] == t_ptr[i]
            );

            reducer.ProcessPoint(
                t_ptr[i],
                &y_ptr[static_cast<size_t>(i) * N],
                &yp_ptr[static_cast<size_t>(i) * N],
                is_breakpoint
            );
        }
        reducer.Finalize();

        out_ts.push_back(vector_to_numpy(std::move(out_t)));
        out_ys.push_back(vector_to_numpy(std::move(out_y)));
        out_yps.push_back(vector_to_numpy(std::move(out_yp)));
    }

    return py::make_tuple(
        py::cast(std::move(out_ts)),
        py::cast(std::move(out_ys)),
        py::cast(std::move(out_yps))
    );
}

#endif // PYBAMM_IDAKLU_REDUCE_HPP
