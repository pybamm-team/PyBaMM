#ifndef HERMITE_KNOT_REDUCER_HPP
#define HERMITE_KNOT_REDUCER_HPP

#include "common.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _MSC_VER
  #define PYBAMM_RESTRICT __restrict
#else
  #define PYBAMM_RESTRICT __restrict__
#endif

/**
 * @brief True streaming knot reducer with Bernstein-certified error bounds
 *        and optional least-squares derivative refinement.
 *
 * Processes points one at a time as IDA generates them, deciding inline
 * whether to keep or discard each point. Only committed points are stored.
 *
 * Error checking uses a three-level hierarchy for maximum efficiency:
 *
 *   Level 1 (hot path): Conservative Bernstein bound with inv_atol weighting.
 *     Division-free, fully SIMD-vectorizable. Catches the common case where
 *     the merge succeeds easily. Endpoint sub-intervals use only 2 of the 4
 *     Bernstein control points (anchor/new-point error is exactly zero).
 *
 *   Level 2 (rare): Bernstein bound with exact WRMS weight.
 *     Uses fmin(|y_left|, |y_right|) for conservative weight certification.
 *     Only entered when Level 1's conservative bound exceeds the threshold.
 *
 *   Level 3 (very rare): De Casteljau midpoint subdivision tightening.
 *     Splits each Bernstein curve at the midpoint via de Casteljau's algorithm,
 *     producing 8 refined control points whose convex hull is ~4x tighter.
 *
 * LeastSquares derivative refinement (integral L2 objective):
 *
 *   After the greedy pass selects knots, a least-squares update adjusts the
 *   derivative (y') values at each knot to minimize the continuous integral
 *   of the squared error between the original and reduced Hermite interpolants:
 *
 *     min_δ  Σ_spans ∫ (H_orig(t) - H_new(t; y'+δ))**2 dt
 *
 *   The sensitivity functions φ_A(t) = h₁₀(u)·h_m and φ_B(t) = h₁₁(u)·h_m
 *   are global cubics on each merged span, so their inner products come from
 *   the Hermite mass matrix (Gram matrix) and are closed-form constants:
 *
 *     diag += h_m**3/105,   offdiag += -h_m**3/140
 *
 *   The matrix is strictly diagonally dominant (1/105 > 1/140), hence always
 *   SPD
 *
 *   The LU solve at Finalize() is O(K) scalar factorization + O(K*S)
 *   fused back-sub + yp update — negligible versus O(M*S) accumulation.
 *
 *   The refinement can only reduce or maintain the L2 error.
 *   δ=0 is always feasible (produces the original y' values), so the LeastSquares
 *   minimizer can never increase the error.
 *
 * Numerical stability: The merged Hermite spline is evaluated using normalized
 * coordinates u = (t - t_anchor) / h_m with Hermite basis functions that
 * are O(1) for u in [0,1]. This avoids the catastrophic cancellation inherent
 * in Horner evaluation of monomial coefficients that scale as O(1/h**k).
 *
 * Memory:
 *   - O(1) for anchor point
 *   - O(span) for intermediate points in current merge window
 *   - O(K) for committed output where K = kept points
 *   - O(K + K*S) for LeastSquares refinement arrays (when enabled): K scalars for
 *     state-independent matrix + K*S for per-state RHS
 *
 * Algorithm:
 *   - anchor: last committed point (fixed endpoint of merged spline)
 *   - window: all points since anchor (to be potentially merged)
 *   - When new point arrives, check if window + new can all be replaced by anchor->new
 *   - If yes: add new to window
 *   - If no: commit last point in window, it becomes anchor, continue
 */
class HermiteKnotReducer {
public:
    HermiteKnotReducer(
        int n_states,
        double rtol,
        const double* atol_ptr,
        double multiplier,
        // Output arrays (solver's arrays, will grow as points are committed)
        std::vector<sunrealtype>& out_t,
        std::vector<sunrealtype>& out_y,
        std::vector<sunrealtype>& out_yp
    ) : n_states_(n_states),
        rtol_(rtol),
        threshold_(static_cast<double>(n_states) * (multiplier - 1.0) * (multiplier - 1.0)),
        active_(multiplier > 1.0),
        ls_knot_count_(0),
        ls_interval_start_(0),
        out_t_(&out_t),
        out_y_(&out_y),
        out_yp_(&out_yp),
        out_count_(0),
        has_anchor_(false)
    {
        // Precompute 1/atol (solver's raw tolerance, no multiplier scaling)
        inv_atol_.resize(n_states);
        for (int j = 0; j < n_states; ++j) {
            inv_atol_[j] = 1.0 / atol_ptr[j];
        }

        // Double-buffered knot error caches for Bernstein bound computation
        dk_buf0_.resize(n_states);
        dpk_buf0_.resize(n_states);
        dk_buf1_.resize(n_states);
        dpk_buf1_.resize(n_states);

        // Sentinel states: top-K by inv_atol (most sensitive to error).
        // Used for O(K) early-exit on reject paths before the full O(n) SIMD loop.
        // ~5% of states, capped at 32.
        n_sentinels_ = std::min(
            std::min(n_states, 32),
            static_cast<int>(std::pow(2, std::floor(std::log2(0.05 * n_states))))
        );
        sentinels_.resize(n_sentinels_);
        {
            std::vector<int> idx(n_states);
            for (int j = 0; j < n_states; ++j) {
                idx[j] = j;
            }
            
            std::partial_sort(idx.begin(), idx.begin() + n_sentinels_, idx.end(),
                [this](int a, int b) { return inv_atol_[a] > inv_atol_[b]; });
            
            for (int s = 0; s < n_sentinels_; ++s) {
                sentinels_[s] = idx[s];
            }
        }

        // Reserve space for window (will grow dynamically)
        const int n_est = 32;
        window_t_.reserve(n_est);
        window_y_.reserve(n_est * n_states);
        window_yp_.reserve(n_est * n_states);

        // LeastSquares work buffers (factored accumulation)
        ls_sum_A_.resize(n_states, 0.0);
        ls_sum_B_.resize(n_states, 0.0);
    }

    bool IsActive() const { return active_; }

    /**
     * @brief Process a new point from the solver.
     *
     * Matches post-processing greedy algorithm:
     * - Track "candidate" = last point for which merge succeeded
     * - On failure, commit candidate, anchor = candidate, start new window
     *
     * When LeastSquares refinement is enabled, the closing span's interior points
     * are accumulated into the tri-diagonal normal equations just before
     * the candidate is committed (cache-hot from the merge check).
     */
    void ProcessPoint(sunrealtype t, const sunrealtype* y, const sunrealtype* yp, bool is_breakpoint) {
        // First point ever: commit as anchor
        if (!has_anchor_) {
            CommitPoint(t, y, yp);
            SetAnchor(t, y, yp);
            return;
        }

        // Breakpoint: must commit candidate (if any) and this point
        if (is_breakpoint) {
            sunrealtype h_span;
            if (!window_t_.empty()) {
                // trailing span: candidate → breakpoint
                FlushCandidate();
                h_span = t - window_t_.back();                
            } else {
                // span: anchor → breakpoint (no interior)
                h_span = t - anchor_t_;
            }
            LeastSquaresAccumulateMatrix(h_span);
            CommitPoint(t, y, yp);
            SetAnchor(t, y, yp);
            // Back-to-back anchor: finalize LeastSquares for the completed interval
            FinalizeInterval();
            return;
        }

        // First point after anchor: just add to window
        // Try to extend window with new point
        if (window_t_.empty() || CanMergeWindow(t, y, yp)) {
            AddToWindow(t, y, yp);
            return;
        }

        // Failure: commit candidate (= window.back()), it becomes new anchor
        const bool back_to_back = (window_t_.size() == 1);
        FlushCandidate();
        const size_t last = window_t_.size() - 1;
        SetAnchor(window_t_[last], &window_y_[last * n_states_],
                    &window_yp_[last * n_states_]);
        // Back-to-back anchor (window had only 1 point = candidate):
        // finalize LeastSquares since there are no interior points to couple across
        if (back_to_back) {
            FinalizeInterval();
        }
        AddToWindow(t, y, yp);
    }

    /**
     * @brief Flush remaining candidate at end of solve, then finalize
     *        the last interval's LeastSquares refinement.
     */
    void Finalize() {
        if (!window_t_.empty()) {
            FlushCandidate();
        }
        FinalizeInterval();
    }

    /**
     * @brief Solve LeastSquares refinement for the current continuous interval,
     *        then reset the LeastSquares system for the next interval.
     *
     * Called at every interval boundary: integrator reinitialize (t_eval),
     * events, and end-of-solve. Ensures the LeastSquares tridiagonal system is
     * scoped to a single continuous interval — never spanning across
     * discontinuities where the integrator reinitializes.
     *
     * The last committed point becomes knot 0 of the next interval.
     */
    void FinalizeInterval() {
        if (ls_knot_count_ >= 2) {
            LeastSquaresSolveAndUpdate();
        }

        // Reset LeastSquares state: last committed point becomes knot 0 of next interval
        ls_interval_start_ = out_count_ - 1;
        ls_knot_count_ = 1;
        ls_diag_.assign(1, 0.0);
        ls_offdiag_.clear();
        ls_rhs_.assign(static_cast<size_t>(n_states_), 0.0);
    }

    int GetOutputCount() const { return out_count_; }

private:
    //  Data Members

    // Core
    int n_states_;
    double rtol_;
    double threshold_;
    bool active_;
    std::vector<double> inv_atol_;

    // Output (pointers to solver's arrays)
    std::vector<sunrealtype>* out_t_;
    std::vector<sunrealtype>* out_y_;
    std::vector<sunrealtype>* out_yp_;
    int out_count_;

    // Anchor (last committed point)
    sunrealtype anchor_t_;
    std::vector<sunrealtype> anchor_y_, anchor_yp_;
    bool has_anchor_;

    // Window: points since anchor that may be removed
    std::vector<sunrealtype> window_t_;
    std::vector<sunrealtype> window_y_;   // Flat: [n_states * window_size]
    std::vector<sunrealtype> window_yp_;  // Flat: [n_states * window_size]

    // Greedy: double-buffered knot error caches
    std::vector<sunrealtype> dk_buf0_, dpk_buf0_, dk_buf1_, dpk_buf1_;

    // Greedy: sentinel states (top-K by inv_atol)
    std::vector<int> sentinels_;
    int n_sentinels_;

    // LeastSquares derivative refinement
    //
    // Tridiagonal normal-equation system for the integral L2 objective.
    // Matrix entries are state-independent O(1) per span (Gram constants).
    // RHS is accumulated per state via factored 4-FMA inner loop.
    int ls_knot_count_;           // Knots in current interval's LeastSquares system
    int ls_interval_start_;       // Output index of current interval's first knot
    std::vector<double> ls_diag_;      // Main diagonal:  K scalars
    std::vector<double> ls_offdiag_;   // Off-diagonal:   (K-1) scalars
    std::vector<double> ls_rhs_;       // Right-hand side: K * S (state-interleaved)
    std::vector<double> ls_sum_A_;     // Per-state work buffer (S doubles)
    std::vector<double> ls_sum_B_;     // Per-state work buffer (S doubles)

    // Hermite mass matrix G_{ij} = ∫₀¹ hᵢ(u) hⱼ(u) du
    // Basis order: {h₀₀, h₁₀, h₀₁, h₁₁}
    // 10 unique entries of the 4×4 symmetric matrix.
    static constexpr double kG00 = 13.0/35.0;
    static constexpr double kG01 = 11.0/210.0;
    static constexpr double kG02 = 9.0/70.0;
    static constexpr double kG03 = -13.0/420.0;
    static constexpr double kG11 = 1.0/105.0;
    static constexpr double kG12 = 13.0/420.0;
    static constexpr double kG13 = -1.0/140.0;
    static constexpr double kG22 = 13.0/35.0;
    static constexpr double kG23 = -11.0/210.0;
    static constexpr double kG33 = 1.0/105.0;

    //  Hermite Basis: all 8 values at a normalized coordinate u

    /**
     * @brief All 8 Hermite basis function values at a normalized coordinate.
     *
     * For a merged span of length h with u = (t - t_anchor) / h:
     *
     *   Sensitivity basis (LeastSquares φ functions):
     *     phiA  = h₁₀(u)·h = (u**3 - 2u**2 + u)·h      sensitivity to anchor y'
     *     dphiA = h₁₀'(u)  = 3u**2 - 4u + 1            its derivative
     *     phiB  = h₁₁(u)·h = (u**3 - u**2)·h             sensitivity to candidate y'
     *     dphiB = h₁₁'(u)  = 3u**2 - 2u                 its derivative
     *
     *   Value interpolation basis (for knot error d_k = y_k - H(t_k)):
     *     cv0 = h₀₀(u) = 2u**3 - 3u**2 + 1               anchor value weight
     *     cv2 = h₀₁(u) = -2u**3 + 3u**2                   candidate value weight
     *     cd0 = h₀₀'(u)/h = 6(u**2-u)/h                 anchor value deriv weight
     *     cd2 = h₀₁'(u)/h = -6(u**2-u)/h = -cd0         candidate value deriv weight
     *
     * Note: cv1 = phiA, cd1 = dphiA, cv3 = phiB, cd3 = dphiB.
     */
    struct MergedBasis {
        double phiA, dphiA, phiB, dphiB;
        double cv0, cv2, cd0, cd2;

        static MergedBasis AtInterior(const double u, const double h, const double inv_h) {
            const double u2 = u * u, u3 = u2 * u;
            return {
                (u3 - 2.0*u2 + u) * h,       // phiA
                3.0*u2 - 4.0*u + 1.0,         // dphiA
                (u3 - u2) * h,                 // phiB
                3.0*u2 - 2.0*u,                // dphiB
                2.0*u3 - 3.0*u2 + 1.0,         // cv0
                -2.0*u3 + 3.0*u2,              // cv2
                6.0*(u2 - u) * inv_h,           // cd0
                -6.0*(u2 - u) * inv_h           // cd2
            };
        }

        static constexpr MergedBasis Anchor()    { return {0,1,0,0, 1,0,0,0}; }
        static constexpr MergedBasis Candidate() { return {0,0,0,1, 0,1,0,0}; }
    };

    //  Point Management

    void CommitPoint(sunrealtype t, const sunrealtype* y, const sunrealtype* yp) {
        LeastSquaresEnsureArrays();
        ++ls_knot_count_;
        
        out_t_->push_back(t);
        out_y_->insert(out_y_->end(), y, y + n_states_);
        out_yp_->insert(out_yp_->end(), yp, yp + n_states_);
        
        ++out_count_;
    }

    void SetAnchor(sunrealtype t, const sunrealtype* y, const sunrealtype* yp) {
        anchor_t_ = t;
        anchor_y_.assign(y, y + n_states_);
        anchor_yp_.assign(yp, yp + n_states_);
        has_anchor_ = true;

        window_t_.clear();
        window_y_.clear();
        window_yp_.clear();
    }

    void AddToWindow(sunrealtype t, const sunrealtype* y, const sunrealtype* yp) {
        window_t_.push_back(t);
        window_y_.insert(window_y_.end(), y, y + n_states_);
        window_yp_.insert(window_yp_.end(), yp, yp + n_states_);
    }

    /**
     * @brief Commit the current candidate (window.back()) after accumulating
     *        its LeastSquares span contributions. Does NOT clear the window.
     *
     * Extracts the repeated pattern: LeastSquaresAccumulateSpan + CommitPoint for the
     * last point in the window. After this call, window data is still valid
     * (only SetAnchor clears it).
     */
    void FlushCandidate() {
        const size_t last = window_t_.size() - 1;
        LeastSquaresAccumulateSpan(last);
        CommitPoint(window_t_[last], &window_y_[last * n_states_],
                    &window_yp_[last * n_states_]);
    }

    //  Greedy Merge Check (Bernstein-certified, three-level hierarchy)

    /**
     * @brief Compute knot errors d_k = y_k - H_merged(t_k) and d'_k via
     *        numerically stable Hermite basis evaluation in normalized coords.
     *
     * @param k       Window index of the knot (window_y_[k*n], window_yp_[k*n])
     * @param inv_hm  1 / h_merged
     * @param h_m     h_merged = t_new - t_anchor
     * @param ya   Anchor state array (n_states)
     * @param ypa  Anchor derivative array (n_states)
     * @param yn   New-point state array (n_states)
     * @param ypn  New-point derivative array (n_states)
     * @param dk   [out] Knot value errors (n_states)
     * @param dpk  [out] Knot derivative errors (n_states)
     */
    inline void ComputeKnotErrors(
            size_t k, sunrealtype inv_hm, sunrealtype h_m,
            const sunrealtype* PYBAMM_RESTRICT ya,
            const sunrealtype* PYBAMM_RESTRICT ypa,
            const sunrealtype* PYBAMM_RESTRICT yn,
            const sunrealtype* PYBAMM_RESTRICT ypn,
            sunrealtype* PYBAMM_RESTRICT dk,
            sunrealtype* PYBAMM_RESTRICT dpk) const
    {
        const int n = n_states_;
        const double u = (window_t_[k] - anchor_t_) * inv_hm;
        const MergedBasis B = MergedBasis::AtInterior(u, h_m, inv_hm);

        const sunrealtype* PYBAMM_RESTRICT yk  = &window_y_[k * n];
        const sunrealtype* PYBAMM_RESTRICT ypk = &window_yp_[k * n];

        #pragma omp simd
        for (int j = 0; j < n; ++j) {
            const sunrealtype hm_val = B.cv0*ya[j] + B.phiA*ypa[j]
                                     + B.cv2*yn[j] + B.phiB*ypn[j];
            const sunrealtype hm_deriv = B.cd0*ya[j] + B.dphiA*ypa[j]
                                       + B.cd2*yn[j] + B.dphiB*ypn[j];
            dk[j]  = yk[j]  - hm_val;
            dpk[j] = ypk[j] - hm_deriv;
        }
    }

    /**
     * @brief Sentinel pre-check: O(n_sentinels) lower bound on Level 1 sum.
     *
     * Checks the top-n_sentinels states (by inv_atol) for a certified lower bound.
     * If sentinels alone exceed threshold, the full sum certainly does too.
     * Valid because partial sum <= full sum.
     *
     * @return true if sentinels predict rejection (skip Level 1, go to Level 2)
     */
    inline bool CheckSentinels(
            const sunrealtype* PYBAMM_RESTRICT dk_left,
            const sunrealtype* PYBAMM_RESTRICT dpk_left,
            sunrealtype h_third) const
    {
        const int* PYBAMM_RESTRICT sent = sentinels_.data();
        const double* PYBAMM_RESTRICT inv_atol = inv_atol_.data();

        double sentinel_sum = 0.0;
        for (int s = 0; s < n_sentinels_; ++s) {
            const int j = sent[s];
            const sunrealtype abs_b0 = std::fabs(dk_left[j]);
            const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
            const sunrealtype err = std::fmax(abs_b0, abs_b1);
            const sunrealtype e = err * inv_atol[j];
            sentinel_sum += e * e;
        }
        return sentinel_sum > threshold_;
    }

    /**
     * @brief Level 1: Conservative Bernstein bound
     *
     * Uses inv_atol >= inv_w as upper bound on the WRMS weight.
     * First/last sub-intervals check only 2 control points (anchor/new-point
     * error is exactly zero).
     *
     * @return Sum of squared weighted Bernstein bounds across all states.
     */
    inline double CheckBernsteinConservative(
            const sunrealtype* PYBAMM_RESTRICT dk_left,
            const sunrealtype* PYBAMM_RESTRICT dpk_left,
            const sunrealtype* PYBAMM_RESTRICT dk_right,
            const sunrealtype* PYBAMM_RESTRICT dpk_right,
            sunrealtype h_third, bool is_first, bool is_last) const
    {
        const int n = n_states_;
        const double* PYBAMM_RESTRICT inv_atol = inv_atol_.data();
        double sum_sq = 0.0;

        if (is_first) {
            // Anchor error is zero: beta0 = beta1 = 0
            #pragma omp simd reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                const sunrealtype abs_b2 = std::fabs(dk_right[j] - dpk_right[j] * h_third);
                const sunrealtype abs_b3 = std::fabs(dk_right[j]);
                const sunrealtype err = std::fmax(abs_b2, abs_b3);
                const sunrealtype e = err * inv_atol[j];
                sum_sq += e * e;
            }
        } else if (is_last) {
            // New-point error is zero: beta2 = beta3 = 0
            #pragma omp simd reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                const sunrealtype abs_b0 = std::fabs(dk_left[j]);
                const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
                const sunrealtype err = std::fmax(abs_b0, abs_b1);
                const sunrealtype e = err * inv_atol[j];
                sum_sq += e * e;
            }
        } else {
            // General: all 4 Bernstein control points
            #pragma omp simd reduction(+:sum_sq)
            for (int j = 0; j < n; ++j) {
                const sunrealtype abs_b0 = std::fabs(dk_left[j]);
                const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
                const sunrealtype abs_b2 = std::fabs(dk_right[j] - dpk_right[j] * h_third);
                const sunrealtype abs_b3 = std::fabs(dk_right[j]);
                const sunrealtype err = std::fmax(std::fmax(abs_b0, abs_b1),
                                                  std::fmax(abs_b2, abs_b3));
                const sunrealtype e = err * inv_atol[j];
                sum_sq += e * e;
            }
        }
        return sum_sq;
    }

    /**
     * @brief Level 2: Bernstein bound with exact WRMS weight (rare path).
     *
     * Uses fmin(|y_left|, |y_right|) for conservative weight certification.
     * Only entered when Level 1's conservative bound exceeds the threshold.
     *
     * @return Sum of squared WRMS-weighted Bernstein bounds across all states.
     */
    inline double CheckBernsteinExactWRMS(
            const sunrealtype* PYBAMM_RESTRICT dk_left,
            const sunrealtype* PYBAMM_RESTRICT dpk_left,
            const sunrealtype* PYBAMM_RESTRICT dk_right,
            const sunrealtype* PYBAMM_RESTRICT dpk_right,
            sunrealtype h_third,
            const sunrealtype* PYBAMM_RESTRICT y_left,
            const sunrealtype* PYBAMM_RESTRICT y_right) const
    {
        const int n = n_states_;
        const double* PYBAMM_RESTRICT inv_atol = inv_atol_.data();
        const double rtol = rtol_;
        double sum_sq = 0.0;

        #pragma omp simd reduction(+:sum_sq)
        for (int j = 0; j < n; ++j) {
            const sunrealtype abs_b0 = std::fabs(dk_left[j]);
            const sunrealtype abs_b1 = std::fabs(dk_left[j] + dpk_left[j] * h_third);
            const sunrealtype abs_b2 = std::fabs(dk_right[j] - dpk_right[j] * h_third);
            const sunrealtype abs_b3 = std::fabs(dk_right[j]);
            const sunrealtype err = std::fmax(std::fmax(abs_b0, abs_b1),
                                              std::fmax(abs_b2, abs_b3));
            const sunrealtype ymin = std::fmin(std::fabs(y_left[j]),
                                               std::fabs(y_right[j]));
            const sunrealtype inv_w = inv_atol[j] / (1.0 + rtol * ymin * inv_atol[j]);
            const sunrealtype e = err * inv_w;
            sum_sq += e * e;
        }
        return sum_sq;
    }

    /**
     * @brief Level 3: De Casteljau midpoint subdivision tightening (very rare).
     *
     * Splits each Bernstein curve at the midpoint, producing 8 refined control
     * points whose convex hull is ~4x tighter. Uses exact WRMS weight.
     *
     * @return Sum of squared WRMS-weighted refined Bernstein bounds.
     */
    inline double CheckDeCasteljauRefinement(
            const sunrealtype* PYBAMM_RESTRICT dk_left,
            const sunrealtype* PYBAMM_RESTRICT dpk_left,
            const sunrealtype* PYBAMM_RESTRICT dk_right,
            const sunrealtype* PYBAMM_RESTRICT dpk_right,
            sunrealtype h_third,
            const sunrealtype* PYBAMM_RESTRICT y_left,
            const sunrealtype* PYBAMM_RESTRICT y_right) const
    {
        const int n = n_states_;
        const double* PYBAMM_RESTRICT inv_atol = inv_atol_.data();
        const double rtol = rtol_;
        double sum_sq = 0.0;

        #pragma omp simd reduction(+:sum_sq)
        for (int j = 0; j < n; ++j) {
            // Apply the absolute value to the Bernstein control points
            const sunrealtype B0_abs = std::fabs(dk_left[j]);
            const sunrealtype B1_abs = std::fabs(dk_left[j] + dpk_left[j] * h_third);
            const sunrealtype B2_abs = std::fabs(dk_right[j] - dpk_right[j] * h_third);
            const sunrealtype B3_abs = std::fabs(dk_right[j]);

            // The Li and Ri are already their absolute values since we
            // only add and multiply by positive constants
            const sunrealtype L1_abs = 0.5 * (B0_abs + B1_abs);
            const sunrealtype L2_abs = 0.25 * (B0_abs + 2.0 * B1_abs + B2_abs);
            const sunrealtype L3_abs = 0.125 * (B0_abs + 3.0 * (B1_abs + B2_abs) + B3_abs);

            const sunrealtype R1_abs = 0.25 * (B1_abs + 2.0 * B2_abs + B3_abs);
            const sunrealtype R2_abs = 0.5 * (B2_abs + B3_abs);

            const sunrealtype max_left = std::fmax(
                std::fmax(B0_abs, L1_abs),
                std::fmax(L2_abs, L3_abs));
            const sunrealtype max_right = std::fmax(
                std::fmax(L3_abs, R1_abs),
                std::fmax(R2_abs, B3_abs));
            const sunrealtype err = std::fmax(max_left, max_right);

            const sunrealtype ymin = std::fmin(std::fabs(y_left[j]),
                                               std::fabs(y_right[j]));
            const sunrealtype inv_w = inv_atol[j] / (1.0 + rtol * ymin * inv_atol[j]);
            const sunrealtype e = err * inv_w;
            sum_sq += e * e;
        }
        return sum_sq;
    }

    /**
     * @brief Three-level Bernstein-certified merge check with sentinel early exit.
     *
     * For the error polynomial d(s) on sub-interval [t_k, t_{k+1}] with step h:
     *   beta0 = d_k,  beta1 = d_k + d'_k*h/3,
     *   beta2 = d_{k+1} - d'_{k+1}*h/3,  beta3 = d_{k+1}
     *
     * Convex hull property: max|d(s)| <= max(|beta_i|) for s in [0, h]
     *
     * SENTINEL EARLY EXIT:
     *   Before the full O(n) SIMD loop, check sentinel states (top-32 by
     *   inv_atol). Their partial WRMS sum is a certified lower bound on the
     *   full sum. If sentinels alone exceed threshold, skip O(n) Level 1
     *   entirely and proceed to Level 2. Zero risk: subset sum <= full sum.
     *
     * Knot errors (d_k, d'_k) are evaluated ONCE per knot using numerically
     * stable Hermite basis functions in normalized coordinates, then shared
     * between adjacent sub-intervals via double buffering.
     */
    bool CanMergeWindow(const sunrealtype t_new, const sunrealtype* y_new, const sunrealtype* yp_new) {
        const int n = n_states_;

        const sunrealtype h_merged = t_new - anchor_t_;
        const sunrealtype inv_hm = 1.0 / h_merged;

        const sunrealtype* PYBAMM_RESTRICT ya = anchor_y_.data();
        const sunrealtype* PYBAMM_RESTRICT ypa = anchor_yp_.data();

        // Double-buffered knot error caches
        sunrealtype* PYBAMM_RESTRICT dk_left = dk_buf0_.data();
        sunrealtype* PYBAMM_RESTRICT dpk_left = dpk_buf0_.data();
        sunrealtype* PYBAMM_RESTRICT dk_right = dk_buf1_.data();
        sunrealtype* PYBAMM_RESTRICT dpk_right = dpk_buf1_.data();

        const size_t window_size = window_t_.size();

        for (size_t k = 0; k <= window_size; ++k) {
            const bool is_first = (k == 0);
            const sunrealtype t_left = is_first ? anchor_t_ : window_t_[k - 1];
            const sunrealtype* PYBAMM_RESTRICT y_left = is_first ? ya : &window_y_[(k - 1) * n];

            const bool is_last = (k == window_size);
            const sunrealtype t_right = is_last ? t_new : window_t_[k];
            const sunrealtype* PYBAMM_RESTRICT y_right = is_last ? y_new : &window_y_[k * n];

            const sunrealtype h_sub = t_right - t_left;

            // Skip degenerate sub-intervals
            if (h_sub <= 0.0) {
                if (!is_last) {
                    std::swap(dk_left, dk_right);
                    std::swap(dpk_left, dpk_right);
                }
                continue;
            }

            const sunrealtype h_third = h_sub / 3.0;

            // Sentinel pre-check
            bool skip_level1 = false;
            if (!is_first && n_sentinels_ > 0) {
                if (CheckSentinels(dk_left, dpk_left, h_third)) {
                    if (!is_last) {
                        ComputeKnotErrors(k, inv_hm, h_merged,
                                          ya, ypa, y_new, yp_new,
                                          dk_right, dpk_right);
                    }
                    skip_level1 = true;
                }
            }

            if (!skip_level1) {
                if (!is_last) {
                    ComputeKnotErrors(k, inv_hm, h_merged,
                                      ya, ypa, y_new, yp_new,
                                      dk_right, dpk_right);
                }

                // Level 1: Conservative Bernstein (division-free)
                const double l1_sum = CheckBernsteinConservative(dk_left, dpk_left,
                                                  dk_right, dpk_right,
                                                  h_third, is_first, is_last);
                if (l1_sum <= threshold_) {
                    std::swap(dk_left, dk_right);
                    std::swap(dpk_left, dpk_right);
                    continue;
                }
            }

            // Level 2: Bernstein with exact WRMS weight (rare path)
            // Deferred memset: only zero the endpoint buffers on this rare path.
            if (is_first) {
                std::memset(dk_left, 0, n * sizeof(sunrealtype));
                std::memset(dpk_left, 0, n * sizeof(sunrealtype));
            }
            if (is_last) {
                std::memset(dk_right, 0, n * sizeof(sunrealtype));
                std::memset(dpk_right, 0, n * sizeof(sunrealtype));
            }

            const double l2_sum = CheckBernsteinExactWRMS(dk_left, dpk_left,
                                              dk_right, dpk_right,
                                              h_third, y_left, y_right);

            if (l2_sum > threshold_) {
                // Level 3: De Casteljau midpoint subdivision.
                if (l2_sum > 9.0 * threshold_) {
                    return false;
                }
                const double l3_sum = CheckDeCasteljauRefinement(dk_left, dpk_left,
                                                  dk_right, dpk_right,
                                                  h_third, y_left, y_right);
                if (l3_sum > threshold_) {
                    return false;
                }
            }

            std::swap(dk_left, dk_right);
            std::swap(dpk_left, dpk_right);
        }

        return true;
    }

    //  LeastSquares Derivative Refinement: Streaming Accumulation + LU Solve

    /**
     * @brief Ensure LeastSquares arrays are sized for the current knot count.
     *
     * Resizes diag, offdiag, and rhs arrays to accommodate ls_knot_count_ + 1
     * entries. New entries are zero-initialized
     */
    void LeastSquaresEnsureArrays() {
        const size_t kp1 = static_cast<size_t>(ls_knot_count_ + 1);
        ls_diag_.resize(kp1, 0.0);
        ls_rhs_.resize(kp1 * n_states_, 0.0);
        if (ls_knot_count_ > 0) {
            ls_offdiag_.resize(static_cast<size_t>(ls_knot_count_), 0.0);
        }
    }

    /**
     * @brief Accumulate LeastSquares matrix entries (O(1) per span, closed-form).
     *
     * Adds the Hermite mass matrix contributions for a span of length h_span:
     *   diag[k-1] += h**3/105,  diag[k] += h**3/105,  offdiag[k-1] += -h**3/140
     *
     * Used for both full spans (via LeastSquaresAccumulateSpan) and empty spans with
     * no interior points (e.g., candidate→breakpoint or anchor→breakpoint).
     *
     * @param h_span  Length of the span
     */
    void LeastSquaresAccumulateMatrix(const sunrealtype h_span) {
        if (ls_knot_count_ == 0 || h_span <= 0.0) return;
        LeastSquaresEnsureArrays();
        const int k = ls_knot_count_;
        const double h3 = h_span * h_span * h_span;
        ls_diag_[k - 1]    += h3 * kG11;   // h**3/105
        ls_diag_[k]        += h3 * kG33;   // h**3/105
        ls_offdiag_[k - 1] += h3 * kG13;   // -h**3/140
    }

    /**
     * @brief Accumulate LeastSquares matrix + RHS for a closing span.
     *
     * Thin dispatcher: accumulates the closed-form matrix entries, then
     * (if there are interior points) the factored RHS contributions.
     *
     * Called just BEFORE CommitPoint for window[last].
     *
     * @param last  Index of the candidate in the window (window.back())
     */
    void LeastSquaresAccumulateSpan(const size_t last) {
        if (ls_knot_count_ == 0) return;
        const sunrealtype h = window_t_[last] - anchor_t_;
        LeastSquaresAccumulateMatrix(h);
        if (last > 0 && h > 0.0) {
            LeastSquaresAccumulateRHS(last, h);
        }
    }

    /**
     * @brief Accumulate LeastSquares RHS via factored sliding-window accumulation.
     *
     * Processes interior points p = 0..last-1 in a single pass with a
     * prev → curr → next sliding window of MergedBasis values.
     *
     * For each interior point, computes Gram-weighted per-node coefficients
     * from the left [prev,p] and right [p,next] sub-intervals, then:
     *   - Updates scalar accumulators for constant endpoint correction
     *   - Accumulates per-state sums via 4 FMA (the dominant cost)
     *
     * At span closure, a single finalization pass combines the per-state
     * sums with the scalar corrections to form the final RHS entries.
     *
     * @param last  Index of the candidate (window.back())
     * @param h     Merged span length (t_candidate - t_anchor, must be > 0)
     */
    void LeastSquaresAccumulateRHS(const size_t last, const sunrealtype h) {
        const int S = n_states_;
        const int k = ls_knot_count_;
        const int km1 = k - 1;
        const sunrealtype inv_h = 1.0 / h;

        const sunrealtype* PYBAMM_RESTRICT ya  = anchor_y_.data();
        const sunrealtype* PYBAMM_RESTRICT ypa = anchor_yp_.data();
        const sunrealtype* PYBAMM_RESTRICT yr  = &window_y_[last * S];
        const sunrealtype* PYBAMM_RESTRICT ypr = &window_yp_[last * S];

        double* PYBAMM_RESTRICT r_left  = &ls_rhs_[static_cast<size_t>(km1) * S];
        double* PYBAMM_RESTRICT r_right = &ls_rhs_[static_cast<size_t>(k) * S];

        // Zero per-state work buffers
        double* PYBAMM_RESTRICT sum_A = ls_sum_A_.data();
        double* PYBAMM_RESTRICT sum_B = ls_sum_B_.data();
        std::memset(sum_A, 0, static_cast<size_t>(S) * sizeof(double));
        std::memset(sum_B, 0, static_cast<size_t>(S) * sizeof(double));

        // Scalar accumulators for endpoint correction (hoisted out of state loop)
        double s_ya_A = 0, s_ypa_A = 0, s_yr_A = 0, s_ypr_A = 0;
        double s_ya_B = 0, s_ypa_B = 0, s_yr_B = 0, s_ypr_B = 0;

        // Sliding window: prev → curr → next
        MergedBasis prev = MergedBasis::Anchor();
        double t_prev = anchor_t_;

        const double u0 = (window_t_[0] - anchor_t_) * inv_h;
        MergedBasis curr = MergedBasis::AtInterior(u0, h, inv_h);

        for (size_t p = 0; p < last; ++p) {
            // Lookahead: compute next point's basis
            const double t_next = (p + 1 < last) ? window_t_[p + 1] : window_t_[last];
            const MergedBasis next = (p + 1 < last)
                ? MergedBasis::AtInterior((t_next - anchor_t_) * inv_h, h, inv_h)
                : MergedBasis::Candidate();

            // Sub-interval lengths
            const double h_L = window_t_[p] - t_prev;
            const double h_R = t_next - window_t_[p];

            // Left sub-interval [prev, p]: Gram-weighted rows 2,3
            const double aL0 = prev.phiA,  aL1 = h_L * prev.dphiA;
            const double aL2 = curr.phiA,  aL3 = h_L * curr.dphiA;
            const double qAL2 = kG02*aL0 + kG12*aL1 + kG22*aL2 + kG23*aL3;
            const double qAL3 = kG03*aL0 + kG13*aL1 + kG23*aL2 + kG33*aL3;

            const double bL0 = prev.phiB,  bL1 = h_L * prev.dphiB;
            const double bL2 = curr.phiB,  bL3 = h_L * curr.dphiB;
            const double qBL2 = kG02*bL0 + kG12*bL1 + kG22*bL2 + kG23*bL3;
            const double qBL3 = kG03*bL0 + kG13*bL1 + kG23*bL2 + kG33*bL3;

            // Right sub-interval [p, next]: Gram-weighted rows 0,1
            const double aR0 = curr.phiA,  aR1 = h_R * curr.dphiA;
            const double aR2 = next.phiA,  aR3 = h_R * next.dphiA;
            const double qAR0 = kG00*aR0 + kG01*aR1 + kG02*aR2 + kG03*aR3;
            const double qAR1 = kG01*aR0 + kG11*aR1 + kG12*aR2 + kG13*aR3;

            const double bR0 = curr.phiB,  bR1 = h_R * curr.dphiB;
            const double bR2 = next.phiB,  bR3 = h_R * next.dphiB;
            const double qBR0 = kG00*bR0 + kG01*bR1 + kG02*bR2 + kG03*bR3;
            const double qBR1 = kG01*bR0 + kG11*bR1 + kG12*bR2 + kG13*bR3;

            // Combined per-node coefficients
            const double c_d_A  = h_L * qAL2 + h_R * qAR0;
            const double c_dp_A = h_L * h_L * qAL3 + h_R * h_R * qAR1;
            const double c_d_B  = h_L * qBL2 + h_R * qBR0;
            const double c_dp_B = h_L * h_L * qBL3 + h_R * h_R * qBR1;

            // Scalar accumulators: absorb constant endpoint data
            s_ya_A  += c_d_A * curr.cv0 + c_dp_A * curr.cd0;
            s_ypa_A += c_d_A * curr.phiA + c_dp_A * curr.dphiA;  // cv1=phiA, cd1=dphiA
            s_yr_A  += c_d_A * curr.cv2 + c_dp_A * curr.cd2;
            s_ypr_A += c_d_A * curr.phiB + c_dp_A * curr.dphiB;  // cv3=phiB, cd3=dphiB

            s_ya_B  += c_d_B * curr.cv0 + c_dp_B * curr.cd0;
            s_ypa_B += c_d_B * curr.phiA + c_dp_B * curr.dphiA;
            s_yr_B  += c_d_B * curr.cv2 + c_dp_B * curr.cd2;
            s_ypr_B += c_d_B * curr.phiB + c_dp_B * curr.dphiB;

            // Per-state accumulation (4 FMA per state)
            const sunrealtype* PYBAMM_RESTRICT y_p  = &window_y_[p * S];
            const sunrealtype* PYBAMM_RESTRICT yp_p = &window_yp_[p * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                sum_A[s] += c_d_A * y_p[s] + c_dp_A * yp_p[s];
                sum_B[s] += c_d_B * y_p[s] + c_dp_B * yp_p[s];
            }

            // Advance sliding window
            t_prev = window_t_[p];
            prev = curr;
            curr = next;
        }

        // Finalize: RHS = accumulated sums - scalar corrections × endpoints
        #pragma omp simd
        for (int s = 0; s < S; ++s) {
            r_left[s]  += sum_A[s] - s_ya_A * ya[s] - s_ypa_A * ypa[s]
                                   - s_yr_A * yr[s] - s_ypr_A * ypr[s];
            r_right[s] += sum_B[s] - s_ya_B * ya[s] - s_ypa_B * ypa[s]
                                   - s_yr_B * yr[s] - s_ypr_B * ypr[s];
        }
    }

    /**
     * @brief Solve the tri-diagonal LeastSquares system and update yp.
     *
     * LU algorithm for the SPD tri-diagonal system arising from the
     * integral L2 objective.
     *
     * Because the normal-equation matrix is state-independent, the LU
     * factorization is O(K) scalar operations. The scalar multiplier m[k]
     * is then broadcast to all S states during the forward RHS sweep and
     * back-substitution (O(K*S)).
     *
     * The integral objective guarantees the matrix is always SPD:
     *   - diag[k] = h_left**3/105 + h_right**3/105 > 0  for all k
     *   - Strict diagonal dominance: 1/105 > 1/140
     *   - No zero-diagonal checks needed
     */
    void LeastSquaresSolveAndUpdate() {
        const int K = ls_knot_count_;
        const int S = n_states_;

        if (K < 2) return;

        // For non-first intervals, knot 0 was already updated by the previous
        // interval's LeastSquares solve (as its last knot). Fix δ_0 = 0 to prevent
        // double-update: zero row 0 and decouple it from knot 1.
        if (ls_interval_start_ > 0) {
            ls_diag_[0] = 1.0;
            ls_offdiag_[0] = 0.0;
            std::memset(&ls_rhs_[0], 0, static_cast<size_t>(S) * sizeof(double));
        }

        // 1. LU factorization + forward RHS sweep
        for (int k = 1; k < K; ++k) {
            const double m = ls_offdiag_[k - 1] / ls_diag_[k - 1];
            ls_diag_[k] -= m * ls_offdiag_[k - 1];

            const double* PYBAMM_RESTRICT r_prev = &ls_rhs_[static_cast<size_t>(k - 1) * S];
            double* PYBAMM_RESTRICT r_curr       = &ls_rhs_[static_cast<size_t>(k) * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                r_curr[s] -= m * r_prev[s];
            }
        }

        // 2. Fused back-substitution + yp update
        sunrealtype* yp_data = out_yp_->data()
                             + static_cast<size_t>(ls_interval_start_) * S;

        // 3. Last knot
        {
            const double d_inv = 1.0 / ls_diag_[K - 1];
            double* PYBAMM_RESTRICT x_last = &ls_rhs_[static_cast<size_t>(K - 1) * S];
            sunrealtype* PYBAMM_RESTRICT yp_k = &yp_data[static_cast<size_t>(K - 1) * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                x_last[s] *= d_inv;
                yp_k[s] += static_cast<sunrealtype>(x_last[s]);
            }
        }

        // 4. Remaining knots (backward)
        for (int k = K - 2; k >= 0; --k) {
            const double off_k = ls_offdiag_[k];
            const double d_inv = 1.0 / ls_diag_[k];
            double* PYBAMM_RESTRICT x_curr       = &ls_rhs_[static_cast<size_t>(k) * S];
            const double* PYBAMM_RESTRICT x_next = &ls_rhs_[static_cast<size_t>(k + 1) * S];
            sunrealtype* PYBAMM_RESTRICT yp_k    = &yp_data[static_cast<size_t>(k) * S];

            #pragma omp simd
            for (int s = 0; s < S; ++s) {
                x_curr[s] = (x_curr[s] - off_k * x_next[s]) * d_inv;
                yp_k[s] += static_cast<sunrealtype>(x_curr[s]);
            }
        }
    }
};

#endif // HERMITE_KNOT_REDUCER_HPP
