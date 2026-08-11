//
// Brent's method on a bracket, written in CasADi's runtime style (a `T1` template
// that `CodeGenerator::sanitize_source` turns into plain C).
//
// This is the single source of the iteration: `Brent::solve` compiles it directly and
// `Brent::codegen_declarations` emits the very same text into generated C, so the
// interpreted and generated paths cannot drift. Keep it valid C once the template
// header and the comments are stripped: declarations first, no C++-only constructs,
// and no preprocessor directives (`sanitize_source` drops `#define`/`#undef` lines).
//
// `res_fn` evaluates the residual at `x` into `*fx` and returns nonzero on failure.
// Returns 0 on success (`*out` is the root, `*iter` the iteration count), 1 if the
// residual failed, 2 if no bracket could be found, 3 if the bracket collapsed on a
// point that is not a root -- a sign change across a pole rather than a zero, which
// bisection converges to just as happily. `*out` always carries the best point seen,
// so a caller that would rather have an answer than an error can take it.
//

// `static` so that two generated files, each carrying its own copy, still link.
template<typename T1>
static int casadi_brent_core(int (*res_fn)(void*, T1, T1*), void* user_data,
                             T1 a, T1 fa, T1 b, T1 fb, T1 abstol, T1 ftol,
                             casadi_int max_iter, T1* out, casadi_int* iter) {
  T1 c, fc, d, e, tol, xm, p, q, s, qq, r, step;
  casadi_int k;
  c = a;
  fc = fa;
  d = b - a;
  e = d;
  for (k = 0; k < max_iter; ++k) {
    if (fb * fc > 0) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
    if (fabs(fc) < fabs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    // 2.2204460492503131e-16 is DBL_EPSILON, spelled out because the generated C
    // cannot reach std::numeric_limits.
    tol = 2 * 2.2204460492503131e-16 * fabs(b) + 0.5 * abstol;
    xm = 0.5 * (c - b);
    if (fabs(xm) <= tol || fb == 0) break;
    if (fabs(e) >= tol && fabs(fa) > fabs(fb)) {
      // Inverse quadratic interpolation, or secant when only two points are distinct
      s = fb / fa;
      if (a == c) {
        p = 2 * xm * s;
        q = 1 - s;
      } else {
        qq = fa / fc;
        r = fb / fc;
        p = s * (2 * xm * qq * (qq - r) - (b - a) * (r - 1));
        q = (qq - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q;
      p = fabs(p);
      // Take the interpolated step only while it keeps bisecting; else bisect. A zero
      // q fails this test, so p / q below is never reached with a zero denominator.
      if (2 * p < fmin(3 * xm * q - fabs(tol * q), fabs(e * q))) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    a = b;
    fa = fb;
    step = fabs(d) > tol ? d : (xm > 0 ? tol : -tol);
    b += step;
    if (res_fn(user_data, b, &fb)) return 1;
  }
  *iter = k;
  *out = b;
  // Brent converges on a sign change, which a pole provides as readily as a root.
  // The bracket having collapsed is therefore not enough; the residual has to be
  // small there too.
  if (!(fabs(fb) <= ftol)) return 3;
  return 0;
}

// Brent, preceded by a search for the bracket when `max_expansions` allows it.
//
// With `max_expansions` zero, `a` and `b` must already bracket a root; this is the
// plain method. Otherwise they are only a starting scale: if they do not bracket, the
// search walks outwards in geometrically growing steps until the sign changes. For a
// monotonic residual that walk is unambiguous -- the root lies past whichever end has
// the smaller residual -- so no direction has to be supplied or guessed.
template<typename T1>
static int casadi_brent(int (*res_fn)(void*, T1, T1*), void* user_data,
                        T1 x0, T1 a, T1 b, T1 abstol, T1 ftol, casadi_int max_iter,
                        casadi_int max_expansions, T1* out, casadi_int* iter) {
  T1 fa, fb, p, fp, q, fq, fbest, step;
  casadi_int k;
  *iter = 0;
  *out = a;
  if (res_fn(user_data, a, &fa)) return 1;
  fbest = fabs(fa);
  if (res_fn(user_data, b, &fb)) return 1;
  if (fabs(fb) < fbest) {
    fbest = fabs(fb);
    *out = b;
  }
  if (fa * fb <= 0) return casadi_brent_core(res_fn, user_data, a, fa, b, fb, abstol,
                                             ftol, max_iter, out, iter);
  // No sign change over the bracket. Without expansion that is the answer: report it
  // rather than returning whichever end happens to be closer.
  if (max_expansions <= 0) return 2;

  // Walk out from the end whose residual is smaller, which for a monotonic residual
  // is the one the root lies past. The guess only sets the size of the first step.
  step = fabs(b - a);
  if (!(step > 0)) step = 1;
  if (x0 > a && x0 < b) step = fmax(fabs(x0 - (fabs(fa) <= fabs(fb) ? a : b)), abstol);
  if (fabs(fa) <= fabs(fb)) {
    p = a;
    fp = fa;
    step = -step;
  } else {
    p = b;
    fp = fb;
  }

  for (k = 0; k < max_expansions; ++k) {
    q = p + step;
    if (res_fn(user_data, q, &fq)) return 1;
    // Expansion walks into extrapolation, where a residual can overflow. Back off
    // rather than treating the overflow as a value.
    if (!isfinite(fq)) {
      step = 0.5 * step;
      continue;
    }
    if (fabs(fq) < fbest) {
      fbest = fabs(fq);
      *out = q;
    }
    // the core does not care which end is which, only that they bracket
    if (fp * fq <= 0) return casadi_brent_core(res_fn, user_data, p, fp, q, fq, abstol,
                                               ftol, max_iter, out, iter);
    p = q;
    fp = fq;
    step = 2 * step;
  }
  return 2;
}
