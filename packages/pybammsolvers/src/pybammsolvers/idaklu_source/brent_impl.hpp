//
// Brent's method on a bracket, written in CasADi's runtime style (a `T1` template
// that `CodeGenerator::sanitize_source` turns into plain C).
//
// This text is both compiled and stringified into the generated C, so keep it valid C
// once the template header and comments are stripped: declarations first, no C++-only
// constructs, no preprocessor directives (`sanitize_source` drops `#define`/`#undef`).
//
// `res_fn` evaluates the residual at `x` into `*fx` and returns nonzero on failure.
// Returns 0 on success (`*out` is the root, `*iter` the iteration count), 1 if the
// residual failed, 2 if the bracket shows no sign change, 3 if `max_iter` was reached
// without the bracket shrinking to `abstol`.
//

// `static` so that two generated files, each carrying its own copy, still link.
template<typename T1>
static int casadi_brent(int (*res_fn)(void*, T1, T1*), void* user_data,
                        T1 a, T1 b, T1 abstol, casadi_int max_iter,
                        T1* out, casadi_int* iter) {
  T1 fa, fb, c, fc, d, e, tol, xm, p, q, s, qq, r, step;
  casadi_int k;
  *iter = 0;
  if (res_fn(user_data, a, &fa)) return 1;
  if (res_fn(user_data, b, &fb)) return 1;
  // No sign change, so the bracket contains no root. Report it rather than
  // returning whichever end happens to be closer.
  if (!(fa * fb <= 0)) return 2;
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
    // DBL_EPSILON, spelled out: the generated C includes no headers of its own
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
      // Take the interpolated step only while it keeps bisecting; else bisect
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
  // Falling out of the loop means the bracket never shrank to `abstol`, so `b` is
  // not a root. Reporting it as one would hand back an arbitrary point silently.
  if (k >= max_iter) return 3;
  *out = b;
  return 0;
}
