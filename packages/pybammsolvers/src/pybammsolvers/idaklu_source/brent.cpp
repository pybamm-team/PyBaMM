#include "brent.hpp"

#include <algorithm>
#include <cmath>

namespace casadi {

extern "C" int CASADI_ROOTFINDER_BRENT_EXPORT
casadi_register_rootfinder_brent(Rootfinder::Plugin* plugin) {
  plugin->creator = Brent::creator;
  plugin->name = "brent";
  plugin->doc = Brent::meta_doc.c_str();
  plugin->version = CASADI_VERSION;
  plugin->options = &Brent::options_;
  plugin->deserialize = &Brent::deserialize;
  return 0;
}

extern "C" void CASADI_ROOTFINDER_BRENT_EXPORT casadi_load_rootfinder_brent() {
  Rootfinder::registerPlugin(casadi_register_rootfinder_brent);
}

const std::string Brent::meta_doc =
  "Brent's method for a scalar residual on a bracket. Derivative free, and the "
  "iterate never leaves the bracket.";

const Options Brent::options_ = {
  {&Rootfinder::options_},
  {{"lo", {OT_DOUBLE, "Lower end of the bracket"}},
   {"hi", {OT_DOUBLE, "Upper end of the bracket"}},
   {"abstol", {OT_DOUBLE, "Absolute tolerance on the unknown"}},
   {"max_iter", {OT_INT, "Maximum number of iterations"}},
   {"lo_index", {OT_INT, "Index of the input carrying the lower end of the bracket"}},
   {"hi_index", {OT_INT, "Index of the input carrying the upper end of the bracket"}}}};

void Brent::init(const Dict& opts) {
  Rootfinder::init(opts);

  for (auto&& op : opts) {
    if (op.first == "lo") {
      lo_ = op.second;
    } else if (op.first == "hi") {
      hi_ = op.second;
    } else if (op.first == "abstol") {
      abstol_ = op.second;
    } else if (op.first == "max_iter") {
      max_iter_ = op.second;
    } else if (op.first == "lo_index") {
      lo_index_ = op.second;
    } else if (op.first == "hi_index") {
      hi_index_ = op.second;
    }
  }

  casadi_assert(n_ == 1, "Brent solves a scalar residual, got n=" + str(n_));
  casadi_assert(max_iter_ > 0, "max_iter must be positive, got " + str(max_iter_));
  casadi_assert(abstol_ > 0, "abstol must be positive, got " + str(abstol_));

  set_function(oracle_, "g");
  alloc_w(2, true);
}

int Brent::init_mem(void* mem) const {
  if (Rootfinder::init_mem(mem)) return 1;
  auto m = static_cast<BrentMemory*>(mem);
  m->iter = 0;
  m->return_status = "unset";
  return 0;
}

void Brent::set_work(void* mem, const double**& arg, double**& res, casadi_int*& iw,
                     double*& w) const {
  Rootfinder::set_work(mem, arg, res, iw, w);
  auto m = static_cast<BrentMemory*>(mem);
  m->x = w;
  w += 1;
  m->f = w;
  w += 1;
}

int Brent::solve(void* mem) const {
  auto m = static_cast<BrentMemory*>(mem);

  auto g = [&](double x) -> double {
    std::copy_n(m->iarg, n_in_, m->arg);
    m->arg[iin_] = &x;
    std::copy_n(m->ires, n_out_, m->res);
    m->res[iout_] = m->f;
    calc_function(m, "g");
    return m->f[0];
  };

  double a = lo_index_ >= 0 ? m->iarg[lo_index_][0] : lo_;
  double b = hi_index_ >= 0 ? m->iarg[hi_index_][0] : hi_;
  double fa = g(a), fb = g(b);

  if (!(fa * fb <= 0)) {
    // No sign change, so the bracket contains no root. Report it rather than
    // returning whichever end happens to be closer.
    m->return_status = "no sign change over the bracket";
    m->unified_return_status = SOLVER_RET_UNKNOWN;
    m->success = false;
    return 0;
  }

  double c = a, fc = fa, d = b - a, e = d;
  for (m->iter = 0; m->iter < max_iter_; ++m->iter) {
    if (fb * fc > 0) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
    if (std::fabs(fc) < std::fabs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    const double tol = 2 * std::numeric_limits<double>::epsilon() * std::fabs(b)
                       + 0.5 * abstol_;
    const double xm = 0.5 * (c - b);
    if (std::fabs(xm) <= tol || fb == 0) break;

    if (std::fabs(e) >= tol && std::fabs(fa) > std::fabs(fb)) {
      // Inverse quadratic interpolation, or secant when only two points are distinct
      double p, q;
      const double s = fb / fa;
      if (a == c) {
        p = 2 * xm * s;
        q = 1 - s;
      } else {
        const double qq = fa / fc, r = fb / fc;
        p = s * (2 * xm * qq * (qq - r) - (b - a) * (r - 1));
        q = (qq - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q;
      p = std::fabs(p);
      // Take the interpolated step only while it keeps bisecting; else bisect
      if (2 * p < std::fmin(3 * xm * q - std::fabs(tol * q), std::fabs(e * q))) {
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
    b += std::fabs(d) > tol ? d : (xm > 0 ? tol : -tol);
    fb = g(b);
  }

  m->x[0] = b;
  m->ires[iout_][0] = m->x[0];
  m->return_status = "success";
  m->success = true;
  return 0;
}

void Brent::serialize_body(SerializingStream& s) const {
  Rootfinder::serialize_body(s);
  s.version("Brent", 1);
  s.pack("Brent::lo", lo_);
  s.pack("Brent::hi", hi_);
  s.pack("Brent::abstol", abstol_);
  s.pack("Brent::max_iter", max_iter_);
  s.pack("Brent::lo_index", lo_index_);
  s.pack("Brent::hi_index", hi_index_);
}

Brent::Brent(DeserializingStream& s) : Rootfinder(s) {
  s.version("Brent", 1);
  s.unpack("Brent::lo", lo_);
  s.unpack("Brent::hi", hi_);
  s.unpack("Brent::abstol", abstol_);
  s.unpack("Brent::max_iter", max_iter_);
  s.unpack("Brent::lo_index", lo_index_);
  s.unpack("Brent::hi_index", hi_index_);
}

Dict Brent::get_stats(void* mem) const {
  Dict stats = Rootfinder::get_stats(mem);
  auto m = static_cast<BrentMemory*>(mem);
  stats["iter_count"] = m->iter;
  stats["return_status"] = m->return_status;
  return stats;
}

}  // namespace casadi
