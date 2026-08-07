#include "brent.hpp"

#include <math.h>

#include <algorithm>

// The iteration itself, shared verbatim with the generated C (see codegen_declarations).
#include "brent_impl.hpp"

// CMake stringifies brent_impl.hpp into brent_impl_str, so the emitted C and the
// compiled iteration are the same text by construction.
#include "brent_impl_source.hpp"

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
}

int Brent::init_mem(void* mem) const {
  if (Rootfinder::init_mem(mem)) return 1;
  auto m = static_cast<BrentMemory*>(mem);
  m->iter = 0;
  m->return_status = "unset";
  return 0;
}

int Brent::residual(void* user_data, double x, double* fx) {
  auto ctx = static_cast<Context*>(user_data);
  const Brent* self = ctx->solver;
  BrentMemory* m = ctx->mem;
  std::copy_n(m->iarg, self->n_in_, m->arg);
  m->arg[self->iin_] = &x;
  std::copy_n(m->ires, self->n_out_, m->res);
  m->res[self->iout_] = fx;
  return self->calc_function(m, "g");
}

int Brent::solve(void* mem) const {
  auto m = static_cast<BrentMemory*>(mem);
  Context ctx{this, m};

  const double a = lo_index_ >= 0 ? m->iarg[lo_index_][0] : lo_;
  const double b = hi_index_ >= 0 ? m->iarg[hi_index_][0] : hi_;

  double root = 0;
  const int flag = casadi_brent<double>(&Brent::residual, &ctx, a, b, abstol_, max_iter_,
                                        &root, &m->iter);
  if (flag) {
    m->return_status = flag == 2 ? "no sign change over the bracket" : "residual failed";
    m->unified_return_status = SOLVER_RET_UNKNOWN;
    m->success = false;
    return 0;
  }

  if (m->ires[iout_]) m->ires[iout_][0] = root;
  m->return_status = "success";
  m->success = true;
  return 0;
}

void Brent::codegen_declarations(CodeGenerator& g) const {
  // Adding the oracle first keeps its definition out of the middle of the wrapper
  // emitted below, which is written straight to the buffer.
  g.add_dependency(get_function("g"));

  // The iteration and its user-data struct do not depend on this instance, so the guard
  // collapses several Brent nodes in one file down to one definition. add_shorthand is
  // off so the name is not CASADI_PREFIX-renamed, which the guard would then defeat.
  g << "#ifndef CASADI_BRENT_IMPL\n"
    << "#define CASADI_BRENT_IMPL\n"
    << "struct casadi_brent_data {\n"
    << "  const casadi_real** arg;\n"
    << "  casadi_real** res;\n"
    << "  casadi_int* iw;\n"
    << "  casadi_real* w;\n"
    << "};\n"
    << g.sanitize_source(brent_impl_str, {"casadi_real"}, false)
    << "#endif\n\n";

  // The residual callback is per instance: it hard-codes this oracle and this
  // implicit input/output pair.
  g << "static int " << g.shorthand("brent_res_" + codegen_name(g, false))
    << "(void* user_data, casadi_real x, casadi_real* fx) {\n"
    << "  struct casadi_brent_data* d = (struct casadi_brent_data*) user_data;\n"
    << "  const casadi_real** arg1 = d->arg + " << n_in_ << ";\n"
    << "  casadi_real** res1 = d->res + " << n_out_ << ";\n";
  for (casadi_int i = 0; i < n_in_; ++i) {
    g << "  arg1[" << i << "] = " << (i == iin_ ? "&x" : "d->" + g.arg(i)) << ";\n";
  }
  for (casadi_int i = 0; i < n_out_; ++i) {
    g << "  res1[" << i << "] = " << (i == iout_ ? "fx" : "d->" + g.res(i)) << ";\n";
  }
  g << "  return " << g(get_function("g"), "arg1", "res1", "d->iw", "d->w") << ";\n"
    << "}\n\n";
}

void Brent::codegen_body(CodeGenerator& g) const {
  g.local("brent_data", "struct casadi_brent_data");
  g.local("brent_iter", "casadi_int");
  g.local("brent_root", "casadi_real");
  g.local("brent_flag", "int");

  // sz_w_per_ is zero, so the oracle's scratch starts at w -- the same slice
  // calc_function hands the oracle on the interpreted path.
  g << "brent_data.arg = arg;\n"
    << "brent_data.res = res;\n"
    << "brent_data.iw = iw;\n"
    << "brent_data.w = w;\n";

  const std::string lo = lo_index_ >= 0 ? g.arg(lo_index_) + "[0]" : g.constant(lo_);
  const std::string hi = hi_index_ >= 0 ? g.arg(hi_index_) + "[0]" : g.constant(hi_);
  g << "brent_flag = casadi_brent(" << g.shorthand("brent_res_" + codegen_name(g, false))
    << ", &brent_data, " << lo << ", " << hi << ", " << g.constant(abstol_) << ", "
    << max_iter_ << ", &brent_root, &brent_iter);\n"
    << "if (brent_flag) return 1;\n"
    << "if (" << g.res(iout_) << ") " << g.res(iout_) << "[0] = brent_root;\n";
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
