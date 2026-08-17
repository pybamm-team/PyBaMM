#include "brent.hpp"

#include <math.h>

#include <algorithm>

// The iteration itself, shared verbatim with the generated C (see codegen_declarations).
#include "brent_impl.hpp"

// CMake stringifies brent_impl.hpp into brent_impl_str, so the emitted C and the
// compiled iteration are the same text by construction.
#include "brent_impl_source.hpp"

namespace casadi {

// The oracle is g(x, lo, hi, ...), so the bracket always arrives as these inputs.
constexpr casadi_int BRACKET_LO = 1;
constexpr casadi_int BRACKET_HI = 2;

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
  {{"abstol", {OT_DOUBLE, "Absolute tolerance on the unknown"}},
   {"max_iter", {OT_INT, "Maximum number of iterations"}}}};

void Brent::init(const Dict& opts) {
  Rootfinder::init(opts);

  for (auto&& op : opts) {
    if (op.first == "abstol") {
      abstol_ = op.second;
    } else if (op.first == "max_iter") {
      max_iter_ = op.second;
    }
  }

  casadi_assert(n_ == 1, "Brent solves a scalar residual, got n=" + str(n_));
  casadi_assert(n_in_ >= 3,
                "Brent reads its bracket from inputs 1 and 2, so the oracle must be "
                "g(x, lo, hi, ...); got " + str(n_in_) + " input(s)");
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

  const double a = m->iarg[BRACKET_LO][0];
  const double b = m->iarg[BRACKET_HI][0];

  // PROTOTYPE cache. The root depends on the bracket and the parameters, never on
  // the initial guess, so the guess is left out of the key.
  key_.clear();
  for (casadi_int i = 0; i < n_in_; ++i) {
    if (i == iin_ || !m->iarg[i]) continue;
    key_.insert(key_.end(), m->iarg[i], m->iarg[i] + nnz_in(i));
  }
  if (m->cache_valid && m->cache_key == key_) {
    ++m->cache_hits;
    if (m->ires[iout_]) m->ires[iout_][0] = m->cache_root;
    m->return_status = "success (cached)";
    m->success = true;
    return 0;
  }

  double root = 0;
  const int flag = casadi_brent<double>(&Brent::residual, &ctx, a, b, abstol_, max_iter_,
                                        &root, &m->iter);
  if (flag) {
    m->return_status =
      flag == 2 ? "no sign change over the bracket" :
      flag == 3 ? "iteration limit reached without converging" : "residual failed";
    m->unified_return_status = SOLVER_RET_UNKNOWN;
    m->success = false;
    return 0;
  }

  m->cache_key = key_;
  m->cache_root = root;
  m->cache_valid = true;

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
    // The cache below is per instance and per thread. Generated code is expected to
    // be reentrant, so plain statics will not do.
    << "#if defined(__cplusplus) && __cplusplus >= 201103L\n"
    << "#define CASADI_BRENT_TLS thread_local\n"
    << "#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L"
       " && !defined(__STDC_NO_THREADS__)\n"
    << "#define CASADI_BRENT_TLS _Thread_local\n"
    << "#elif defined(__GNUC__) || defined(__clang__)\n"
    << "#define CASADI_BRENT_TLS __thread\n"
    << "#else\n"
    << "#define CASADI_BRENT_TLS\n"
    << "#endif\n"
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

  // Last solve, keyed on every input but the guess -- the same cache the interpreted
  // path keeps in BrentMemory. Without it a Brent nested inside another one re-solves
  // on every iteration of the enclosing solve.
  const std::string c = cache_name(g);
  g << "static CASADI_BRENT_TLS casadi_real " << c << "_key[" << cache_size() << "];\n"
    << "static CASADI_BRENT_TLS casadi_real " << c << "_root;\n"
    << "static CASADI_BRENT_TLS int " << c << "_valid = 0;\n\n";
}

std::string Brent::cache_name(CodeGenerator& g) const {
  return g.shorthand("brent_cache_" + codegen_name(g, false));
}

casadi_int Brent::cache_size() const {
  casadi_int n = 0;
  for (casadi_int i = 0; i < n_in_; ++i) if (i != iin_) n += nnz_in(i);
  return n;
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

  const std::string lo = g.arg(BRACKET_LO) + "[0]";
  const std::string hi = g.arg(BRACKET_HI) + "[0]";
  const std::string c = cache_name(g);

  std::vector<std::string> key;
  for (casadi_int i = 0; i < n_in_; ++i) {
    if (i == iin_) continue;
    for (casadi_int e = 0; e < nnz_in(i); ++e) {
      const std::string a = g.arg(i);
      key.push_back("(" + a + " ? " + a + "[" + str(e) + "] : 0)");
    }
  }

  g.local("brent_hit", "int");
  g << "brent_hit = " << c << "_valid;\n";
  for (casadi_int j = 0; j < static_cast<casadi_int>(key.size()); ++j) {
    g << "if (brent_hit && " << c << "_key[" << j << "] != " << key[j]
      << ") brent_hit = 0;\n";
  }
  g << "if (!brent_hit) {\n"
    << "  brent_flag = casadi_brent("
    << g.shorthand("brent_res_" + codegen_name(g, false))
    << ", &brent_data, " << lo << ", " << hi << ", " << g.constant(abstol_) << ", "
    << max_iter_ << ", &brent_root, &brent_iter);\n"
    << "  if (brent_flag) return 1;\n";
  for (casadi_int j = 0; j < static_cast<casadi_int>(key.size()); ++j) {
    g << "  " << c << "_key[" << j << "] = " << key[j] << ";\n";
  }
  g << "  " << c << "_root = brent_root;\n"
    << "  " << c << "_valid = 1;\n"
    << "}\n"
    << "if (" << g.res(iout_) << ") " << g.res(iout_) << "[0] = " << c << "_root;\n";
}

void Brent::serialize_body(SerializingStream& s) const {
  Rootfinder::serialize_body(s);
  s.version("Brent", 1);
  s.pack("Brent::abstol", abstol_);
  s.pack("Brent::max_iter", max_iter_);
}

Brent::Brent(DeserializingStream& s) : Rootfinder(s) {
  s.version("Brent", 1);
  s.unpack("Brent::abstol", abstol_);
  s.unpack("Brent::max_iter", max_iter_);
}

Dict Brent::get_stats(void* mem) const {
  Dict stats = Rootfinder::get_stats(mem);
  auto m = static_cast<BrentMemory*>(mem);
  stats["iter_count"] = m->iter;
  stats["return_status"] = m->return_status;
  return stats;
}

}  // namespace casadi
