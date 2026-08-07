#ifndef PYBAMM_BRENT_HPP
#define PYBAMM_BRENT_HPP

#include "casadi/core/rootfinder_impl.hpp"

// In-tree CasADi plugins get this from a generated export header; out of tree it is
// just default visibility, which is what CASADI_EXPORT expands to anyway.
#define CASADI_ROOTFINDER_BRENT_EXPORT __attribute__((visibility("default")))

namespace casadi {

struct CASADI_ROOTFINDER_BRENT_EXPORT BrentMemory : public RootfinderMemory {
  casadi_int iter;
  const char* return_status;
};

/**
 * @brief Brent's method, registered as the CasADi rootfinder plugin "brent".
 *
 * Solves a scalar g(x, p) = 0 on a bracket. Brent needs only a sign change over that
 * bracket, so it converges on residuals where a Newton iteration stalls or leaves the
 * domain, and the iterate is confined to the bracket by construction.
 *
 * The bracket is taken from the ``lo``/``hi`` options, or, when ``lo_index``/``hi_index``
 * are given, read at solve time from those inputs of the oracle -- so a bracket can be a
 * live value in the surrounding graph rather than a constant.
 *
 * Derivatives come from :class:`Rootfinder`, which applies the implicit function theorem;
 * nothing here is differentiated.
 */
class CASADI_ROOTFINDER_BRENT_EXPORT Brent : public Rootfinder {
public:
  explicit Brent(const std::string& name, const Function& f) : Rootfinder(name, f) {}
  ~Brent() override { clear_mem(); }

  const char* plugin_name() const override { return "brent"; }
  std::string class_name() const override { return "Brent"; }

  static Rootfinder* creator(const std::string& name, const Function& f) {
    return new Brent(name, f);
  }

  static const Options options_;
  const Options& get_options() const override { return options_; }
  static const std::string meta_doc;

  void init(const Dict& opts) override;
  int solve(void* mem) const override;

  void* alloc_mem() const override { return new BrentMemory(); }
  int init_mem(void* mem) const override;
  void free_mem(void* mem) const override { delete static_cast<BrentMemory*>(mem); }
  Dict get_stats(void* mem) const override;

  // Emit the iteration as C so an expression containing a Brent node survives
  // Function::generate() and JIT, which is how PyBaMM AOT-compiles its functions.
  bool has_codegen() const override { return true; }
  void codegen_declarations(CodeGenerator& g) const override;
  void codegen_body(CodeGenerator& g) const override;

  // Without these the deserialised instance silently loses its bracket and then fails
  // to solve, so they are required rather than optional.
  void serialize_body(SerializingStream& s) const override;
  static ProtoFunction* deserialize(DeserializingStream& s) { return new Brent(s); }

protected:
  explicit Brent(DeserializingStream& s);

  /// What ``residual`` needs to reach the oracle, passed through casadi_brent's void*.
  struct Context {
    const Brent* solver;
    BrentMemory* mem;
  };

  /// casadi_brent's residual callback for the interpreted path.
  static int residual(void* user_data, double x, double* fx);

  double lo_{0}, hi_{1}, abstol_{1e-14};
  casadi_int max_iter_{100};
  casadi_int lo_index_{-1}, hi_index_{-1};
};

extern "C" int CASADI_ROOTFINDER_BRENT_EXPORT
casadi_register_rootfinder_brent(Rootfinder::Plugin* plugin);

extern "C" void CASADI_ROOTFINDER_BRENT_EXPORT casadi_load_rootfinder_brent();

}  // namespace casadi

#endif  // PYBAMM_BRENT_HPP
