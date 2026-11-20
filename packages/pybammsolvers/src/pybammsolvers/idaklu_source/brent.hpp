//
// A CasADi rootfinder plugin built out of tree, which takes four things from
// CMakeLists.txt that an in-tree CasADi plugin would get for free:
//
//  1. `rootfinder_impl.hpp` and four siblings. CasADi does not install these
//     (INSTALL_INTERNAL_HEADERS is off by default) and they are LGPL, so CMake stages
//     them from the pinned sdist at configure time instead of vendoring them here.
//     Use -DCASADI_SOURCE_DIR=<tree> to build offline.
//  2. A version check. These headers are not ABI-stable, so CMake reads the linked
//     CasADi's version from its own installed casadi/config.h -- the one source that
//     exists on every discovery path -- and refuses to build on a mismatch.
//  3. The flags CasADi itself was compiled with, also from config.h.
//     CASADI_WITH_THREADSAFE_SYMBOLICS adds a static mutex to Rootfinder, so guessing
//     it would be an ABI break rather than a warning.
//  4. `brent_impl_str`, the iteration stringified from brent_impl.hpp, so codegen can
//     emit the same text this file compiles.
//
// The export macro below is the fifth: in tree it comes from a generated header.
//
// The members below are all read from brent.cpp, but cppcheck also analyses this
// header on its own, where nothing uses them, so it calls every one of them unused.
// cppcheck-suppress-file unusedStructMember
#ifndef PYBAMM_BRENT_HPP
#define PYBAMM_BRENT_HPP

#include "casadi/core/rootfinder_impl.hpp"
#include <vector>

// whatever the toolchain spells "visible in this shared object"
#if defined(_WIN32) || defined(__CYGWIN__)
#define CASADI_ROOTFINDER_BRENT_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define CASADI_ROOTFINDER_BRENT_EXPORT __attribute__((visibility("default")))
#else
#define CASADI_ROOTFINDER_BRENT_EXPORT
#endif

namespace casadi {

struct CASADI_ROOTFINDER_BRENT_EXPORT BrentMemory : public RootfinderMemory {
  casadi_int iter;
  const char* return_status;
  // Last solve, keyed on every input but the guess, so a Brent nested inside another
  // is not re-solved on every iteration of the enclosing one.
  std::vector<double> cache_key;
  // scratch for the key of the solve in flight; per-memory, so two threads
  // evaluating one Function do not share it
  std::vector<double> key;
  double cache_root = 0;
  bool cache_valid = false;
  casadi_int cache_hits = 0;
};

/**
 * @brief Brent's method, registered as the CasADi rootfinder plugin "brent".
 *
 * Solves a scalar g(x, p) = 0 on a bracket. Brent needs only a sign change over that
 * bracket, so it converges on residuals where a Newton iteration stalls or leaves the
 * domain, and the iterate is confined to the bracket by construction.
 *
 * The oracle must be ``g(x, lo, hi, ...)``: the bracket is read from inputs 1 and 2 at
 * solve time, so it can be a live value in the surrounding graph rather than a constant.
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
  std::string cache_name(CodeGenerator& g) const;
  casadi_int cache_size() const;

  // PyBaMM round-trips its functions through serialize(), so the tolerances have to
  // survive it or a deserialised Brent silently reverts to the defaults.
  void serialize_body(SerializingStream& s) const override;
  static ProtoFunction* deserialize(DeserializingStream& s) { return new Brent(s); }

protected:
  explicit Brent(DeserializingStream& s);

  /// What the residual needs to reach the oracle, passed through casadi_brent's void*.
  struct Context {
    const Brent* solver;
    BrentMemory* mem;
  };

  /// casadi_brent's residual callback for the interpreted path.
  static int residual(void* user_data, double x, double* fx);

  double abstol_{1e-14};
  casadi_int max_iter_{100};
};

extern "C" int CASADI_ROOTFINDER_BRENT_EXPORT
casadi_register_rootfinder_brent(Rootfinder::Plugin* plugin);

extern "C" void CASADI_ROOTFINDER_BRENT_EXPORT casadi_load_rootfinder_brent();

}  // namespace casadi

#endif  // PYBAMM_BRENT_HPP
