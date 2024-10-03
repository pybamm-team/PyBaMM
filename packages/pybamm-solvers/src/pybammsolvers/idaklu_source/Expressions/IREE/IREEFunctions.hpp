#ifndef PYBAMM_IDAKLU_IREE_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_IREE_FUNCTIONS_HPP

#include "../../Options.hpp"
#include "../Expressions.hpp"
#include <memory>
#include "iree_jit.hpp"
#include "IREEFunction.hpp"

/**
 * @brief Class for handling iree functions
 */
class IREEFunctions : public ExpressionSet<IREEFunction>
{
public:
  std::unique_ptr<IREECompiler> iree_compiler;

  typedef IREEFunction::BaseFunctionType BaseFunctionType;  // expose typedef in class

  int iree_init_status;

  int iree_init(const std::string& device_uri, const std::string& target_backends) {
    // Initialise IREE
    DEBUG("IREEFunctions: Initialising IREECompiler");
    iree_compiler = std::make_unique<IREECompiler>(device_uri.c_str());

    int iree_argc = 2;
    std::string target_backends_str = "--iree-hal-target-backends=" + target_backends;
    const char* iree_argv[2] = {"iree", target_backends_str.c_str()};
    iree_compiler->init(iree_argc, iree_argv);
    DEBUG("IREEFunctions: Initialised IREECompiler");
    return 0;
  }

  int iree_init() {
    return iree_init("local-sync", "llvm-cpu");
  }


  /**
   * @brief Create a new IREEFunctions object
   */
  IREEFunctions(
    const BaseFunctionType &rhs_alg,
    const BaseFunctionType &jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower,
    const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals_arg,
    const np_array_int &jac_times_cjmass_colptrs_arg,
    const int inputs_length,
    const BaseFunctionType &jac_action,
    const BaseFunctionType &mass_action,
    const BaseFunctionType &sens,
    const BaseFunctionType &events,
    const int n_s,
    const int n_e,
    const int n_p,
    const std::vector<BaseFunctionType*>& var_fcns,
    const std::vector<BaseFunctionType*>& dvar_dy_fcns,
    const std::vector<BaseFunctionType*>& dvar_dp_fcns,
    const SetupOptions& setup_opts
  ) :
    iree_init_status(iree_init()),
    rhs_alg_iree(rhs_alg),
    jac_times_cjmass_iree(jac_times_cjmass),
    jac_action_iree(jac_action),
    mass_action_iree(mass_action),
    sens_iree(sens),
    events_iree(events),
    ExpressionSet<IREEFunction>(
      static_cast<Expression*>(&rhs_alg_iree),
      static_cast<Expression*>(&jac_times_cjmass_iree),
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      jac_times_cjmass_rowvals_arg,
      jac_times_cjmass_colptrs_arg,
      inputs_length,
      static_cast<Expression*>(&jac_action_iree),
      static_cast<Expression*>(&mass_action_iree),
      static_cast<Expression*>(&sens_iree),
      static_cast<Expression*>(&events_iree),
      n_s, n_e, n_p,
      setup_opts)
  {
    // convert BaseFunctionType list to IREEFunction list
    // NOTE: You must allocate ALL std::vector elements before taking references
    for (auto& var : var_fcns)
      var_fcns_iree.push_back(IREEFunction(*var));
    for (int k = 0; k < var_fcns_iree.size(); k++)
      ExpressionSet::var_fcns.push_back(&this->var_fcns_iree[k]);

    for (auto& var : dvar_dy_fcns)
      dvar_dy_fcns_iree.push_back(IREEFunction(*var));
    for (int k = 0; k < dvar_dy_fcns_iree.size(); k++)
      this->dvar_dy_fcns.push_back(&this->dvar_dy_fcns_iree[k]);

    for (auto& var : dvar_dp_fcns)
      dvar_dp_fcns_iree.push_back(IREEFunction(*var));
    for (int k = 0; k < dvar_dp_fcns_iree.size(); k++)
      this->dvar_dp_fcns.push_back(&this->dvar_dp_fcns_iree[k]);

    // copy across numpy array values
    const int n_row_vals = jac_times_cjmass_rowvals_arg.request().size;
    auto p_jac_times_cjmass_rowvals = jac_times_cjmass_rowvals_arg.unchecked<1>();
    jac_times_cjmass_rowvals.resize(n_row_vals);
    for (int i = 0; i < n_row_vals; i++) {
      jac_times_cjmass_rowvals[i] = p_jac_times_cjmass_rowvals[i];
    }

    const int n_col_ptrs = jac_times_cjmass_colptrs_arg.request().size;
    auto p_jac_times_cjmass_colptrs = jac_times_cjmass_colptrs_arg.unchecked<1>();
    jac_times_cjmass_colptrs.resize(n_col_ptrs);
    for (int i = 0; i < n_col_ptrs; i++) {
      jac_times_cjmass_colptrs[i] = p_jac_times_cjmass_colptrs[i];
    }

    inputs.resize(inputs_length);
  }

  IREEFunction rhs_alg_iree;
  IREEFunction jac_times_cjmass_iree;
  IREEFunction jac_action_iree;
  IREEFunction mass_action_iree;
  IREEFunction sens_iree;
  IREEFunction events_iree;

  std::vector<IREEFunction> var_fcns_iree;
  std::vector<IREEFunction> dvar_dy_fcns_iree;
  std::vector<IREEFunction> dvar_dp_fcns_iree;

  realtype* get_tmp_state_vector() override {
    return tmp_state_vector.data();
  }
  realtype* get_tmp_sparse_jacobian_data() override {
    return tmp_sparse_jacobian_data.data();
  }

  ~IREEFunctions() {
    // cleanup IREE
    iree_compiler->cleanup();
  }
};

#endif // PYBAMM_IDAKLU_IREE_FUNCTIONS_HPP
