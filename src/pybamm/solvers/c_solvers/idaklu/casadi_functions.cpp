#include "casadi_functions.hpp"

CasadiFunction::CasadiFunction(const Function &f) : m_func(f)
{
  size_t sz_arg;
  size_t sz_res;
  size_t sz_iw;
  size_t sz_w;
  m_func.sz_work(sz_arg, sz_res, sz_iw, sz_w);
  //int nnz = (sz_res>0) ? m_func.nnz_out() : 0;
  //std::cout << "name = "<< m_func.name() << " arg = " << sz_arg << " res = "
  //  << sz_res << " iw = " << sz_iw << " w = " << sz_w << " nnz = " << nnz <<
  //  std::endl;
  m_arg.resize(sz_arg, nullptr);
  m_res.resize(sz_res, nullptr);
  m_iw.resize(sz_iw, 0);
  m_w.resize(sz_w, 0);
}

// only call this once m_arg and m_res have been set appropriately
void CasadiFunction::operator()()
{
  int mem = m_func.checkout();
  m_func(m_arg.data(), m_res.data(), m_iw.data(), m_w.data(), mem);
  m_func.release(mem);
}

casadi_int CasadiFunction::nnz_out() {
  return m_func.nnz_out();
}

casadi::Sparsity CasadiFunction::sparsity_out(casadi_int ind) {
  return m_func.sparsity_out(ind);
}

void CasadiFunction::operator()(const std::vector<realtype*>& inputs,
                                const std::vector<realtype*>& results)
{
  // Set-up input arguments, provide result vector, then execute function
  // Example call: fcn({in1, in2, in3}, {out1})
  for(size_t k=0; k<inputs.size(); k++)
    m_arg[k] = inputs[k];
  for(size_t k=0; k<results.size(); k++)
    m_res[k] = results[k];
  operator()();
}

CasadiFunctions::CasadiFunctions(
  const Function &rhs_alg, const Function &jac_times_cjmass,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower, const int jac_bandwidth_upper,
  const np_array_int &jac_times_cjmass_rowvals_arg,
  const np_array_int &jac_times_cjmass_colptrs_arg,
  const int inputs_length, const Function &jac_action,
  const Function &mass_action, const Function &sens, const Function &events,
  const int n_s, int n_e, const int n_p,
  const std::vector<Function*>& var_casadi_fcns,
  const std::vector<Function*>& dvar_dy_fcns,
  const std::vector<Function*>& dvar_dp_fcns,
  const Options& options)
  : number_of_states(n_s), number_of_events(n_e), number_of_parameters(n_p),
    number_of_nnz(jac_times_cjmass_nnz),
    jac_bandwidth_lower(jac_bandwidth_lower), jac_bandwidth_upper(jac_bandwidth_upper),
    rhs_alg(rhs_alg),
    jac_times_cjmass(jac_times_cjmass), jac_action(jac_action),
    mass_action(mass_action), sens(sens), events(events),
    tmp_state_vector(number_of_states),
    tmp_sparse_jacobian_data(jac_times_cjmass_nnz),
    options(options)
{
  // convert casadi::Function list to CasadiFunction list
  for (auto& var : var_casadi_fcns) {
    this->var_casadi_fcns.push_back(CasadiFunction(*var));
  }
  for (auto& var : dvar_dy_fcns) {
    this->dvar_dy_fcns.push_back(CasadiFunction(*var));
  }
  for (auto& var : dvar_dp_fcns) {
    this->dvar_dp_fcns.push_back(CasadiFunction(*var));
  }

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

realtype *CasadiFunctions::get_tmp_state_vector() {
  return tmp_state_vector.data();
}
realtype *CasadiFunctions::get_tmp_sparse_jacobian_data() {
  return tmp_sparse_jacobian_data.data();
}
