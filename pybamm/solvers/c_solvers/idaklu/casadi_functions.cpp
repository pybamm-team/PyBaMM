#include "casadi_functions.hpp"

void csc_csr(realtype f[], long c[], long r[], realtype nf[], int nc[], int nr[], int N, int cols) {

  /*for (int i=0; i<N; i++) {
    nf[i] = f[i];
    nr[i] = c[i];
  }
  for (int i=0; i<cols+1; i++)
    nc[i] = r[i];

  return;*/

  int nn[cols+1];
  int rr[N];
  for (int i=0; i<cols+1; i++)
    nc[i] = 0;
  
  for (int k = 0, i = 0; i < cols+1; i++) {
    for (int j = 0; j < r[i+1] - r[i]; j++) {
      if (k == N)  // r does not contain the final element
        break;
      rr[k++] = i;
    }
  }
  for (int i = 0; i < N; i++)
    nc[c[i]+1]++;
  for (int i = 1; i <= cols; i++)
    nc[i] += nc[i-1];
  for (int i = 0; i < cols+1; i++)
    nn[i] = nc[i];

  for (int i = 0; i < N; i++) {
    int x = nn[c[i]]++;
    nf[x] = f[i];
    nr[x] = rr[i];
  }
}

void csc_csr(realtype f[], int c[], int r[], realtype nf[], long nc[], long nr[], int N, int cols) {
  /*for (int i=0; i<N; i++) {
    nf[i] = f[i];
    nr[i] = c[i];
  }
  for (int i=0; i<cols+1; i++)
    nc[i] = r[i];

  return;*/

  int nn[cols+1];
  int rr[N];
  for (int i=0; i<cols+1; i++)
    nc[i] = 0;
  
  for (int k = 0, i = 0; i < cols+1; i++) {
    for (int j = 0; j < r[i+1] - r[i]; j++) {
      if (k == N)  // r does not contain the final element
        break;
      rr[k++] = i;
    }
  }
  for (int i = 0; i < N; i++)
    nc[c[i]+1]++;
  for (int i = 1; i <= cols; i++)
    nc[i] += nc[i-1];
  for (int i = 0; i < cols+1; i++)
    nn[i] = nc[i];

  for (int i = 0; i < N; i++) {
    int x = nn[c[i]]++;
    nf[x] = f[i];
    nr[x] = rr[i];
  }
}

CasadiFunction::CasadiFunction(const Function &f) : m_func(f)
{
    size_t sz_arg;
    size_t sz_res;
    size_t sz_iw;
    size_t sz_w;
    m_func.sz_work(sz_arg, sz_res, sz_iw, sz_w);
    // std::cout << "name = "<< m_func.name() << " arg = " << sz_arg << " res = "
    // << sz_res << " iw = " << sz_iw << " w = " << sz_w << std::endl; for (int i
    // = 0; i < sz_arg; i++) {
    //   std::cout << "Sparsity for input " << i << std::endl;
    //   const Sparsity& sparsity = m_func.sparsity_in(i);
    // }
    // for (int i = 0; i < sz_res; i++) {
    //   std::cout << "Sparsity for output " << i << std::endl;
    //   const Sparsity& sparsity = m_func.sparsity_out(i);
    // }
    m_arg.resize(sz_arg);
    m_res.resize(sz_res);
    m_iw.resize(sz_iw);
    m_w.resize(sz_w);
}

// only call this once m_arg and m_res have been set appropriatelly
void CasadiFunction::operator()()
{
    int mem = m_func.checkout();
    m_func(m_arg.data(), m_res.data(), m_iw.data(), m_w.data(), mem);
    m_func.release(mem);
}

CasadiFunctions::CasadiFunctions(
    const Function &rhs_alg, const Function &jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower, const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals_arg,
    const np_array_int &jac_times_cjmass_colptrs_arg,
    const int inputs_length, const Function &jac_action,
    const Function &mass_action, const Function &sens, const Function &events,
    const int n_s, int n_e, const int n_p, const Options& options)
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
