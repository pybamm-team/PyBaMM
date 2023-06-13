#ifndef PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP

#include "common.hpp"
#include "options.hpp"
#include "solution.hpp"
#include <casadi/casadi.hpp>

template<typename T1, typename T2>
void csc_csr(realtype f[], T1 c[], T1 r[], realtype nf[], T2 nc[], T2 nr[], int N, int cols) {

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

using Function = casadi::Function;

class CasadiFunction
{
public:
  explicit CasadiFunction(const Function &f);

public:
  std::vector<const double *> m_arg;
  std::vector<double *> m_res;
  void operator()();

private:
  const Function &m_func;
  std::vector<casadi_int> m_iw;
  std::vector<double> m_w;
};

class CasadiFunctions
{
public:
  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  int number_of_nnz;
  int jac_bandwidth_lower;
  int jac_bandwidth_upper;
  CasadiFunction rhs_alg;
  CasadiFunction sens;
  CasadiFunction jac_times_cjmass;
  std::vector<int64_t> jac_times_cjmass_rowvals;
  std::vector<int64_t> jac_times_cjmass_colptrs;
  std::vector<realtype> inputs;
  CasadiFunction jac_action;
  CasadiFunction mass_action;
  CasadiFunction events;
  Options options;

  CasadiFunctions(const Function &rhs_alg, const Function &jac_times_cjmass,
                  const int jac_times_cjmass_nnz,
                  const int jac_bandwidth_lower, const int jac_bandwidth_upper,
                  const np_array_int &jac_times_cjmass_rowvals,
                  const np_array_int &jac_times_cjmass_colptrs,
                  const int inputs_length, const Function &jac_action,
                  const Function &mass_action, const Function &sens,
                  const Function &events, const int n_s, int n_e,
                  const int n_p, const Options& options);

  realtype *get_tmp_state_vector();
  realtype *get_tmp_sparse_jacobian_data();

private:
  std::vector<realtype> tmp_state_vector;
  std::vector<realtype> tmp_sparse_jacobian_data;
};

#endif // PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
