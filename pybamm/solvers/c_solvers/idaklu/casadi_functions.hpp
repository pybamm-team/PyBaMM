#ifndef PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP

#include "solution.hpp"
#include "common.hpp"
#include <casadi/casadi.hpp>

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
  CasadiFunction rhs_alg;
  CasadiFunction sens;
  CasadiFunction jac_times_cjmass;
  const np_array_int &jac_times_cjmass_rowvals;
  const np_array_int &jac_times_cjmass_colptrs;
  const np_array_dense &inputs;
  CasadiFunction jac_action;
  CasadiFunction mass_action;
  CasadiFunction events;

  CasadiFunctions(const Function &rhs_alg, const Function &jac_times_cjmass,
                  const int jac_times_cjmass_nnz,
                  const np_array_int &jac_times_cjmass_rowvals,
                  const np_array_int &jac_times_cjmass_colptrs,
                  const np_array_dense &inputs, const Function &jac_action,
                  const Function &mass_action, const Function &sens,
                  const Function &events, const int n_s, int n_e,
                  const int n_p);

  realtype *get_tmp();

private:
  std::vector<realtype> tmp;
};

#endif // PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
