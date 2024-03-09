#include "CasadiFunctions.hpp"

CasadiFunction::CasadiFunction(const casadi::Function &f) : Expression(f)
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
