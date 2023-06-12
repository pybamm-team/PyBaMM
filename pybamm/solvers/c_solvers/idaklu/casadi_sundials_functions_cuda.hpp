#ifndef PYBAMM_IDAKLU_CASADI_SUNDIALS_FUNCTIONS_CUDA_HPP
#define PYBAMM_IDAKLU_CASADI_SUNDIALS_FUNCTIONS_CUDA_HPP

#include "common.hpp"

int residual_casadi_cuda(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
                    void *user_data);

int jtimes_casadi_cuda(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr,
                  N_Vector v, N_Vector Jv, realtype cj, void *user_data,
                  N_Vector tmp1, N_Vector tmp2);

int events_casadi_cuda(realtype t, N_Vector yy, N_Vector yp, realtype *events_ptr,
                  void *user_data);

int sensitivities_casadi_cuda(int Ns, realtype t, N_Vector yy, N_Vector yp,
                         N_Vector resval, N_Vector *yS, N_Vector *ypS,
                         N_Vector *resvalS, void *user_data, N_Vector tmp1,
                         N_Vector tmp2, N_Vector tmp3);

int jacobian_casadi_cuda(realtype tt, realtype cj, N_Vector yy, N_Vector yp,
                    N_Vector resvec, SUNMatrix JJ, void *user_data,
                    N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);

int residual_casadi_approx_cuda(sunindextype Nlocal, realtype tt, N_Vector yy,
                           N_Vector yp, N_Vector gval, void *user_data);
#endif // PYBAMM_IDAKLU_CASADI_SUNDIALS_FUNCTIONS_CUDA_HPP
