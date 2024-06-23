#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP

#include "CasadiSolverOpenMP.hpp"
#include "casadi_solver.hpp"

/**
 * @brief CasadiSolver Dense implementation with OpenMP class
 */
class CasadiSolverOpenMP_Dense : public CasadiSolverOpenMP {
public:
  template<typename ... Args>
  CasadiSolverOpenMP_Dense(Args&& ... args)
    : CasadiSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_Dense(yy, J, sunctx);
    Initialize();
  }
};

/**
 * @brief CasadiSolver KLU implementation with OpenMP class
 */
class CasadiSolverOpenMP_KLU : public CasadiSolverOpenMP {
public:
  template<typename ... Args>
  CasadiSolverOpenMP_KLU(Args&& ... args)
    : CasadiSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_KLU(yy, J, sunctx);
    Initialize();
  }
};

/**
 * @brief CasadiSolver Banded implementation with OpenMP class
 */
class CasadiSolverOpenMP_Band : public CasadiSolverOpenMP {
public:
  template<typename ... Args>
  CasadiSolverOpenMP_Band(Args&& ... args)
    : CasadiSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_Band(yy, J, sunctx);
    Initialize();
  }
};

/**
 * @brief CasadiSolver SPBCGS implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPBCGS : public CasadiSolverOpenMP {
public:
  template<typename ... Args>
  CasadiSolverOpenMP_SPBCGS(Args&& ... args)
    : CasadiSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_SPBCGS(
      yy,
      precon_type,
      options.linsol_max_iterations,
      sunctx
    );
    Initialize();
  }
};

/**
 * @brief CasadiSolver SPFGMR implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPFGMR : public CasadiSolverOpenMP {
public:
  template<typename ... Args>
  CasadiSolverOpenMP_SPFGMR(Args&& ... args)
    : CasadiSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_SPFGMR(
      yy,
      precon_type,
      options.linsol_max_iterations,
      sunctx
    );
    Initialize();
  }
};

/**
 * @brief CasadiSolver SPGMR implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPGMR : public CasadiSolverOpenMP {
public:
  template<typename ... Args>
  CasadiSolverOpenMP_SPGMR(Args&& ... args)
    : CasadiSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_SPGMR(
      yy,
      precon_type,
      options.linsol_max_iterations,
      sunctx
    );
    Initialize();
  }
};

/**
 * @brief CasadiSolver SPTFQMR implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPTFQMR : public CasadiSolverOpenMP {
public:
  template<typename ... Args>
  CasadiSolverOpenMP_SPTFQMR(Args&& ... args)
    : CasadiSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_SPTFQMR(
      yy,
      precon_type,
      options.linsol_max_iterations,
      sunctx
    );
    Initialize();
  }
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
