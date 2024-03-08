#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP

#include "IDAKLUSolverOpenMP.hpp"
#include "idaklu_solver.hpp"

/**
 * @brief IDAKLUSolver Dense implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_Dense : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_Dense(Args&& ... args)
    : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_Dense(yy, J, sunctx);
    Initialize();
  }
};

/**
 * @brief IDAKLUSolver KLU implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_KLU : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_KLU(Args&& ... args)
    : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_KLU(yy, J, sunctx);
    Initialize();
  }
};

/**
 * @brief IDAKLUSolver Banded implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_Band : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_Band(Args&& ... args)
    : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    LS = SUNLinSol_Band(yy, J, sunctx);
    Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPBCGS implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPBCGS : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPBCGS(Args&& ... args)
    : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
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
 * @brief IDAKLUSolver SPFGMR implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPFGMR : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPFGMR(Args&& ... args)
    : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
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
 * @brief IDAKLUSolver SPGMR implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPGMR : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPGMR(Args&& ... args)
    : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
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
 * @brief IDAKLUSolver SPTFQMR implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPTFQMR : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPTFQMR(Args&& ... args)
    : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
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
