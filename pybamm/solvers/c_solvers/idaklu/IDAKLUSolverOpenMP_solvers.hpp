#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP

#include "IDAKLUSolverOpenMP.hpp"

/**
 * @brief IDAKLUSolver Dense implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_Dense : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_Dense(Args&& ... args) : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    IDAKLUSolverOpenMP::LS = SUNLinSol_Dense(IDAKLUSolverOpenMP::yy, IDAKLUSolverOpenMP::J, IDAKLUSolverOpenMP::sunctx);
    IDAKLUSolverOpenMP::Initialize();
  }
};

/**
 * @brief IDAKLUSolver KLU implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_KLU : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_KLU(Args&& ... args) : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    IDAKLUSolverOpenMP::LS = SUNLinSol_KLU(IDAKLUSolverOpenMP::yy, IDAKLUSolverOpenMP::J, IDAKLUSolverOpenMP::sunctx);
    IDAKLUSolverOpenMP::Initialize();
  }
};

/**
 * @brief IDAKLUSolver Banded implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_Band : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_Band(Args&& ... args) : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    IDAKLUSolverOpenMP::LS = SUNLinSol_Band(IDAKLUSolverOpenMP::yy, IDAKLUSolverOpenMP::J, IDAKLUSolverOpenMP::sunctx);
    IDAKLUSolverOpenMP::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPBCGS implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPBCGS : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPBCGS(Args&& ... args) : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    IDAKLUSolverOpenMP::LS = SUNLinSol_SPBCGS(
      IDAKLUSolverOpenMP::yy,
      IDAKLUSolverOpenMP::precon_type,
      IDAKLUSolverOpenMP::options.linsol_max_iterations,
      IDAKLUSolverOpenMP::sunctx
    );
    IDAKLUSolverOpenMP::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPFGMR implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPFGMR : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPFGMR(Args&& ... args) : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    IDAKLUSolverOpenMP::LS = SUNLinSol_SPFGMR(
      IDAKLUSolverOpenMP::yy,
      IDAKLUSolverOpenMP::precon_type,
      IDAKLUSolverOpenMP::options.linsol_max_iterations,
      IDAKLUSolverOpenMP::sunctx
    );
    IDAKLUSolverOpenMP::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPGMR implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPGMR : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPGMR(Args&& ... args) : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    IDAKLUSolverOpenMP::LS = SUNLinSol_SPGMR(
      IDAKLUSolverOpenMP::yy,
      IDAKLUSolverOpenMP::precon_type,
      IDAKLUSolverOpenMP::options.linsol_max_iterations,
      IDAKLUSolverOpenMP::sunctx
    );
    IDAKLUSolverOpenMP::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPTFQMR implementation with OpenMP class
 */
class IDAKLUSolverOpenMP_SPTFQMR : public IDAKLUSolverOpenMP {
public:
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPTFQMR(Args&& ... args) : IDAKLUSolverOpenMP(std::forward<Args>(args) ...)
  {
    IDAKLUSolverOpenMP::LS = SUNLinSol_SPTFQMR(
      IDAKLUSolverOpenMP::yy,
      IDAKLUSolverOpenMP::precon_type,
      IDAKLUSolverOpenMP::options.linsol_max_iterations,
      IDAKLUSolverOpenMP::sunctx
    );
    IDAKLUSolverOpenMP::Initialize();
  }
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
