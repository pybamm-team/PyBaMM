#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP

#include "IDAKLUSolverOpenMP.hpp"

/**
 * @brief IDAKLUSolver Dense implementation with OpenMP class
 */
template <class T>
class IDAKLUSolverOpenMP_Dense : public IDAKLUSolverOpenMP<T> {
public:
  using Base = IDAKLUSolverOpenMP<T>;
  template<typename ... Args>
  IDAKLUSolverOpenMP_Dense(Args&& ... args) : Base(std::forward<Args>(args) ...)
  {
    Base::LS = SUNLinSol_Dense(Base::yy, Base::J, Base::sunctx);
    Base::Initialize();
  }
};

/**
 * @brief IDAKLUSolver KLU implementation with OpenMP class
 */
template <class T>
class IDAKLUSolverOpenMP_KLU : public IDAKLUSolverOpenMP<T> {
public:
  using Base = IDAKLUSolverOpenMP<T>;
  template<typename ... Args>
  IDAKLUSolverOpenMP_KLU(Args&& ... args) : Base(std::forward<Args>(args) ...)
  {
    Base::LS = SUNLinSol_KLU(Base::yy, Base::J, Base::sunctx);
    Base::Initialize();
  }
};

/**
 * @brief IDAKLUSolver Banded implementation with OpenMP class
 */
template <class T>
class IDAKLUSolverOpenMP_Band : public IDAKLUSolverOpenMP<T> {
public:
  using Base = IDAKLUSolverOpenMP<T>;
  template<typename ... Args>
  IDAKLUSolverOpenMP_Band(Args&& ... args) : Base(std::forward<Args>(args) ...)
  {
    Base::LS = SUNLinSol_Band(Base::yy, Base::J, Base::sunctx);
    Base::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPBCGS implementation with OpenMP class
 */
template <class T>
class IDAKLUSolverOpenMP_SPBCGS : public IDAKLUSolverOpenMP<T> {
public:
  using Base = IDAKLUSolverOpenMP<T>;
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPBCGS(Args&& ... args) : Base(std::forward<Args>(args) ...)
  {
    Base::LS = SUNLinSol_SPBCGS(
      Base::yy,
      Base::precon_type,
      Base::setup_opts.linsol_max_iterations,
      Base::sunctx
    );
    Base::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPFGMR implementation with OpenMP class
 */
template <class T>
class IDAKLUSolverOpenMP_SPFGMR : public IDAKLUSolverOpenMP<T> {
public:
  using Base = IDAKLUSolverOpenMP<T>;
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPFGMR(Args&& ... args) : Base(std::forward<Args>(args) ...)
  {
    Base::LS = SUNLinSol_SPFGMR(
      Base::yy,
      Base::precon_type,
      Base::setup_opts.linsol_max_iterations,
      Base::sunctx
    );
    Base::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPGMR implementation with OpenMP class
 */
template <class T>
class IDAKLUSolverOpenMP_SPGMR : public IDAKLUSolverOpenMP<T> {
public:
  using Base = IDAKLUSolverOpenMP<T>;
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPGMR(Args&& ... args) : Base(std::forward<Args>(args) ...)
  {
    Base::LS = SUNLinSol_SPGMR(
      Base::yy,
      Base::precon_type,
      Base::setup_opts.linsol_max_iterations,
      Base::sunctx
    );
    Base::Initialize();
  }
};

/**
 * @brief IDAKLUSolver SPTFQMR implementation with OpenMP class
 */
template <class T>
class IDAKLUSolverOpenMP_SPTFQMR : public IDAKLUSolverOpenMP<T> {
public:
  using Base = IDAKLUSolverOpenMP<T>;
  template<typename ... Args>
  IDAKLUSolverOpenMP_SPTFQMR(Args&& ... args) : Base(std::forward<Args>(args) ...)
  {
    Base::LS = SUNLinSol_SPTFQMR(
      Base::yy,
      Base::precon_type,
      Base::setup_opts.linsol_max_iterations,
      Base::sunctx
    );
    Base::Initialize();
  }
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
