#ifndef PYBAMM_IDAKLU_SOLVEROPENMP_HPP
#define PYBAMM_IDAKLU_SOLVEROPENMP_HPP

#include "IDAKLUSolver.hpp"
#include "common.hpp"
#include <vector>
using std::vector;

#include "Options.hpp"
#include "Solution.hpp"
#include "sundials_legacy_wrapper.hpp"

/**
 * @brief Abstract solver class based on OpenMP vectors
 *
 * An abstract class that implements a solution based on OpenMP
 * vectors but needs to be provided with a suitable linear solver.
 *
 * This class broadly implements the following skeleton workflow:
 * (https://sundials.readthedocs.io/en/latest/ida/Usage/index.html)
 *    1. (N/A) Initialize parallel or multi-threaded environment
 *    2. Create the SUNDIALS context object
 *    3. Create the vector of initial values
 *    4. Create matrix object (if appropriate)
 *    5. Create linear solver object
 *    6. (N/A) Create nonlinear solver object
 *    7. Create IDA object
 *    8. Initialize IDA solver
 *    9. Specify integration tolerances
 *   10. Attach the linear solver
 *   11. Set linear solver optional inputs
 *   12. (N/A) Attach nonlinear solver module
 *   13. (N/A) Set nonlinear solver optional inputs
 *   14. Specify rootfinding problem (optional)
 *   15. Set optional inputs
 *   16. Correct initial values (optional)
 *   17. Advance solution in time
 *   18. Get optional outputs
 *   19. Destroy objects
 *   20. (N/A) Finalize MPI
 */
template <class ExprSet>
class IDAKLUSolverOpenMP : public IDAKLUSolver
{
  // NB: cppcheck-suppress unusedStructMember is used because codacy reports
  //     these members as unused even though they are important in child
  //     classes, but are passed by variadic arguments (and are therefore unnamed)
public:
  void *ida_mem = nullptr;
  np_array atol_np;
  np_array rhs_alg_id;
  int const number_of_states;  // cppcheck-suppress unusedStructMember
  int const number_of_parameters;  // cppcheck-suppress unusedStructMember
  int const number_of_events;  // cppcheck-suppress unusedStructMember
  int number_of_timesteps;
  int precon_type;  // cppcheck-suppress unusedStructMember
  N_Vector yy, yyp, y_cache, avtol;  // y, y', y cache vector, and absolute tolerance
  N_Vector *yyS;  // cppcheck-suppress unusedStructMember
  N_Vector *yypS;  // cppcheck-suppress unusedStructMember
  N_Vector id;              // rhs_alg_id
  realtype rtol;
  int const jac_times_cjmass_nnz;  // cppcheck-suppress unusedStructMember
  int const jac_bandwidth_lower;  // cppcheck-suppress unusedStructMember
  int const jac_bandwidth_upper;  // cppcheck-suppress unusedStructMember
  SUNMatrix J;
  SUNLinearSolver LS = nullptr;
  std::unique_ptr<ExprSet> functions;
  vector<realtype> res;
  vector<realtype> res_dvar_dy;
  vector<realtype> res_dvar_dp;
  bool const sensitivity;  // cppcheck-suppress unusedStructMember
  bool const save_outputs_only; // cppcheck-suppress unusedStructMember
  bool save_hermite;  // cppcheck-suppress unusedStructMember
  bool is_ODE;  // cppcheck-suppress unusedStructMember
  int length_of_return_vector;  // cppcheck-suppress unusedStructMember
  vector<realtype> t;  // cppcheck-suppress unusedStructMember
  vector<vector<realtype>> y;  // cppcheck-suppress unusedStructMember
  vector<vector<realtype>> yp;  // cppcheck-suppress unusedStructMember
  vector<vector<vector<realtype>>> yS;  // cppcheck-suppress unusedStructMember
  vector<vector<vector<realtype>>> ypS;  // cppcheck-suppress unusedStructMember
  SetupOptions const setup_opts;
  SolverOptions const solver_opts;

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext sunctx;
#endif

public:
  /**
   * @brief Constructor
   */
  IDAKLUSolverOpenMP(
    np_array atol_np,
    double rel_tol,
    np_array rhs_alg_id,
    int number_of_parameters,
    int number_of_events,
    int jac_times_cjmass_nnz,
    int jac_bandwidth_lower,
    int jac_bandwidth_upper,
    std::unique_ptr<ExprSet> functions,
    const SetupOptions &setup_opts,
    const SolverOptions &solver_opts
  );

  /**
   * @brief Destructor
   */
  virtual ~IDAKLUSolverOpenMP();

  /**
   * @brief The main solve method that solves for each variable and time step
   */
  SolutionData solve(
    const std::vector<realtype> &t_eval,
    const std::vector<realtype> &t_interp,
    const realtype *y0,
    const realtype *yp0,
    const realtype *inputs,
    bool save_adaptive_steps,
    bool save_interp_steps
  ) override;


  /**
   * @brief Concrete implementation of initialization method
   */
  void Initialize() override;

  /**
   * @brief Allocate memory for OpenMP vectors
   */
  void AllocateVectors();

  /**
   * @brief Allocate memory for matrices (noting appropriate matrix format/types)
   */
  void SetMatrix();

  /**
   * @brief Get the length of the return vector
   */
  int ReturnVectorLength();

  /**
   * @brief Initialize the storage for the solution
   */
  void InitializeStorage(int const N);

  /**
   * @brief Initialize the storage for Hermite interpolation
   */
  void InitializeHermiteStorage(int const N);

  /**
   * @brief Apply user-configurable IDA options
   */
  void SetSolverOptions();

  /**
   * @brief Check the return flag for errors
   */
  void CheckErrors(int const & flag);

  /**
   * @brief Print the solver statistics
   */
  void PrintStats();

  /**
   * @brief Set a consistent initialization for ODEs
   */
  void ReinitializeIntegrator(const realtype& t_val);

  /**
   * @brief Set a consistent initialization for the system of equations
   */
  void ConsistentInitialization(
    const realtype& t_val,
    const realtype& t_next,
    const int& icopt);

  /**
   * @brief Set a consistent initialization for DAEs
   */
  void ConsistentInitializationDAE(
    const realtype& t_val,
    const realtype& t_next,
    const int& icopt);

  /**
   * @brief Set a consistent initialization for ODEs
   */
  void ConsistentInitializationODE(const realtype& t_val);

  /**
   * @brief Extend the adaptive arrays by 1
   */
  void ExtendAdaptiveArrays();

  /**
   * @brief Extend the Hermite interpolation info by 1
   */
  void ExtendHermiteArrays();

  /**
   * @brief Set the step values
   */
  void SetStep(
    realtype &tval,
    realtype *y_val,
    realtype *yp_val,
    vector<realtype *> const &yS_val,
    vector<realtype *> const &ypS_val,
    int &i_save
  );

  /**
   * @brief Save the interpolated step values
   */
  void SetStepInterp(
    int &i_interp,
    realtype &t_interp_next,
    vector<realtype> const &t_interp,
    realtype &t_val,
    realtype &t_prev,
    realtype const &t_next,
    realtype *y_val,
    realtype *yp_val,
    vector<realtype *> const &yS_val,
    vector<realtype *> const &ypS_val,
    int &i_save
  );

  /**
   * @brief Save y and yS at the current time
   */
  void SetStepFull(
    realtype &t_val,
    realtype *y_val,
    vector<realtype *> const &yS_val,
    int &i_save
  );

  /**
   * @brief Save yS at the current time
   */
  void SetStepFullSensitivities(
    realtype &t_val,
    realtype *y_val,
    vector<realtype *> const &yS_val,
    int &i_save
  );

  /**
   * @brief Save the output function results at the requested time
   */
  void SetStepOutput(
    realtype &t_val,
    realtype *y_val,
    const vector<realtype*> &yS_val,
    int &i_save
  );

  /**
   * @brief Save the output function sensitivities at the requested time
   */
  void SetStepOutputSensitivities(
    realtype &t_val,
    realtype *y_val,
    const vector<realtype*> &yS_val,
    int &i_save
  );

  /**
   * @brief Save the output function results at the requested time
   */
  void SetStepHermite(
    realtype &t_val,
    realtype *yp_val,
    const vector<realtype*> &ypS_val,
    int &i_save
  );

  /**
   * @brief Save the output function sensitivities at the requested time
   */
  void SetStepHermiteSensitivities(
    realtype &t_val,
    realtype *yp_val,
    const vector<realtype*> &ypS_val,
    int &i_save
  );

};

#include "IDAKLUSolverOpenMP.inl"

#endif // PYBAMM_IDAKLU_SOLVEROPENMP_HPP
