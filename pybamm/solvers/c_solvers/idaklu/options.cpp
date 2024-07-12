#include "options.hpp"
#include <iostream>
#include <stdexcept>


using namespace std::string_literals;

SetupOptions::SetupOptions(py::dict &options)
    : jacobian(options["jacobian"].cast<std::string>()),
      preconditioner(options["preconditioner"].cast<std::string>()),
      precon_half_bandwidth(options["precon_half_bandwidth"].cast<int>()),
      precon_half_bandwidth_keep(options["precon_half_bandwidth_keep"].cast<int>()),
      num_threads(options["num_threads"].cast<int>()),
      linear_solver(options["linear_solver"].cast<std::string>()),
      linsol_max_iterations(options["linsol_max_iterations"].cast<int>())
{

    using_sparse_matrix = true;
    using_banded_matrix = false;
    if (jacobian == "sparse")
    {
    }
    else if (jacobian == "banded") {
        using_banded_matrix = true;
        using_sparse_matrix = false;
    }
    else if (jacobian == "dense" || jacobian == "none")
    {
        using_sparse_matrix = false;
    }
    else if (jacobian == "matrix-free")
    {
    }
    else
    {
        throw std::domain_error(
            "Unknown jacobian type \""s + jacobian +
            "\". Should be one of \"sparse\", \"banded\", \"dense\", \"matrix-free\" or \"none\"."s
        );
    }

    using_iterative_solver = false;
    if (linear_solver == "SUNLinSol_Dense" && (jacobian == "dense" || jacobian == "none"))
    {
    }
    else if (linear_solver == "SUNLinSol_KLU" && jacobian == "sparse")
    {
    }
    else if (linear_solver == "SUNLinSol_cuSolverSp_batchQR" && jacobian == "sparse")
    {
    }
    else if (linear_solver == "SUNLinSol_Band" && jacobian == "banded")
    {
    }
    else if (jacobian == "banded") {
        throw std::domain_error(
            "Unknown linear solver or incompatible options: "
            "jacobian = \"" + jacobian + "\" linear solver = \"" + linear_solver +
            "\". For a banded jacobian "
            "please use the SUNLinSol_Band linear solver"
        );
    }
    else if ((linear_solver == "SUNLinSol_SPBCGS" ||
              linear_solver == "SUNLinSol_SPFGMR" ||
              linear_solver == "SUNLinSol_SPGMR" ||
              linear_solver == "SUNLinSol_SPTFQMR") &&
             (jacobian == "sparse" || jacobian == "matrix-free"))
    {
        using_iterative_solver = true;
    }
    else if (jacobian == "sparse")
    {
        throw std::domain_error(
            "Unknown linear solver or incompatible options: "
            "jacobian = \"" + jacobian + "\" linear solver = \"" + linear_solver +
            "\". For a sparse jacobian "
            "please use the SUNLinSol_KLU linear solver"
        );
    }
    else if (jacobian == "matrix-free")
    {
        throw std::domain_error(
            "Unknown linear solver or incompatible options. "
            "jacobian = \"" + jacobian + "\" linear solver = \"" + linear_solver +
            "\". For a matrix-free jacobian "
            "please use one of the iterative linear solvers: \"SUNLinSol_SPBCGS\", "
            "\"SUNLinSol_SPFGMR\", \"SUNLinSol_SPGMR\", or \"SUNLinSol_SPTFQMR\"."
        );
    }
    else if (jacobian == "none")
    {
        throw std::domain_error(
            "Unknown linear solver or incompatible options: "
            "jacobian = \"" + jacobian + "\" linear solver = \"" + linear_solver +
            "\". For no jacobian please use the SUNLinSol_Dense solver"
        );
    }
    else
    {
        throw std::domain_error(
            "Unknown linear solver or incompatible options. "
            "jacobian = \"" + jacobian + "\" linear solver = \"" + linear_solver + "\""
        );
    }

    if (using_iterative_solver)
    {
        if (preconditioner != "none" && preconditioner != "BBDP")
        {
            throw std::domain_error(
                "Unknown preconditioner \""s + preconditioner +
                "\", use one of \"BBDP\" or \"none\""s
            );
        }
    }
    else
    {
        preconditioner = "none";
    }
}

SolverOptions::SolverOptions(py::dict &options)
    : print_stats(options["print_stats"].cast<bool>()),
      // IDA main solver
      max_order_bdf(options["max_order_bdf"].cast<int>()),
      max_num_steps(options["max_num_steps"].cast<int>()),
      dt_init(RCONST(options["dt_init"].cast<double>())),
      dt_max(RCONST(options["dt_max"].cast<double>())),
      max_error_test_failures(options["max_error_test_failures"].cast<int>()),
      max_nonlinear_iterations(options["max_nonlinear_iterations"].cast<int>()),
      max_convergence_failures(options["max_convergence_failures"].cast<int>()),
      nonlinear_convergence_coefficient(RCONST(options["nonlinear_convergence_coefficient"].cast<double>())),
      nonlinear_convergence_coefficient_ic(RCONST(options["nonlinear_convergence_coefficient_ic"].cast<double>())),
      suppress_algebraic_error(options["suppress_algebraic_error"].cast<sunbooleantype>()),
      // IDA initial conditions calculation
      calc_ic(options["calc_ic"].cast<bool>()),
      init_all_y_ic(options["init_all_y_ic"].cast<bool>()),
      max_num_steps_ic(options["max_num_steps_ic"].cast<int>()),
      max_num_jacobians_ic(options["max_num_jacobians_ic"].cast<int>()),
      max_num_iterations_ic(options["max_num_iterations_ic"].cast<int>()),
      max_linesearch_backtracks_ic(options["max_linesearch_backtracks_ic"].cast<int>()),
      linesearch_off_ic(options["linesearch_off_ic"].cast<sunbooleantype>()),
      // IDALS linear solver interface
      linear_solution_scaling(options["linear_solution_scaling"].cast<sunbooleantype>()),
      epsilon_linear_tolerance(RCONST(options["epsilon_linear_tolerance"].cast<double>())),
      increment_factor(RCONST(options["increment_factor"].cast<double>()))
{}
