#include "options.hpp"
#include <iostream>
#include <stdexcept>

 
using namespace std::string_literals;

Options::Options(py::dict options)
    : print_stats(options["print_stats"].cast<bool>()),
      jacobian(options["jacobian"].cast<std::string>()),
      preconditioner(options["preconditioner"].cast<std::string>()),
      linsol_max_iterations(options["linsol_max_iterations"].cast<int>()),
      linear_solver(options["linear_solver"].cast<std::string>()),
      precon_half_bandwidth(options["precon_half_bandwidth"].cast<int>()),
      precon_half_bandwidth_keep(options["precon_half_bandwidth_keep"].cast<int>()),
      num_threads(options["num_threads"].cast<int>())
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
