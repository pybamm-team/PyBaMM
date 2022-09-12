#include "options.hpp"

Options::Options(py::dict options)
    : print_stats(options["print_stats"].cast<bool>()),
      jacobian(options["jacobian"].cast<std::string>()),
      preconditioner(options["preconditioner"].cast<std::string>()),
      linsol_max_iterations(options["linsol_max_iterations"].cast<int>()),
      linear_solver(options["linear_solver"].cast<std::string>())
{

  using_sparse_matrix = true;
  if (jacobian == "sparse")
  {
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
    py::print("Unknown jacobian type, using sparse by default");
    jacobian = "sparse";
  }

  using_iterative_solver = false;
  if (linear_solver == "SUNLinSol_Dense" && jacobian == "dense")
  {
  }
  else if (linear_solver == "SUNLinSol_LapackDense" && jacobian == "dense")
  {
  }
  else if (linear_solver == "SUNLinSol_KLU" && jacobian == "sparse")
  {
  }
  else if (linear_solver == "SUNLinSol_SPBCGS" &&
           (jacobian == "sparse" || jacobian == "matrix-free"))
  {
    using_iterative_solver = true;
  }
  else if (jacobian == "sparse")
  {
    py::print("Unknown linear solver or incompatible options using "
              "SUNLinSol_KLU by default");
    linear_solver = "SUNLinSol_KLU";
  }
  else if (jacobian == "matrix-free")
  {
    py::print("Unknown linear solver or incompatible options using "
              "SUNLinSol_SPBCGS by default");
    linear_solver = "SUNLinSol_SPBCGS";
    using_iterative_solver = true;
  }
  else
  {
    py::print("Unknown linear solver or incompatible options using "
              "SUNLinSol_Dense by default");
    linear_solver = "SUNLinSol_Dense";
  }

  if (using_iterative_solver)
  {
    if (preconditioner != "none" && preconditioner != "BBDP")
    {
      py::print("Unknown preconditioner using BBDP by default");
      preconditioner = "BBDP";
    }
  }
  else
  {
    preconditioner = "none";
  }
}
