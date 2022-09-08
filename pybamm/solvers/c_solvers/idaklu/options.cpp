#include "options.hpp"

Options::Options(py::dict options)
    : print_stats(options["print_stats"].cast<bool>()),
      use_jacobian(options["use_jacobian"].cast<bool>()),
      linear_solver(options["linear_solver"].cast<std::string>()),
      dense_jacobian(options["dense_jacobian"].cast<bool>())
{
}
