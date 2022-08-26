#include "options.hpp"

Options::Options(py::dict options)
    : print_stats(options["print_stats"].cast<bool>()),
      use_jacobian(options["use_jacobian"].cast<bool>())
{
}
