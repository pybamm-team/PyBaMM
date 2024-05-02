#include "IREEFunctions.hpp"
#include "iree_jit.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

IREEFunction::IREEFunction(const BaseFunctionType &f) : Expression()
{
  DEBUG("IreeFunction constructor");
  m_func = f;

  // Parse IREE (MLIR) function string
  if (f.size() == 0) {
    std::cout << "Empty function --- skipping..." << std::endl;
    return;
  }

  // Parse module name
  std::regex module_name_regex("module @([^\\s]+)");  // Match until first whitespace
  std::smatch module_name_match;
  std::regex_search(f, module_name_match, module_name_regex);
  if (module_name_match.size() == 0) {
    std::cerr << "Could not find module name in module" << std::endl;
    std::cerr << "Module snippet: " << f.substr(0, 1000) << std::endl;
    throw std::runtime_error("Could not find module name in module");
  }
  module_name = module_name_match[1].str();

  // Assign function name
  function_name = module_name + ".main";

  // Isolate 'main' function call signature
  std::regex main_func("public @main\\((.*?)\\) -> \\((.*?)\\)");
  std::smatch match;
  std::regex_search(f, match, main_func);
  if (match.size() == 0) {
    std::cerr << "Could not find 'main' function in module" << std::endl;
    std::cerr << "Module snippet: " << f.substr(0, 1000) << std::endl;
    throw std::runtime_error("Could not find 'main' function in module");
  }
  std::string main_sig_inputs = match[1].str();
  std::string main_sig_outputs = match[2].str();
  DEBUG(
    "Main function signature: " << main_sig_inputs << " -> " << main_sig_outputs << '\n'
  );

  // Parse input sizes
  std::regex input_size("tensor<(.*?)>");
  for(std::sregex_iterator i = std::sregex_iterator(main_sig_inputs.begin(), main_sig_inputs.end(), input_size);
      i != std::sregex_iterator();
      ++i)
  {
    std::smatch matchi = *i;
    std::string match_str = matchi.str();
    std::string shape_str = match_str.substr(7, match_str.size() - 8);  // Remove 'tensor<>' from string
    std::vector<int> shape;
    std::string dim_str;
    for (char c : shape_str) {
      if (c == 'x') {
        shape.push_back(std::stoi(dim_str));
        dim_str = "";
      } else {
        dim_str += c;
      }
    }
    input_shape.push_back(shape);
  }

  // Parse output sizes
  std::regex output_size("tensor<(.*?)>");
  for(
    std::sregex_iterator i = std::sregex_iterator(main_sig_outputs.begin(), main_sig_outputs.end(), output_size);
    i != std::sregex_iterator();
    ++i
  ) {
    std::smatch matchi = *i;
    std::string match_str = matchi.str();
    std::string shape_str = match_str.substr(7, match_str.size() - 8);  // Remove 'tensor<>' from string
    std::vector<int> shape;
    std::string dim_str;
    for (char c : shape_str) {
      if (c == 'x') {
        shape.push_back(std::stoi(dim_str));
        dim_str = "";
      } else {
        dim_str += c;
      }
    }
    output_shape.push_back(shape);
  }

  DEBUG("Compiling module: '" << module_name << "'");
  const char* device_uri = "local-sync";
  session = std::make_unique<IREESession>(device_uri, f);
  DEBUG("compile complete.");

  m_arg.resize(input_shape.size(), nullptr);
  m_res.resize(output_shape.size(), nullptr);

  input_data.resize(input_shape.size());
  for(int j=0; j<input_shape.size(); j++) {
    input_data[j].resize(input_shape[j][0]);  // assumes 1D input
  }
}

// only call this once m_arg and m_res have been set appropriately
void IREEFunction::operator()()
{
  DEBUG("IreeFunction operator(): " << module_name);

  // ***********************************************************************************
  // Specialise to each function call (should not be necessary, but inputs are not being
  // carried over from JAX reliably; this appears to be due to aggressive optimisations
  // in the lowering process).
  // 
  // For instance:
  //   def fcn(x, y, z): return x + y + z
  //   produces MLIR with an {arg0, arg1, arg2} -> {res0} call signature
  //
  // But:
  //   def fcn(x, y, z): return 2 * y
  //   produces MLIR with an {arg0} -> {res0} signature (i.e. x and z are reduced out)
  //
  // ***********************************************************************************

  // rhs_algebraic
  if (module_name == "jit_fcn_rhs_algebraic") {
    std::cerr << "Identified rhs_algebraic function --- reassigning input arguments..." << std::endl;
    int m_arg_from = 1;
    int m_arg_to = 0;
    for(int k=0; k<input_shape[m_arg_to][0]; k++) {
      input_data[m_arg_to][k] = static_cast<float>(m_arg[m_arg_from][k]);
    }
  } else {
    // Default: copy m_arg to input_data
    for (int j=0; j<input_shape.size(); j++) {
      for (int k=0; k<input_shape[j][0]; k++) {
        input_data[j][k] = static_cast<float>(m_arg[j][k]);
      }
    }
  }

  // Call the 'main' function of the module
  const int RETRIES = 5;
  for (int k=0; k < RETRIES; k++) {
    auto status = session->iree_runtime_exec(function_name, input_shape, input_data, result);
    if (iree_status_is_ok(status)) {
      break;
    } else if (k == RETRIES-1) {
      std::cerr << "Execution failed" << std::endl;
      iree_status_fprint(stderr, status);
      std::cerr << "MLIR: " << m_func.substr(0,1000) << std::endl;
      throw std::runtime_error("Execution failed");
    } else {
      std::cerr << "Execution failed (attempt " << k << "), retrying..." << std::endl;
      iree_status_fprint(stderr, status);
    }
  }

  // Copy result to output
  for(size_t k=0; k<result.size(); k++) {
    m_res[0][k] = result[k];
  }
}

expr_int IREEFunction::nnz_out() {
  std::cout << "IreeFunction nnz_out" << std::endl;
  throw std::runtime_error("IreeFunction nnz_out not implemented");
  /*return static_cast<expr_int>(m_func.nnz_out());*/
  return static_cast<expr_int>(0);
}

ExpressionSparsity *IREEFunction::sparsity_out(expr_int ind) {
  std::cout << "IreeFunction sparsity_out" << std::endl;
  throw std::runtime_error("IreeFunction sparsity_out not implemented");
  /*iree::Sparsity iree_sparsity = m_func.sparsity_out(ind);
  IreeSparsity *cs = new IreeSparsity();
  cs->_nnz = iree_sparsity.nnz();
  cs->_get_row = iree_sparsity.get_row();
  cs->_get_col = iree_sparsity.get_col();
  return cs;*/
  return nullptr;
}

void IREEFunction::operator()(const std::vector<realtype*>& inputs,
                                const std::vector<realtype*>& results)
{
  std::cout << "IreeFunction operator() with inputs and results" << std::endl;
  throw std::runtime_error("IreeFunction operator() with inputs and results not implemented");
  // Set-up input arguments, provide result vector, then execute function
  // Example call: fcn({in1, in2, in3}, {out1})
  for(size_t k=0; k<inputs.size(); k++)
    m_arg[k] = inputs[k];
  for(size_t k=0; k<results.size(); k++)
    m_res[k] = results[k];
  operator()();
}
