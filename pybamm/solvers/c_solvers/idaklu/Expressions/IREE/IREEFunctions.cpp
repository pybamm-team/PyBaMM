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
  const std::string& mlir = f.get_mlir();

  // Parse IREE (MLIR) function string
  if (mlir.size() == 0) {
    DEBUG("Empty function --- skipping...");
    return;
  }

  // Parse module name
  std::regex module_name_regex("module @([^\\s]+)");  // Match until first whitespace
  std::smatch module_name_match;
  std::regex_search(mlir, module_name_match, module_name_regex);
  if (module_name_match.size() == 0) {
    std::cerr << "Could not find module name in module" << std::endl;
    std::cerr << "Module snippet: " << mlir.substr(0, 1000) << std::endl;
    throw std::runtime_error("Could not find module name in module");
  }
  module_name = module_name_match[1].str();
  DEBUG("Module name: " << module_name);

  // Assign function name
  function_name = module_name + ".main";

  // Isolate 'main' function call signature
  std::regex main_func("public @main\\((.*?)\\) -> \\((.*?)\\)");
  std::smatch match;
  std::regex_search(mlir, match, main_func);
  if (match.size() == 0) {
    std::cerr << "Could not find 'main' function in module" << std::endl;
    std::cerr << "Module snippet: " << mlir.substr(0, 1000) << std::endl;
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
  session = std::make_unique<IREESession>(device_uri, mlir);
  DEBUG("compile complete.");

  // Size ranges are unreliable, so we need to overallocate memory
  //m_arg.resize(input_shape.size(), nullptr);
  //m_res.resize(output_shape.size(), nullptr);
  m_arg.resize(10, nullptr);  // 10 inputs should be enough
  m_res.resize(1, nullptr);  // pretty sure we only ever have one output


  // Allocate memory for result (also check that the input is a vector)
  input_data.resize(input_shape.size());
  for(int j=0; j<input_shape.size(); j++) {
    if ((input_shape[j].size() > 2) || ((input_shape[j].size() == 2) && (input_shape[j][1] > 1))) {
      std::cerr << "Unsupported input shape: " << input_shape[j].size() << " [";
      for (int k=0; k<input_shape[j].size(); k++) {
        std::cerr << input_shape[j][k] << " ";
      }
      std::cerr << "]" << std::endl;
      throw std::runtime_error("Only 1D inputs are supported");
    }
    input_data[j].resize(input_shape[j][0]);  // assumes 1D input
  }

  // Check output count
  if (output_shape.size() != 1) {
    std::cerr << "Unsupported output shape: " << output_shape.size() << std::endl;
    throw std::runtime_error("Only single outputs are supported");
  }
}

// only call this once m_arg and m_res have been set appropriately
void IREEFunction::operator()()
{
  DEBUG("IreeFunction operator(): " << module_name);

  // ***********************************************************************************
  // 
  // MLIR output from Jax does not retain the proper call signature of the original
  // function. This appears to be due to aggressive optimisations in the lowering
  // process. As a result, we need to manually map the input arguments to the
  // correct positions in the MLIR function signature. We obtain these in Python and
  // pass them (per function) as m_func.kept_var_idx.
  // 
  // For example:
  //   def fcn(x, y, z): return 2 * y
  //   produces MLIR with an {arg0} -> {res0} signature (i.e. x and z are reduced out)
  //   with kept_var_idx = [1]
  //
  // ***********************************************************************************
  

  DEBUG("Copying m_arg to input_data (" << m_func.kept_var_idx.size() << " vars)");
  for (int j=0; j<m_func.kept_var_idx.size(); j++) {
    int m_arg_from = m_func.kept_var_idx[j];
    int m_arg_to = j;
    DEBUG("Copying m_arg[" << m_arg_from << "] to input_data[" << m_arg_to << "]");
    for(int k=0; k<input_shape[m_arg_to][0]; k++) {
      input_data[m_arg_to][k] = static_cast<float>(m_arg[m_arg_from][k]);
    }
  }

  // Call the 'main' function of the module
  const std::string mlir = m_func.get_mlir();
  auto status = session->iree_runtime_exec(function_name, input_shape, input_data, result);
  if (iree_status_is_ok(status)) {
    DEBUG("MLIR execution successful");
  } else {
    iree_status_fprint(stderr, status);
    std::cerr << "MLIR: " << mlir.substr(0,1000) << std::endl;
    throw std::runtime_error("Execution failed");
  }

  // Copy result to output
  for(size_t k=0; k<result.size(); k++) {
    m_res[0][k] = result[k];
  }
}

expr_int IREEFunction::nnz_out() {
  DEBUG("IreeFunction nnz_out");
  throw std::runtime_error("IreeFunction nnz_out not implemented");
  /*return static_cast<expr_int>(m_func.nnz_out());*/
  return static_cast<expr_int>(0);
}

ExpressionSparsity *IREEFunction::sparsity_out(expr_int ind) {
  DEBUG("IreeFunction sparsity_out");
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
  DEBUG("IreeFunction operator() with inputs and results");
  throw std::runtime_error("IreeFunction operator() with inputs and results not implemented");
  // Set-up input arguments, provide result vector, then execute function
  // Example call: fcn({in1, in2, in3}, {out1})
  for(size_t k=0; k<inputs.size(); k++)
    m_arg[k] = inputs[k];
  for(size_t k=0; k<results.size(); k++)
    m_res[k] = results[k];
  operator()();
}
