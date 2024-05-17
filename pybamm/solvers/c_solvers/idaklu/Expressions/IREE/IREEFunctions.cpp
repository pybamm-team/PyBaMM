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
  input_shape.clear();
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
  output_shape.clear();
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

  // Create index vectors into m_arg
  // This is required since Jax expands input arguments through PyTrees, which need to
  // be remapped to the corresponding expression call. For example:
  //   fcn(t, y, inputs, cj) with inputs = [[in1], [in2], [in3]]
  // will produce a function with six inputs; we therefore need to be able to map
  // arguments to their 1) correspinding input argument, and 2) the correct position
  // within that argument.
  m_arg_argno.clear();
  m_arg_argix.clear();
  int current_element = 0;
  for (int i=0; i<m_func.pytree_shape.size(); i++) {
    int ix = 0;
    bool touched = false;
    for (int j=0; j<m_func.pytree_shape[i]; j++) {
      m_arg_argno.push_back(i);
      m_arg_argix.push_back(ix);
      ix+=m_func.pytree_sizes[current_element++];
      touched = true;
    }
    if (!touched) {
      // Default index into the first argument if length = 0 (won't be read)
      current_element++;
    }
  }

  // Debug print arguments lists
  DEBUG("m_arg_argno:");
  for (int i=0; i<m_arg_argno.size(); i++) {
    DEBUG("  " << i << ": " << m_arg_argno[i]);
  }
  DEBUG("m_arg_argix:");
  for (int i=0; i<m_arg_argix.size(); i++) {
    DEBUG("  " << i << ": " << m_arg_argix[i]);
  }
  DEBUG("m_func.kept_var_idx:");
  for (int i=0; i<m_func.kept_var_idx.size(); i++) {
    DEBUG("  " << i << ": " << m_func.kept_var_idx[i]);
  }
  DEBUG("m_func.pytree_shape:");
  for (int i=0; i<m_func.pytree_shape.size(); i++) {
    DEBUG("  " << i << ": " << m_func.pytree_shape[i]);
  }
  DEBUG("m_func.pytree_sizes:");
  for (int i=0; i<m_func.pytree_sizes.size(); i++) {
    DEBUG("  " << i << ": " << m_func.pytree_sizes[i]);
  }

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

  // Allocate memory for input arguments
  m_arg.resize(m_arg_argno.size(), nullptr);

  // Size iree results vector (single precision) and idaklu results vector (double precision)
  result.resize(output_shape.size());
  for(int k=0; k<output_shape.size(); k++) {
    result[k].resize(output_shape[k][0], 0.0f);
  }
  m_res.resize(output_shape.size(), nullptr);
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
  // pass them (per function) as m_func.kept_var_idx.  Additionally, model inputs can be
  // of arbitrary length, so we need to index into the input arguments using the
  // corresponding shape. This is done by m_arg_argno and m_arg_argix.
  //
  // For example:
  //   def fcn(x, y, z): return 2 * y
  //   produces MLIR with an {arg0} -> {res0} signature (i.e. x and z are reduced out)
  //   with kept_var_idx = [1]
  //
  // ***********************************************************************************

  DEBUG("Copying m_arg to input_data (" << m_func.kept_var_idx.size() << " vars)");
  for (int j=0; j<m_func.kept_var_idx.size(); j++) {
    int mlir_arg = m_func.kept_var_idx[j];
    int m_arg_from = m_arg_argno[mlir_arg];
    int m_arg_to = j;
    if (m_func.pytree_shape[m_arg_from] > 1) {
      // Index into argument using appropriate shape
      DEBUG("Copying m_arg[" << m_arg_from << "] to input_data[" << m_arg_to << "][" << m_arg_argix[mlir_arg] << "..] size " << m_func.pytree_sizes[mlir_arg]);
      for(int k=0; k<m_func.pytree_sizes[mlir_arg]; k++) {
        input_data[m_arg_to][k] = static_cast<float>(m_arg[m_arg_from][m_arg_argix[mlir_arg]+k]);
      }
    } else {
      // Copy the entire vector
      DEBUG("Copying m_arg[" << m_arg_from << "] to input_data[" << m_arg_to << "]");
      for(int k=0; k<input_shape[m_arg_to][0]; k++) {
        input_data[m_arg_to][k] = static_cast<float>(m_arg[m_arg_from][k]);
      }
    }
  }

  // Call the 'main' function of the module
  const std::string mlir = m_func.get_mlir();
  DEBUG("Executing function '" << function_name << "'");
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
    for(size_t j=0; j<result[k].size(); j++) {
      m_res[k][j] = result[k][j];
    }
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
