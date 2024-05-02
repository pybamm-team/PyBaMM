#include "IREEFunctions.hpp"
#include "iree_jit.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

IREEFunction::IREEFunction(const BaseFunctionType &f) : Expression()
{
  std::cout << "IreeFunction constructor" << std::endl;

  //size_t sz_arg;
  size_t sz_res;
  size_t sz_iw;
  size_t sz_w;
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
  std::cout << "Module name: " << module_name << std::endl;

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
  std::cout << "Main function signature: " << main_sig_inputs << " -> " << main_sig_outputs << std::endl;

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
  for(std::sregex_iterator i = std::sregex_iterator(main_sig_outputs.begin(), main_sig_outputs.end(), output_size);
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
    output_shape.push_back(shape);
  }

  std::cout << "Compiling module: '" << module_name << "'" << std::endl;
  const char* device_uri = "local-sync";
  session = std::make_unique<IREESession>(device_uri, f);
  std::cout << " compile complete." << std::endl;

  m_arg.resize(input_shape.size(), nullptr);
  m_res.resize(output_shape.size(), nullptr);

  input_data.resize(input_shape.size());


  /*m_func.sz_work(sz_arg, sz_res, sz_iw, sz_w);
  //int nnz = (sz_res>0) ? m_func.nnz_out() : 0;
  //std::cout << "name = "<< m_func.name() << " arg = " << sz_arg << " res = "
  //  << sz_res << " iw = " << sz_iw << " w = " << sz_w << " nnz = " << nnz <<
  //  std::endl;
  m_arg.resize(sz_arg, nullptr);
  m_res.resize(sz_res, nullptr);
  m_iw.resize(sz_iw, 0);
  m_w.resize(sz_w, 0);*/
}

// only call this once m_arg and m_res have been set appropriately
void IREEFunction::operator()()
{
  std::cout << "IreeFunction operator()" << std::endl;

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
    int m_arg_from = 1;
    int m_arg_to = 0;
    for(int k=0; k<input_shape[m_arg_to][0]; k++) {
      input_data[m_arg_to].push_back(static_cast<float>(m_arg[m_arg_from][k]));
    }
    std::cout << std::endl;
  } else {
    // Default: copy m_arg to input_data
    for (int j=0; j<input_shape.size(); j++) {
      for (int k=0; k<input_shape[j][0]; k++) {
        input_data[j].push_back(static_cast<float>(m_arg[j][k]));
      }
    }

    std::cerr << "Module name: " << module_name << std::endl;
    std::cerr << "Copying input data..." << std::endl;
  }

  std::cout << "Input data arguments: " << input_data.size() << std::endl;
  for (int j=0; j<input_data.size(); j++) {
    std::cout << "  data shape [" << j << "]: " << input_data[j].size() << std::endl;
  }


  // Call the 'main' function of the module
  std::string function_name = module_name + ".main";
  auto status = session->iree_runtime_exec(function_name.c_str(), input_shape, input_data, result);
  if (iree_status_is_ok(status)) {
    std::cout << "Execution succeeded" << std::endl;
  } else {
    std::cout << "Execution failed" << std::endl;
    iree_status_fprint(stderr, status);
    std::cout << "MLIR: " << m_func.substr(0,1000) << std::endl;
    throw std::runtime_error("Execution failed");
  }
  
  std::cout << "Output data arguments: " << output_shape.size() << std::endl;
  for (int j=0; j<output_shape.size(); j++) {
    std::cout << "  data shape[" << j << "]: " << output_shape[j].size() << std::endl;
  }

  // Print result
  std::cout << "Evaluation result for " << module_name << " [" << result.size() << "] (10 samples): ";
  for (size_t i = 0; i < 10; i++)
    std::cout << result[i] << " ";
  std::cout << std::endl;

  // Copy result to output
  std::cout << "Copying result to m_res (size: " << result.size() << ")" << std::endl;
  for(size_t k=0; k<result.size(); k++) {
    m_res[0][k] = result[k];
  }

  std::cout << "done." << std::endl;
}

expr_int IREEFunction::nnz_out() {
  std::cout << "IreeFunction nnz_out" << std::endl;
  /*return static_cast<expr_int>(m_func.nnz_out());*/
  return static_cast<expr_int>(0);
}

ExpressionSparsity *IREEFunction::sparsity_out(expr_int ind) {
  std::cout << "IreeFunction sparsity_out" << std::endl;
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
  // Set-up input arguments, provide result vector, then execute function
  // Example call: fcn({in1, in2, in3}, {out1})
  for(size_t k=0; k<inputs.size(); k++)
    m_arg[k] = inputs[k];
  for(size_t k=0; k<results.size(); k++)
    m_res[k] = results[k];
  operator()();
}
