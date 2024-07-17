#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

#include "IREEFunctions.hpp"
#include "iree_jit.hpp"
#include "ModuleParser.hpp"

IREEFunction::IREEFunction(const BaseFunctionType &f) : Expression(), m_func(f)
{
  DEBUG("IreeFunction constructor");
  const std::string& mlir = f.get_mlir();

  // Parse IREE (MLIR) function string
  if (mlir.size() == 0) {
    DEBUG("Empty function --- skipping...");
    return;
  }

  // Parse MLIR for module name, input and output shapes
  ModuleParser parser(mlir);
  module_name = parser.getModuleName();
  function_name = parser.getFunctionName();
  input_shape = parser.getInputShape();
  output_shape = parser.getOutputShape();

  DEBUG("Compiling module: '" << module_name << "'");
  const char* device_uri = "local-sync";
  session = std::make_unique<IREESession>(device_uri, mlir);
  DEBUG("compile complete.");
  // Create index vectors into m_arg
  // This is required since Jax expands input arguments through PyTrees, which need to
  // be remapped to the corresponding expression call. For example:
  //   fcn(t, y, inputs, cj) with inputs = [[in1], [in2], [in3]]
  // will produce a function with six inputs; we therefore need to be able to map
  // arguments to their 1) corresponding input argument, and 2) the correct position
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
  DEBUG("Number of original function arguments: " << m_func.n_args);
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
  numel.resize(input_shape.size());
  for(int j=0; j<input_shape.size(); j++) {
    if (
      (input_shape[j].size() > 2) ||
      ((input_shape[j].size() == 2) && (input_shape[j][1] > 1))
    ) {
      std::cerr << "Unsupported input shape: " << input_shape[j].size() << " [";
      for (int k=0; k<input_shape[j].size(); k++) {
        std::cerr << input_shape[j][k] << " ";
      }
      std::cerr << "]" << std::endl;
      throw std::runtime_error("Only 1D column vectors are supported as input arguments");
    }
    int count = 1;
    for(int k=0; k<input_shape[j].size(); k++) {
      count *= input_shape[j][k];
    }
    numel[j] = count;
    input_data[j].resize(numel[j]);
  }

  // Allocate memory for input arguments
  m_arg.resize(m_func.n_args, nullptr);

  // Size iree results vector (single precision) and casadi results vector (double precision)
  result.clear();
  result.resize(output_shape.size());
  for(int k=0; k<output_shape.size(); k++) {
    DEBUG("Output " << k << " size: " << output_shape[k][0]);
    auto elements = 1;
    for (auto i : output_shape[k]) {
      elements *= i;
    }
    // Sparse functions return NNZ elements, so we don't need to worry about sparsity
    result[k].resize(elements, 0.0f);
  }
  m_res.resize(output_shape.size(), nullptr);
}

// only call this once m_arg and m_res have been set appropriately
void IREEFunction::operator()()
{
  DEBUG("IreeFunction operator(): " << module_name);
  evaluate(output_shape.size());  // return all outputs
}

void IREEFunction::evaluate(int n_outputs) {
  // n_outputs is the number of outputs to return

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

  DEBUG("Copying inputs, shape " << input_shape.size() << " - " << m_func.kept_var_idx.size());
  for (int j=0; j<m_func.kept_var_idx.size(); j++) {
    int mlir_arg = m_func.kept_var_idx[j];
    int m_arg_from = m_arg_argno[mlir_arg];
    int m_arg_to = j;
    if (m_func.pytree_shape[m_arg_from] > 1) {
      // Index into argument using appropriate shape
      for(int k=0; k<m_func.pytree_sizes[mlir_arg]; k++) {
        input_data[m_arg_to][k] = static_cast<float>(m_arg[m_arg_from][m_arg_argix[mlir_arg]+k]);
      }
    } else {
      // Copy the entire vector
      for(int k=0; k<input_shape[m_arg_to][0]; k++) {
        input_data[m_arg_to][k] = static_cast<float>(m_arg[m_arg_from][k]);
      }
    }
  }

  // Call the 'main' function of the module
  const std::string mlir = m_func.get_mlir();
  DEBUG("Calling function '" << function_name << "'");
  auto status = session->iree_runtime_exec(function_name, input_shape, input_data, result);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    std::cerr << "MLIR: " << mlir.substr(0,1000) << std::endl;
    throw std::runtime_error("Execution failed");
  }

  // Copy results to output array
  for(size_t k=0; k<n_outputs; k++) {
    for(size_t j=0; j<result[k].size(); j++) {
      m_res[k][j] = static_cast<realtype>(result[k][j]);
    }
  }

  DEBUG("IreeFunction operator() complete");
}

expr_int IREEFunction::out_shape(int k) {
  DEBUG("IreeFunction nnz(" << k << "): " << m_func.nnz);
  auto elements = 1;
  for (auto i : output_shape[k]) {
    elements *= i;
  }
  return elements;
}

expr_int IREEFunction::nnz() {
  DEBUG("IreeFunction nnz: " << m_func.nnz);
  return nnz_out();
}

expr_int IREEFunction::nnz_out() {
  DEBUG("IreeFunction nnz_out" << m_func.nnz);
  return m_func.nnz;
}

std::vector<expr_int> IREEFunction::get_row() {
  DEBUG("IreeFunction get_row" << m_func.row.size());
  return m_func.row;
}

std::vector<expr_int> IREEFunction::get_col() {
  DEBUG("IreeFunction get_col" << m_func.col.size());
  return m_func.col;
}

void IREEFunction::operator()(const std::vector<realtype*>& inputs,
                                const std::vector<realtype*>& results)
{
  DEBUG("IreeFunction operator() with inputs and results");
  // Set-up input arguments, provide result vector, then execute function
  // Example call: fcn({in1, in2, in3}, {out1})
  ASSERT(inputs.size() == m_func.n_args);
  for(size_t k=0; k<inputs.size(); k++) {
    m_arg[k] = inputs[k];  // Copy references to vectors, not elements
  }
  for(size_t k=0; k<results.size(); k++) {
    m_res[k] = results[k];  // Copy references to vectors, not elements
  }
  evaluate(results.size());  // only copy the number of requested outputs back
}
