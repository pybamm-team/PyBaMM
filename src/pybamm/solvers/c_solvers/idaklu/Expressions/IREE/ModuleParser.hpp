#ifndef PYBAMM_IDAKLU_IREE_MODULE_PARSER_HPP
#define PYBAMM_IDAKLU_IREE_MODULE_PARSER_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

#include "../../common.hpp"

class ModuleParser {
private:
  std::string mlir;  // cppcheck-suppress unusedStructMember
                     // codacy fix: member is referenced as this->mlir in parse()
  std::string module_name;
  std::string function_name;
  std::vector<std::vector<int>> input_shape;
  std::vector<std::vector<int>> output_shape;
public:
  /**
   * @brief Constructor
   * @param mlir: string representation of MLIR code for the module
   */
  explicit ModuleParser(const std::string& mlir);

  /**
   * @brief Get the module name
   * @return module name
   */
  const std::string& getModuleName() const { return module_name; }

  /**
   * @brief Get the function name
   * @return function name
   */
  const std::string& getFunctionName() const { return function_name; }

  /**
   * @brief Get the input shape
   * @return input shape
   */
  const std::vector<std::vector<int>>& getInputShape() const { return input_shape; }

  /**
   * @brief Get the output shape
   * @return output shape
   */
  const std::vector<std::vector<int>>& getOutputShape() const { return output_shape; }

private:
  void parse();
};

#endif // PYBAMM_IDAKLU_IREE_MODULE_PARSER_HPP
