#include "ModuleParser.hpp"

ModuleParser::ModuleParser(const std::string& mlir) : mlir(mlir)
{
  parse();
}

void ModuleParser::parse()
{
  // Parse module name
  std::regex module_name_regex("module @([^\\s]+)");  // Match until first whitespace
  std::smatch module_name_match;
  std::regex_search(this->mlir, module_name_match, module_name_regex);
  if (module_name_match.size() == 0) {
    std::cerr << "Could not find module name in module" << std::endl;
    std::cerr << "Module snippet: " << this->mlir.substr(0, 1000) << std::endl;
    throw std::runtime_error("Could not find module name in module");
  }
  module_name = module_name_match[1].str();
  DEBUG("Module name: " << module_name);

  // Assign function name
  function_name = module_name + ".main";

  // Isolate 'main' function call signature
  std::regex main_func("public @main\\((.*?)\\) -> \\((.*?)\\)");
  std::smatch match;
  std::regex_search(this->mlir, match, main_func);
  if (match.size() == 0) {
    std::cerr << "Could not find 'main' function in module" << std::endl;
    std::cerr << "Module snippet: " << this->mlir.substr(0, 1000) << std::endl;
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
    // If shape is empty, assume scalar (i.e. "tensor<f32>" or some singleton variant)
    if (shape.size() == 0) {
      shape.push_back(1);
    }
    // Add output to list
    output_shape.push_back(shape);
  }
}
