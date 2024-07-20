#ifndef IREE_JIT_HPP
#define IREE_JIT_HPP

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>
#include <iree/runtime/api.h>

#define IREE_COMPILER_EXPECTED_API_MAJOR 1 // At most this major version
#define IREE_COMPILER_EXPECTED_API_MINOR 2 // At least this minor version

// Forward declaration
class IREESession;

/*
 * @brief IREECompiler class
 * @details This class is used to compile MLIR code to IREE bytecode and
 *         create IREE sessions.
 */
class IREECompiler {
private:
  /*
   * @brief Device Uniform Resource Identifier (URI)
   * @details The device URI is used to specify the device to be used by the
   *         IREE runtime. E.g. "local-sync" for CPU, "vulkan" for GPU, etc.
   */
  const char *device_uri = NULL;

private:
  /*
   * @brief Initialize the IREE runtime
   */
  int initIREE(int argc, const char **argv);

public:
  /*
   * @brief Default constructor
   */
  IREECompiler();

  /*
   * @brief Destructor
   */
  ~IREECompiler();

  /*
   * @brief Constructor with device URI
   * @param device_uri Device URI
   */
  explicit IREECompiler(const char *device_uri)
    : IREECompiler() { this->device_uri=device_uri; }

  /*
   * @brief Initialize the compiler
   */
  int init(int argc, const char **argv);

  /*
   * @brief Cleanup the compiler
   * @details This method cleans up the compiler and all the IREE sessions
   *        created by the compiler. Returns 0 on success.
   */
  int cleanup();
};

/*
 * @brief Compiler state
 */
typedef struct compiler_state_t {
  iree_compiler_session_t *session;  // cppcheck-suppress unusedStructMember
  iree_compiler_source_t *source;  // cppcheck-suppress unusedStructMember
  iree_compiler_output_t *output;  // cppcheck-suppress unusedStructMember
  iree_compiler_invocation_t *inv;  // cppcheck-suppress unusedStructMember
} compiler_state_t;

/*
 * @brief IREE session class
 */
class IREESession {
private:  // data members
  const char *device_uri = NULL;
  compiler_state_t s;
  iree_compiler_error_t *error = NULL;
  void *contents = NULL;
  uint64_t size = 0;
  iree_runtime_session_t* session = NULL;
  iree_status_t status;
  iree_hal_device_t* device = NULL;
  iree_runtime_instance_t* instance = NULL;
  std::string mlir_code;  // cppcheck-suppress unusedStructMember
  iree_runtime_call_t call;
  iree_allocator_t host_allocator;

private:  // private methods
  void handle_compiler_error(iree_compiler_error_t *error);
  void cleanup_compiler_state(compiler_state_t s);
  int init();
  int initCompiler();
  int initCompileToByteCode();
  int initRuntime();

public:  // public methods

  /*
   * @brief Default constructor
   */
  IREESession();

  /*
   * @brief Constructor with device URI and MLIR code
   * @param device_uri Device URI
   * @param mlir_code MLIR code
   */
  explicit IREESession(const char *device_uri, const std::string& mlir_code);

  /*
   * @brief Cleanup the IREE session
   */
  int cleanup();

  /*
   * @brief Execute the pre-compiled byte-code with the given inputs
   * @param function_name Function name to execute
   * @param inputs List of input shapes
   * @param data List of input data
   * @param result List of output data
   */
  iree_status_t iree_runtime_exec(
    const std::string& function_name,
    const std::vector<std::vector<int>>& inputs,
    const std::vector<std::vector<float>>& data,
    std::vector<std::vector<float>>& result
  );
};

#endif // IREE_JIT_HPP
