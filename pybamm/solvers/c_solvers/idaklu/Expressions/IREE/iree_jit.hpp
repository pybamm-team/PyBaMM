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

class IREESession;

class IREECompiler {
private:
  const char *device_uri = NULL;
  std::vector<IREESession> iree_sessions;
private:
  int initIREE(int argc, const char **argv);
public:
  IREECompiler();
  ~IREECompiler();
  explicit IREECompiler(const char *device_uri) : IREECompiler() { this->device_uri=device_uri; }
  int init(int argc, const char **argv);
  IREESession addSession(const std::string& mlir_fcn);
  std::vector<IREESession> addSessions(const std::vector<std::string>& mlir_fcns);
  int cleanup();
  void testSessions();
};

typedef struct compiler_state_t {
  iree_compiler_session_t *session;
  iree_compiler_source_t *source;
  iree_compiler_output_t *output;
  iree_compiler_invocation_t *inv;
} compiler_state_t;

class IREESession {
// Properties
private:
  const char *device_uri = NULL;
  compiler_state_t s;
  iree_compiler_error_t *error = NULL;
  void *contents = NULL;
  uint64_t size = 0;
  iree_runtime_session_t* session = NULL;
  iree_status_t status;
  iree_hal_device_t* device = NULL;
  iree_runtime_instance_t* instance = NULL;
  std::string mlir_code;
  iree_runtime_call_t call;
  iree_allocator_t host_allocator;

// Methods
private:
  void handle_compiler_error(iree_compiler_error_t *error);
  void cleanup_compiler_state(compiler_state_t s);
  int init();
  int initCompiler();
  int initCompileToByteCode();
  int initRuntime();
public:
  IREESession();
  explicit IREESession(const char *device_uri, const std::string& mlir_code);
  int buildAndIssueCall(std::string function_name);
  int cleanup();
  // IREE runtime functions
  iree_status_t iree_runtime_exec(
    std::string function_name,
    const std::vector<std::vector<int>>& inputs,
    const std::vector<std::vector<float>>& data,
    std::vector<std::vector<float>>& result
  );
};

#endif // IREE_JIT_HPP
