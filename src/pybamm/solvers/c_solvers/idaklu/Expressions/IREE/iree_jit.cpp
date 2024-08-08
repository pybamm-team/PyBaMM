#include "iree_jit.hpp"
#include "iree/hal/buffer_view.h"
#include "iree/hal/buffer_view_util.h"
#include "../../common.hpp"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <cstdio>
#include <unistd.h>

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>
#include <iree/runtime/api.h>

// Used to suppress stderr output (see initIREE below)
#ifdef _WIN32
#include <io.h>
#define close _close
#define dup _dup
#define fileno _fileno
#define open _open
#define dup2 _dup2
#define NULL_DEVICE "NUL"
#else
#define NULL_DEVICE "/dev/null"
#endif

void IREESession::handle_compiler_error(iree_compiler_error_t *error) {
  const char *msg = ireeCompilerErrorGetMessage(error);
  fprintf(stderr, "Error from compiler API:\n%s\n", msg);
  ireeCompilerErrorDestroy(error);
}

void IREESession::cleanup_compiler_state(compiler_state_t s) {
  if (s.inv)
    ireeCompilerInvocationDestroy(s.inv);
  if (s.output)
    ireeCompilerOutputDestroy(s.output);
  if (s.source)
    ireeCompilerSourceDestroy(s.source);
  if (s.session)
    ireeCompilerSessionDestroy(s.session);
}

IREECompiler::IREECompiler() {
  this->device_uri = "local-sync";
};

IREECompiler::~IREECompiler() {
  ireeCompilerGlobalShutdown();
};

int IREECompiler::init(int argc, const char **argv) {
  return initIREE(argc, argv);  // Initialisation and version checking
};

int IREECompiler::cleanup() {
  return 0;
};

IREESession::IREESession() {
  s.session = NULL;
  s.source = NULL;
  s.output = NULL;
  s.inv = NULL;
};

IREESession::IREESession(const char *device_uri, const std::string& mlir_code) : IREESession() {
  this->device_uri=device_uri;
  this->mlir_code=mlir_code;
  init();
}

int IREESession::init() {
  if (initCompiler() != 0)  // Prepare compiler inputs and outputs
    return 1;
  if (initCompileToByteCode() != 0)  // Compile to bytecode
    return 1;
  if (initRuntime() != 0)  // Initialise runtime environment
    return 1;
  return 0;
};

int IREECompiler::initIREE(int argc, const char **argv) {

  if (device_uri == NULL) {
    DEBUG("No device URI provided, using local-sync\n");
    this->device_uri = "local-sync";
  }

  int cl_argc = argc;
  const char *iree_compiler_lib = std::getenv("IREE_COMPILER_LIB");

  // Load the compiler library and initialize it
  // NOTE: On second and subsequent calls, the function will return false and display
  // a message on stderr, but it is safe to ignore this message. For an improved user
  // experience we actively suppress stderr during the call to this function but since
  // this also suppresses any other error message, we actively check for the presence
  // of the library file prior to the call.

  // Check if the library file exists
  if (iree_compiler_lib == NULL) {
    fprintf(stderr, "Error: IREE_COMPILER_LIB environment variable not set\n");
    return 1;
  }
  if (access(iree_compiler_lib, F_OK) == -1) {
    fprintf(stderr, "Error: IREE_COMPILER_LIB file not found\n");
    return 1;
  }
  // Suppress stderr
  int saved_stderr = dup(fileno(stderr));
  if (!freopen(NULL_DEVICE, "w", stderr))
    DEBUG("Error: failed redirecting stderr");
  // Load library
  bool result = ireeCompilerLoadLibrary(iree_compiler_lib);
  // Restore stderr
  fflush(stderr);
  dup2(saved_stderr, fileno(stderr));
  close(saved_stderr);
  // Process result
  if (!result) {
    // Library may have already been loaded (can be safely ignored),
    // or may not be found (critical error), we cannot tell which from the return value.
    return 1;
  }
  // Must be balanced with a call to ireeCompilerGlobalShutdown()
  ireeCompilerGlobalInitialize();

  // To set global options (see `iree-compile --help` for possibilities), use
  // |ireeCompilerGetProcessCLArgs| and |ireeCompilerSetupGlobalCL|
  ireeCompilerGetProcessCLArgs(&cl_argc, &argv);
  ireeCompilerSetupGlobalCL(cl_argc, argv, "iree-jit", false);

  // Check the API version before proceeding any further
  uint32_t api_version = (uint32_t)ireeCompilerGetAPIVersion();
  uint16_t api_version_major = (uint16_t)((api_version >> 16) & 0xFFFFUL);
  uint16_t api_version_minor = (uint16_t)(api_version & 0xFFFFUL);
  DEBUG("Compiler API version: " << api_version_major << "." << api_version_minor);
  if (api_version_major > IREE_COMPILER_EXPECTED_API_MAJOR ||
      api_version_minor < IREE_COMPILER_EXPECTED_API_MINOR) {
    fprintf(stderr,
            "Error: incompatible API version; built for version %" PRIu16
            ".%" PRIu16 " but loaded version %" PRIu16 ".%" PRIu16 "\n",
            IREE_COMPILER_EXPECTED_API_MAJOR, IREE_COMPILER_EXPECTED_API_MINOR,
            api_version_major, api_version_minor);
    ireeCompilerGlobalShutdown();
    return 1;
  }

  // Check for a build tag with release version information
  const char *revision = ireeCompilerGetRevision();  // cppcheck-suppress unreadVariable
  DEBUG("Compiler revision: '" << revision << "'");
  return 0;
};

int IREESession::initCompiler() {

  // A session provides a scope where one or more invocations can be executed
  s.session = ireeCompilerSessionCreate();

  // Read the MLIR from memory
  error = ireeCompilerSourceWrapBuffer(
    s.session,
    "expr_buffer",  // name of the buffer (does not need to match MLIR)
    mlir_code.c_str(),
    mlir_code.length() + 1,
    true,
    &s.source
  );
  if (error) {
    fprintf(stderr, "Error wrapping source buffer\n");
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }
  DEBUG("Wrapped buffer as a compiler source");

  return 0;
};

int IREESession::initCompileToByteCode() {
  // Use an invocation to compile from the input source to the output stream
  iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(s.session);
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  if (!ireeCompilerInvocationParseSource(inv, s.source)) {
    fprintf(stderr, "Error parsing input source into invocation\n");
    cleanup_compiler_state(s);
    return 1;
  }

  // Compile, specifying the target dialect phase
  ireeCompilerInvocationSetCompileToPhase(inv, "end");

  // Run the compiler invocation pipeline
  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    fprintf(stderr, "Error running compiler invocation\n");
    cleanup_compiler_state(s);
    return 1;
  }
  DEBUG("Compilation successful");

  // Create compiler 'output' to a memory buffer
  error = ireeCompilerOutputOpenMembuffer(&s.output);
  if (error) {
    fprintf(stderr, "Error opening output membuffer\n");
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }

  // Create bytecode in memory
  error = ireeCompilerInvocationOutputVMBytecode(inv, s.output);
  if (error) {
    fprintf(stderr, "Error creating VM bytecode\n");
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }

  // Once the bytecode has been written, retrieve the memory map
  ireeCompilerOutputMapMemory(s.output, &contents, &size);

  return 0;
};

int IREESession::initRuntime() {
  // Setup the shared runtime instance
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance);

  // Create the HAL device used to run the workloads
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(
        iree_runtime_instance_driver_registry(instance),
        iree_make_cstring_view(device_uri),
        iree_runtime_instance_host_allocator(instance), &device);
  }

  // Set up the session to run the module
  if (iree_status_is_ok(status)) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  // Load the compiled user module from a file
  if (iree_status_is_ok(status)) {
    /*status = iree_runtime_session_append_bytecode_module_from_file(session, module_path);*/
    status = iree_runtime_session_append_bytecode_module_from_memory(
      session,
      iree_make_const_byte_span(contents, size),
      iree_allocator_null());
  }

  if (!iree_status_is_ok(status))
    return 1;

  return 0;
};

// Release the session and free all cached resources.
int IREESession::cleanup() {
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  int ret = (int)iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }
  cleanup_compiler_state(s);
  return ret;
}

iree_status_t IREESession::iree_runtime_exec(
  const std::string& function_name,
  const std::vector<std::vector<int>>& inputs,
  const std::vector<std::vector<float>>& data,
  std::vector<std::vector<float>>& result
) {

  // Initialize the call to the function.
  status = iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(function_name.c_str()), &call);
  if (!iree_status_is_ok(status)) {
    std::cerr << "Error: iree_runtime_call_initialize_by_name failed" << std::endl;
    iree_status_fprint(stderr, status);
    return status;
  }

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  host_allocator = iree_runtime_session_host_allocator(session);
  status = iree_ok_status();
  if (iree_status_is_ok(status)) {

    for(int k=0; k<inputs.size(); k++) {
      auto input_shape = inputs[k];
      const auto input_data = data[k];

      iree_hal_buffer_view_t* arg = NULL;
      if (iree_status_is_ok(status)) {
        std::vector<iree_hal_dim_t> arg_shape(input_shape.size());
        for (int i = 0; i < input_shape.size(); i++) {
          arg_shape[i] = input_shape[i];
        }
        int numel = 1;
        for(int i = 0; i < input_shape.size(); i++) {
          numel *= input_shape[i];
        }
        std::vector<float> arg_data(numel);
        for(int i = 0; i < numel; i++) {
          arg_data[i] = input_data[i];
        }

        status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator,
          // Shape rank and dimensions:
          arg_shape.size(), arg_shape.data(),
          // Element type:
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          // Encoding type:
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              // Intended usage of the buffer (transfers, dispatches, etc):
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
              // Access to allow to this memory:
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              // Where to allocate (host or device):
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          },
          // The actual heap buffer to wrap or clone and its allocator:
          iree_make_const_byte_span(&arg_data[0], sizeof(float) * arg_data.size()),
          // Buffer view + storage are returned and owned by the caller:
          &arg);
      }
      if (iree_status_is_ok(status)) {
        // Add to the call inputs list (which retains the buffer view).
        status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg);
        if (!iree_status_is_ok(status)) {
          std::cerr << "Error: iree_runtime_call_inputs_push_back_buffer_view failed" << std::endl;
          iree_status_fprint(stderr, status);
        }
      }
      // Since the call retains the buffer view we can release it here.
      iree_hal_buffer_view_release(arg);
    }
  }

  // Synchronously perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }
  if (!iree_status_is_ok(status)) {
    std::cerr << "Error: iree_runtime_call_invoke failed" << std::endl;
    iree_status_fprint(stderr, status);
  }

  for(int k=0; k<result.size(); k++) {
    // Dump the function outputs
    iree_hal_buffer_view_t* result_view = NULL;
    if (iree_status_is_ok(status)) {
      // Try to get the first call result as a buffer view.
      auto pop_status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result_view);
      if (!iree_status_is_ok(pop_status)) {
        std::cerr << "Error: iree_runtime_call_outputs_pop_front_buffer_view failed" << std::endl;
        iree_status_fprint(stderr, status);
      }
    }
    if (iree_status_is_ok(status)) {
      // Get the buffer view contents as a numeric array
      iree_host_size_t buffer_length = iree_hal_buffer_view_element_count(result_view);
      if (buffer_length != result[k].size()) {
        throw std::runtime_error(
          "Error: buffer_length (" + std::to_string(buffer_length) +
          ") != result[k].size()" + std::to_string(result[k].size())
        );
      }
      status = iree_hal_buffer_map_read(iree_hal_buffer_view_buffer(result_view), 0,
                               &result[k][0], sizeof(float) * result[k].size());
      if (!iree_status_is_ok(status)) {
        std::cerr << "Error: iree_hal_buffer_map_read failed" << std::endl;
        iree_status_fprint(stderr, status);
        iree_hal_buffer_view_release(result_view);
        return status;
      }
    }
    iree_hal_buffer_view_release(result_view);
  }

  iree_runtime_call_deinitialize(&call);

  return status;
}
