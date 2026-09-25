cmake_minimum_required(VERSION 3.22)
project(statim VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(STATIM_NATIVE "Optimise for the build machine's CPU" ON)
option(STATIM_BUILD_TESTS "Build tests" ON)

# ---- ggml (vendored)
set(GGML_NATIVE ${STATIM_NATIVE} CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ---- tokenizer
# Unicode data lives in src/unicode_tables.inc (generated from HF behavior by
# tools/gen_unicode_tables.py); no external Unicode library needed.
add_library(statim_tokenizer STATIC src/tokenizer.cpp)
target_include_directories(statim_tokenizer PUBLIC include PRIVATE third_party src)

# ---- core runtime
add_library(statim_core STATIC src/model.cpp src/engine.cpp src/tokenizer_gguf.cpp)
target_include_directories(statim_core PUBLIC include third_party PRIVATE third_party/ggml/src)
target_link_libraries(statim_core PUBLIC statim_tokenizer ggml)

if(STATIM_BUILD_TESTS)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tokenizer.cpp)
    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_include_directories(test_tokenizer PRIVATE third_party)
    target_compile_definitions(test_tokenizer PRIVATE STATIM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_tokenizer PRIVATE statim_tokenizer)
  endif()

  add_executable(test_model_parity tests/test_model_parity.cpp)
  target_link_libraries(test_model_parity PRIVATE statim_core)
  add_executable(test_engine_parity tests/test_engine_parity.cpp)
  target_link_libraries(test_engine_parity PRIVATE statim_core)
endif()

# ---- tools
add_executable(statim-quantize tools/quantize.cpp)
target_link_libraries(statim-quantize PRIVATE ggml)

# ---- statim binary (CLI + HTTP server)
find_package(Threads REQUIRED)
add_executable(statim src/main.cpp src/server.cpp)
target_compile_definitions(statim PRIVATE STATIM_VERSION="${PROJECT_VERSION}")
target_link_libraries(statim PRIVATE statim_core Threads::Threads)
install(TARGETS statim statim-quantize RUNTIME DESTINATION bin)
