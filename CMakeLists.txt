cmake_minimum_required(VERSION 3.22)
project(statim VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(STATIM_NATIVE "Optimise for the build machine's CPU" ON)
option(STATIM_BUILD_TESTS "Build tests" ON)
option(STATIM_VULKAN "Build the Vulkan GPU backend (--device vulkan)" OFF)

# ---- ggml (vendored)
set(GGML_NATIVE ${STATIM_NATIVE} CACHE BOOL "" FORCE)
if(NOT STATIM_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # portable x86-64-v3 baseline (Haswell and newer)
  foreach(opt GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${opt} ON CACHE BOOL "" FORCE)
  endforeach()
endif()
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${STATIM_VULKAN} CACHE BOOL "" FORCE)
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
add_library(statim_core STATIC src/model.cpp src/engine.cpp src/tokenizer_gguf.cpp src/kernels.cpp)
set_source_files_properties(src/kernels.cpp PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-fopenmp-simd;$<$<BOOL:${STATIM_NATIVE}>:-march=native>")
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

  enable_testing()
  set(M ${CMAKE_CURRENT_SOURCE_DIR}/models)
  set(D ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
  add_test(NAME tokenizer COMMAND test_tokenizer)
  add_test(NAME model_parity_multilingual COMMAND test_model_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_laya-multilingual.jsonl 1e-3)
  add_test(NAME engine_parity_multilingual COMMAND test_engine_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_inputs.json ${D}/golden_laya-multilingual.jsonl)
  add_test(NAME model_parity_english COMMAND test_model_parity ${M}/laya-english-f32.gguf ${D}/golden_laya.jsonl 1e-3)
  add_test(NAME engine_parity_english COMMAND test_engine_parity ${M}/laya-english-f32.gguf ${D}/golden_inputs.json ${D}/golden_laya.jsonl)
  if(STATIM_VULKAN)
    # same gates on the GPU (needs a Vulkan device at test time)
    add_test(NAME model_parity_multilingual_vulkan COMMAND test_model_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_laya-multilingual.jsonl 1e-3)
    add_test(NAME engine_parity_multilingual_vulkan COMMAND test_engine_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_inputs.json ${D}/golden_laya-multilingual.jsonl)
    add_test(NAME model_parity_english_vulkan COMMAND test_model_parity ${M}/laya-english-f32.gguf ${D}/golden_laya.jsonl 1e-3)
    add_test(NAME engine_parity_english_vulkan COMMAND test_engine_parity ${M}/laya-english-f32.gguf ${D}/golden_inputs.json ${D}/golden_laya.jsonl)
    set_tests_properties(model_parity_multilingual_vulkan engine_parity_multilingual_vulkan model_parity_english_vulkan
                         engine_parity_english_vulkan PROPERTIES ENVIRONMENT STATIM_DEVICE=vulkan)
  endif()
endif()

# ---- tools
add_executable(statim-quantize tools/quantize.cpp)
target_link_libraries(statim-quantize PRIVATE ggml)

# ---- statim binary (CLI + HTTP server)
find_package(Threads REQUIRED)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/playground.inc
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/src/playground.html -DDST=${CMAKE_CURRENT_BINARY_DIR}/playground.inc
          -P ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.playground.cmake
  DEPENDS src/playground.html CMakeLists.playground.cmake)
add_executable(statim src/main.cpp src/server.cpp ${CMAKE_CURRENT_BINARY_DIR}/playground.inc)
target_include_directories(statim PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(statim PRIVATE STATIM_VERSION="${PROJECT_VERSION}")
target_link_libraries(statim PRIVATE statim_core Threads::Threads)
install(TARGETS statim statim-quantize RUNTIME DESTINATION bin)
