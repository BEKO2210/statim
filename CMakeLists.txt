cmake_minimum_required(VERSION 3.22)
project(statim VERSION 0.9.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(STATIM_NATIVE "Optimise for the build machine's CPU" ON)
option(STATIM_BUILD_TESTS "Build tests" ON)
option(STATIM_VULKAN "Build the Vulkan GPU backend (--device vulkan)" OFF)
option(STATIM_CUDA "Build the CUDA GPU backend (--device cuda)" OFF)
option(STATIM_FUZZ "Build the libFuzzer harnesses (clang; instruments everything with ASan + UBSan)" OFF)
option(STATIM_SANITIZE "Build everything with ASan and UBSan" OFF)
option(STATIM_TSAN "Build everything with ThreadSanitizer" OFF)
option(STATIM_HARDEN "Hardening flags for release builds" ON)

if(STATIM_TSAN AND (STATIM_SANITIZE OR STATIM_FUZZ))
  message(FATAL_ERROR "STATIM_TSAN cannot be combined with STATIM_SANITIZE or STATIM_FUZZ")
endif()

if(STATIM_FUZZ OR STATIM_SANITIZE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(STATIM_FUZZ)
      message(FATAL_ERROR "STATIM_FUZZ needs clang (-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)")
    else()
      message(FATAL_ERROR "STATIM_SANITIZE needs clang (-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)")
    endif()
  endif()
  # ASan + UBSan for every object, ggml included, so memory errors our inputs cause inside ggml are
  # reported; coverage feedback only for Statim's parsers/runtime (below), which keeps
  # the kernels fast enough to run real forward passes per input.
  # -fno-sanitize=function: ggml calls typed converters through void* function pointers (upstream idiom).
  # UBSAN_OPTIONS=halt_on_error=1 makes every report abort; malformed GGUF is structurally checked
  # before it reaches ggml.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize=function -fno-sanitize-recover=undefined
                      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined -fno-sanitize=function -fno-sanitize-recover=undefined)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)  # libomp's thread pool is not ASan-instrumented
endif()

if(STATIM_TSAN)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "STATIM_TSAN needs clang (-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)")
  endif()
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g -O1)
  add_link_options(-fsanitize=thread)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)  # libomp's thread pool is not instrumented
endif()

if(STATIM_HARDEN AND CMAKE_SYSTEM_NAME MATCHES "Linux|Android" AND (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  include(CheckPIESupported)
  check_pie_supported(OUTPUT_VARIABLE STATIM_PIE_ERR LANGUAGES C CXX)
  if(CMAKE_C_LINK_PIE_SUPPORTED AND CMAKE_CXX_LINK_PIE_SUPPORTED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  else()
    message(WARNING "PIE is not supported: ${STATIM_PIE_ERR}")
  endif()

  add_compile_options(-fstack-protector-strong)
  add_link_options(-Wl,-z,relro,-z,now -Wl,-z,noexecstack)

  if(NOT (STATIM_FUZZ OR STATIM_SANITIZE OR STATIM_TSAN))
    include(CheckCSourceCompiles)
    set(_orig_req_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "-O2 -Werror -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3")
    check_c_source_compiles("
#include <stdio.h>
#include <string.h>
int main(void) {
    char buf[16];
    memcpy(buf, \"test\", 4);
    return 0;
}
" STATIM_HAS_FORTIFY_3)
    set(CMAKE_REQUIRED_FLAGS "${_orig_req_flags}")

    if(STATIM_HAS_FORTIFY_3)
      set(STATIM_FORTIFY_LEVEL 3)
    else()
      set(STATIM_FORTIFY_LEVEL 2)
    endif()

    add_compile_options($<$<NOT:$<CONFIG:Debug>>:-U_FORTIFY_SOURCE>
                        $<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=${STATIM_FORTIFY_LEVEL}>)
  endif()
endif()

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
set(GGML_CUDA ${STATIM_CUDA} CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

add_library(statim_gguf_preflight STATIC src/gguf_preflight.cpp)
target_include_directories(statim_gguf_preflight PUBLIC include)
target_link_libraries(statim_gguf_preflight PUBLIC ggml)

# ---- tokenizer
# Unicode data lives in src/unicode_tables.inc (generated from HF behavior by
# tools/gen_unicode_tables.py); no external Unicode library needed.
add_library(statim_tokenizer STATIC src/tokenizer.cpp)
target_include_directories(statim_tokenizer PUBLIC include PRIVATE third_party src)

# ---- core runtime
add_library(statim_core STATIC src/model.cpp src/engine.cpp src/security.cpp src/tokenizer_gguf.cpp src/kernels.cpp src/sha256.cpp)
set_source_files_properties(src/kernels.cpp PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-fopenmp-simd;$<$<BOOL:${STATIM_NATIVE}>:-march=native>")
target_include_directories(statim_core PUBLIC include third_party PRIVATE third_party/ggml/src)
target_link_libraries(statim_core PUBLIC statim_tokenizer ggml statim_gguf_preflight)
find_package(Threads REQUIRED)
if(STATIM_FUZZ)
  target_compile_options(statim_tokenizer PRIVATE -fsanitize=fuzzer-no-link)
  target_compile_options(statim_gguf_preflight PRIVATE -fsanitize=fuzzer-no-link)
  target_compile_options(statim_core PRIVATE -fsanitize=fuzzer-no-link)
endif()

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
  add_executable(test_batch_parity tests/test_batch_parity.cpp)
  target_link_libraries(test_batch_parity PRIVATE statim_core)

  add_executable(test_sgemm tests/test_sgemm.cpp)
  target_include_directories(test_sgemm PRIVATE src)
  target_link_libraries(test_sgemm PRIVATE statim_core Threads::Threads)

  add_executable(test_lora tests/test_lora.cpp)
  target_include_directories(test_lora PRIVATE src)
  target_link_libraries(test_lora PRIVATE statim_core)

  include(fuzz/CMakeLists.txt)

  add_executable(test_security tests/test_security.cpp)
  target_link_libraries(test_security PRIVATE statim_core)
  enable_testing()
  add_test(NAME security COMMAND test_security)
  add_executable(test_model_validation tests/test_model_validation.cpp)
  target_include_directories(test_model_validation PRIVATE third_party/ggml/src)
  target_link_libraries(test_model_validation PRIVATE statim_core)
  add_test(NAME model_validation COMMAND test_model_validation ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/data/tiny-metaspace.gguf)
  add_executable(test_cpu_check tests/test_cpu_check.cpp)
  target_include_directories(test_cpu_check PRIVATE include)
  target_compile_definitions(test_cpu_check PRIVATE STATIM_REQUIRE_X86_V3)
  add_test(NAME cpu_check COMMAND test_cpu_check)
  add_executable(test_gguf_preflight tests/test_gguf_preflight.cpp)
  target_link_libraries(test_gguf_preflight PRIVATE statim_core)
  add_test(NAME gguf_preflight COMMAND test_gguf_preflight ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(gguf_preflight PROPERTIES SKIP_RETURN_CODE 77)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME security_http COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/security/test_http.py
      --binary $<TARGET_FILE:statim> --model ${CMAKE_CURRENT_SOURCE_DIR}/models/laya-multilingual-f32.gguf)
    set_tests_properties(security_http PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    add_test(NAME api_contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_api_contract.py
      --binary $<TARGET_FILE:statim>
      --english ${CMAKE_CURRENT_SOURCE_DIR}/models/laya-english-f32.gguf
      --multilingual ${CMAKE_CURRENT_SOURCE_DIR}/models/laya-multilingual-f32.gguf)
    set_tests_properties(api_contract PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)
    add_test(NAME server_microbatch COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_server_microbatch.py
      --binary $<TARGET_FILE:statim> --model ${CMAKE_CURRENT_SOURCE_DIR}/models/laya-multilingual-f32.gguf
      --inputs ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden_inputs.json)
    set_tests_properties(server_microbatch PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_test(NAME server_lifecycle COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_server_lifecycle.py
      --binary $<TARGET_FILE:statim> --model ${CMAKE_CURRENT_SOURCE_DIR}/models/laya-multilingual-f32.gguf
      --inputs ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden_inputs.json)
    set_tests_properties(server_lifecycle PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1500)
    # LoRA adapters: PEFT fixtures (numpy only) -> tools/convert_lora.py -> engine and server gates
    set(LORA ${CMAKE_CURRENT_BINARY_DIR}/lora)
    set(LORA_BASE ${CMAKE_CURRENT_SOURCE_DIR}/models/laya-multilingual-f32.gguf)
    add_test(NAME lora_fixtures COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/lora/build_fixtures.py
      --base ${LORA_BASE} --out ${LORA})
    set_tests_properties(lora_fixtures PROPERTIES FIXTURES_SETUP lora TIMEOUT 120)
    add_test(NAME lora_convert COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/lora/test_convert_lora.py --base ${LORA_BASE}
      --binary $<TARGET_FILE:statim>)
    set_tests_properties(lora_convert PROPERTIES TIMEOUT 120)
    add_test(NAME lora_checkpoint_binding COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/lora/test_checkpoint_binding.py --binary $<TARGET_FILE:statim>
      --quantize $<TARGET_FILE:statim-quantize> --work ${CMAKE_CURRENT_BINARY_DIR}/lora-checkpoint-binding)
    set_tests_properties(lora_checkpoint_binding PROPERTIES TIMEOUT 120)
    add_test(NAME lora_parity COMMAND test_lora ${LORA_BASE} ${LORA}/zero.gguf ${LORA}/random.gguf
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden_inputs.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden_laya-multilingual.jsonl
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden_lora_random.jsonl ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/lora_random_weights.json)
    set_tests_properties(lora_parity PROPERTIES FIXTURES_REQUIRED lora TIMEOUT 600)
    # merge into quantized base weights: the base quantized with statim-quantize, q4_0 repacked on AVX2
    foreach(qt q8_0 q4_0)
      add_test(NAME lora_quantize_${qt} COMMAND statim-quantize ${LORA_BASE} ${LORA}/base-${qt}.gguf ${qt})
      set_tests_properties(lora_quantize_${qt} PROPERTIES FIXTURES_REQUIRED lora FIXTURES_SETUP lora_quantized TIMEOUT 120)
    endforeach()
    add_test(NAME lora_quantized COMMAND test_lora --quantized ${LORA_BASE} ${LORA}/random.gguf
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden_lora_random.jsonl ${LORA}/base-q8_0.gguf ${LORA}/base-q4_0.gguf)
    set_tests_properties(lora_quantized PROPERTIES FIXTURES_REQUIRED "lora;lora_quantized" TIMEOUT 600)
    add_test(NAME server_lora COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_server_lora.py
      --binary $<TARGET_FILE:statim> --model ${LORA_BASE} --adapters ${LORA})
    set_tests_properties(server_lora PROPERTIES FIXTURES_REQUIRED lora SKIP_RETURN_CODE 77 TIMEOUT 600)
    add_test(NAME server_concurrency COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_server_concurrency.py
      --binary $<TARGET_FILE:statim> --model ${LORA_BASE} --adapters ${LORA})
    set_tests_properties(server_concurrency PROPERTIES FIXTURES_REQUIRED lora SKIP_RETURN_CODE 77 TIMEOUT 600)
  endif()
  set(M ${CMAKE_CURRENT_SOURCE_DIR}/models)
  set(D ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
  add_test(NAME security_model COMMAND test_security ${M}/laya-multilingual-f32.gguf)
  add_test(NAME tokenizer COMMAND test_tokenizer)
  add_test(NAME sgemm COMMAND test_sgemm)
  set_tests_properties(sgemm PROPERTIES SKIP_RETURN_CODE 77)  # no AVX2/FMA (e.g. arm64)
  add_test(NAME model_parity_multilingual COMMAND test_model_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_laya-multilingual.jsonl 1e-3)
  add_test(NAME engine_parity_multilingual COMMAND test_engine_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_inputs.json ${D}/golden_laya-multilingual.jsonl)
  add_test(NAME batch_parity_multilingual COMMAND test_batch_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_inputs.json)
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
  if(STATIM_CUDA)
    # same gates on CUDA (needs an NVIDIA GPU at test time)
    add_test(NAME model_parity_multilingual_cuda COMMAND test_model_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_laya-multilingual.jsonl 1e-3)
    add_test(NAME engine_parity_multilingual_cuda COMMAND test_engine_parity ${M}/laya-multilingual-f32.gguf ${D}/golden_inputs.json ${D}/golden_laya-multilingual.jsonl)
    add_test(NAME model_parity_english_cuda COMMAND test_model_parity ${M}/laya-english-f32.gguf ${D}/golden_laya.jsonl 1e-3)
    add_test(NAME engine_parity_english_cuda COMMAND test_engine_parity ${M}/laya-english-f32.gguf ${D}/golden_inputs.json ${D}/golden_laya.jsonl)
    set_tests_properties(model_parity_multilingual_cuda engine_parity_multilingual_cuda model_parity_english_cuda
                         engine_parity_english_cuda PROPERTIES ENVIRONMENT STATIM_DEVICE=cuda)
  endif()
endif()

# ---- tools
add_executable(statim-quantize tools/quantize.cpp)
target_include_directories(statim-quantize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(statim-quantize PRIVATE ggml statim_gguf_preflight)
if(STATIM_BUILD_TESTS AND Python3_Interpreter_FOUND)
  add_test(NAME quantize_types COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quantize_types.py
    --binary $<TARGET_FILE:statim-quantize>
    --model ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/data/tiny-metaspace.gguf)
  set_tests_properties(quantize_types PROPERTIES TIMEOUT 60)
endif()

# ---- statim binary (CLI + HTTP server)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/playground.inc
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/src/playground.html -DDST=${CMAKE_CURRENT_BINARY_DIR}/playground.inc
          -P ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.playground.cmake
  DEPENDS src/playground.html CMakeLists.playground.cmake)
add_executable(statim src/main.cpp src/server.cpp ${CMAKE_CURRENT_BINARY_DIR}/playground.inc)
target_include_directories(statim PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(statim PRIVATE STATIM_VERSION="${PROJECT_VERSION}")
if(NOT STATIM_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  # the portable build assumes x86-64-v3 (see GGML_AVX2 above); check it before any vector code runs
  target_compile_definitions(statim PRIVATE STATIM_REQUIRE_X86_V3)
  target_compile_definitions(statim-quantize PRIVATE STATIM_REQUIRE_X86_V3)
endif()
target_link_libraries(statim PRIVATE statim_core Threads::Threads)
install(TARGETS statim statim-quantize RUNTIME DESTINATION bin)
