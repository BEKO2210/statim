# Included from the top-level CMakeLists.txt (inside STATIM_BUILD_TESTS).
#   * fuzz_replay_<name>: always built; ctest replays the seed corpus and every crash regression.
#   * fuzz_<name>: libFuzzer binaries, only with -DSTATIM_FUZZ=ON (clang).
set(STATIM_FUZZ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fuzz)
foreach(name request tokenizer gguf)
  add_executable(fuzz_replay_${name} ${STATIM_FUZZ_DIR}/fuzz_${name}.cpp ${STATIM_FUZZ_DIR}/replay_main.cpp)
  target_compile_definitions(fuzz_replay_${name} PRIVATE STATIM_FUZZ_DATA="${STATIM_FUZZ_DIR}/data"
                             STATIM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(fuzz_replay_${name} PRIVATE statim_core)
  add_test(NAME fuzz_regressions_${name}
           COMMAND fuzz_replay_${name} ${STATIM_FUZZ_DIR}/corpus/${name} ${STATIM_FUZZ_DIR}/regressions/${name})
  if(STATIM_FUZZ)
    add_executable(fuzz_${name} ${STATIM_FUZZ_DIR}/fuzz_${name}.cpp)
    target_compile_definitions(fuzz_${name} PRIVATE STATIM_FUZZ_DATA="${STATIM_FUZZ_DIR}/data"
                               STATIM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_options(fuzz_${name} PRIVATE -fsanitize=fuzzer)
    target_link_libraries(fuzz_${name} PRIVATE statim_core)
  endif()
endforeach()
