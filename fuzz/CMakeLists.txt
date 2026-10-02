# Included from the top-level CMakeLists.txt (inside STATIM_BUILD_TESTS).
#   * fuzz_replay_<name>: always built; ctest replays the seeds and every crash regression.
#     The corpus the fuzzers grow is not committed; CI keeps it between runs (actions/cache).
#   * fuzz_<name>: libFuzzer binaries, only with -DSTATIM_FUZZ=ON (clang).
set(STATIM_FUZZ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fuzz)
foreach(name request tokenizer gguf)
  if(name STREQUAL "gguf" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    continue()  # fuzz_gguf.cpp loads models from memory through memfd_create, which only Linux has
  endif()
  if(name STREQUAL "gguf")
    set(seed_dir ${STATIM_FUZZ_DIR}/data)  # tiny models, including the q4_0 and q8_0 repack seeds
  else()
    set(seed_dir ${STATIM_FUZZ_DIR}/seeds/${name})
  endif()
  add_executable(fuzz_replay_${name} ${STATIM_FUZZ_DIR}/fuzz_${name}.cpp ${STATIM_FUZZ_DIR}/replay_main.cpp)
  statim_enable_coverage(fuzz_replay_${name})
  target_compile_definitions(fuzz_replay_${name} PRIVATE STATIM_FUZZ_DATA="${STATIM_FUZZ_DIR}/data"
                             STATIM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(fuzz_replay_${name} PRIVATE statim_core)
  add_test(NAME fuzz_regressions_${name}
           COMMAND fuzz_replay_${name} ${seed_dir} ${STATIM_FUZZ_DIR}/regressions/${name})
  if(STATIM_FUZZ)
    add_executable(fuzz_${name} ${STATIM_FUZZ_DIR}/fuzz_${name}.cpp)
    target_compile_definitions(fuzz_${name} PRIVATE STATIM_FUZZ_DATA="${STATIM_FUZZ_DIR}/data"
                               STATIM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_options(fuzz_${name} PRIVATE -fsanitize=fuzzer)
    target_link_libraries(fuzz_${name} PRIVATE statim_core)
  endif()
endforeach()
