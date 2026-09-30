// Shared helpers for the libFuzzer harnesses (fuzz_*.cpp) and their replay driver.
#pragma once

// A direct reproduction (build-fuzz/fuzz_<name> <file>) without fuzz/run.sh must also stop at the first
extern "C" const char* __ubsan_default_options() { return "halt_on_error=1:print_stacktrace=1"; }
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef STATIM_FUZZ_DATA
#define STATIM_FUZZ_DATA "fuzz/data"
#endif
#ifndef STATIM_SOURCE_DIR
#define STATIM_SOURCE_DIR "."
#endif

namespace statim_fuzz {
// Directory with the tiny GGUF models; $STATIM_FUZZ_DATA overrides the build-time path.
inline std::string data_dir() {
    const char* env = std::getenv("STATIM_FUZZ_DATA");
    return env && *env ? env : STATIM_FUZZ_DATA;
}
// An invariant violation: report and abort so libFuzzer records the input as a crash.
[[noreturn]] inline void violated(const char* what) {
    std::fprintf(stderr, "statim fuzz invariant violated: %s\n", what);
    std::abort();
}
}  // namespace statim_fuzz
