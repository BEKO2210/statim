// Replays fuzzer inputs without libFuzzer: `replay FILE_OR_DIR...` runs LLVMFuzzerTestOneInput on
// every file (directories are walked recursively). Used by ctest to keep crash regressions and
// the seed corpus green in ordinary (non-sanitizer) builds.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv);
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

int main(int argc, char** argv) {
    LLVMFuzzerInitialize(&argc, &argv);
    std::vector<std::filesystem::path> files;
    for (int i = 1; i < argc; ++i) {
        std::filesystem::path p(argv[i]);
        if (std::filesystem::is_directory(p)) {
            for (const auto& e : std::filesystem::recursive_directory_iterator(p))
                if (e.is_regular_file() && e.path().filename().string()[0] != '.') files.push_back(e.path());  // .gitkeep
        } else if (std::filesystem::exists(p)) {
            files.push_back(p);
        } else {
            std::fprintf(stderr, "replay: no such file or directory: %s\n", argv[i]);
            return 2;
        }
    }
    for (const auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        std::string buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        LLVMFuzzerTestOneInput(reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
    }
    std::printf("replayed %zu inputs\n", files.size());
    return 0;
}
