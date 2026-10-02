#pragma once
// A GGUF file opened once and mapped read-only. The structural preflight, ggml's parser and the
// zero-copy tensors all read this one mapping, so a file swapped or replaced at the same path
// between those steps cannot slip past the preflight (open-once instead of open-by-name per step).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace statim {

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        // O_NONBLOCK: opening a FIFO without a writer would block before the regular-file check below
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) throw std::runtime_error("cannot open '" + path + "': " + std::strerror(errno));
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            const int err = errno;
            ::close(fd);
            throw std::runtime_error("cannot stat '" + path + "': " + std::strerror(err));
        }
        // a FIFO or device would block or change under the reader; a model is a regular file
        if (!S_ISREG(st.st_mode)) {
            ::close(fd);
            throw std::runtime_error("'" + path + "' is not a regular file");
        }
        if (st.st_size <= 0) {
            ::close(fd);
            throw std::runtime_error("'" + path + "' is empty");
        }
        size_ = static_cast<size_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        const int err = errno;
        ::close(fd);  // the mapping keeps the file alive; the path is never opened again
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed for '" + path + "': " + std::strerror(err));
        data_ = p;
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() {
        if (data_) ::munmap(data_, size_);
    }

    const void* data() const { return data_; }
    size_t size() const { return size_; }

    // Hands the mapping to an owner that unmaps it later (the model keeps its weights mapped).
    std::pair<void*, size_t> release() {
        auto out = std::make_pair(data_, size_);
        data_ = nullptr;
        size_ = 0;
        return out;
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace statim
