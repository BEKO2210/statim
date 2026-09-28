// SHA-256 (FIPS 180-4) for model fingerprints. Not for secrets: nothing here is constant-time.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace statim {

class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t n);
    std::array<uint8_t, 32> digest();  // finishes the hash; update() must not be called afterwards
    std::string hex();                 // digest() as 64 lowercase hex digits

private:
    void block(const uint8_t* p);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t n_ = 0;       // bytes in buf_
    uint64_t bytes_ = 0;  // total message length
};

}  // namespace statim
