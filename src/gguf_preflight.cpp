#include "statim/gguf_preflight.h"

#include "statim/mapped_file.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace statim {
namespace {

constexpr uint64_t kMaxStringLength = 1024ULL * 1024 * 1024;
constexpr uint64_t kMaxArrayElements = 1024ULL * 1024 * 1024;

[[noreturn]] void reject(const std::string& field, const std::string& value, const std::string& limit) {
    throw std::runtime_error("GGUF preflight: " + field + " is " + value + ", limit " + limit);
}

class Reader {
public:
    // Reads the file's bytes in memory: the caller maps the file once (MappedFile) and hands the same
    // mapping to ggml afterwards, so what is checked here is exactly what gets parsed.
    Reader(const void* data, uint64_t size) : data_(static_cast<const char*>(data)), size_(size) {}

    uint64_t size() const { return size_; }
    uint64_t pos() const { return pos_; }
    uint64_t remaining() const { return size_ - pos_; }

    void bytes(void* dst, uint64_t n, const std::string& field) {
        if (n > remaining()) reject(field + " length", std::to_string(n), std::to_string(remaining()) + " remaining bytes");
        std::memcpy(dst, data_ + pos_, static_cast<size_t>(n));
        pos_ += n;
    }

    void skip(uint64_t n, const std::string& field) {
        if (n > remaining()) reject(field + " length", std::to_string(n), std::to_string(remaining()) + " remaining bytes");
        pos_ += n;
    }

    uint32_t u32(const std::string& field) {
        std::array<uint8_t, 4> b{};
        bytes(b.data(), b.size(), field);
        return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
    }

    uint64_t u64(const std::string& field) {
        std::array<uint8_t, 8> b{};
        bytes(b.data(), b.size(), field);
        uint64_t v = 0;
        for (unsigned i = 0; i < b.size(); ++i) v |= uint64_t(b[i]) << (8 * i);
        return v;
    }

    int64_t i64(const std::string& field) {
        const uint64_t bits = u64(field);
        int64_t value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string string(const std::string& field, uint64_t maximum = kMaxStringLength) {
        const uint64_t n = u64(field + " length");
        if (n > remaining()) reject(field + " length", std::to_string(n), std::to_string(remaining()) + " remaining bytes");
        if (n > maximum) reject(field + " length", std::to_string(n), std::to_string(maximum));
        if (n > std::numeric_limits<size_t>::max()) reject(field + " length", std::to_string(n), "SIZE_MAX");
        std::string value(static_cast<size_t>(n), '\0');
        bytes(value.data(), n, field);
        return value;
    }

    void skip_string(const std::string& field) {
        const uint64_t n = u64(field + " length");
        if (n > remaining()) reject(field + " length", std::to_string(n), std::to_string(remaining()) + " remaining bytes");
        if (n > kMaxStringLength) reject(field + " length", std::to_string(n), std::to_string(kMaxStringLength));
        skip(n, field);
    }

    // skip_string for element i of an array: the label is built only for an error message.
    void skip_string_at(const std::string& field, uint64_t i) {
        const uint64_t n = remaining() >= 8 ? u64(field) : u64(field + "[" + std::to_string(i) + "] length");
        if (n > remaining() || n > kMaxStringLength) {
            const std::string label = field + "[" + std::to_string(i) + "]";
            if (n > remaining()) reject(label + " length", std::to_string(n), std::to_string(remaining()) + " remaining bytes");
            reject(label + " length", std::to_string(n), std::to_string(kMaxStringLength));
        }
        skip(n, field);
    }

private:
    const char* data_;
    uint64_t size_;
    uint64_t pos_ = 0;
};

uint64_t checked_mul(uint64_t a, uint64_t b, const std::string& field) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        reject(field, std::to_string(a) + " * " + std::to_string(b), "UINT64_MAX");
    return a * b;
}

uint64_t checked_add(uint64_t a, uint64_t b, const std::string& field) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        reject(field, std::to_string(a) + " + " + std::to_string(b), "UINT64_MAX");
    return a + b;
}

uint64_t padded(uint64_t n, uint32_t alignment, const std::string& field) {
    const uint64_t extra = (alignment - n % alignment) % alignment;
    return checked_add(n, extra, field);
}

size_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

void valid_gguf_type(uint32_t type, const std::string& field, bool array_element) {
    if (type >= GGUF_TYPE_COUNT)
        reject(field, std::to_string(type), "less than GGUF_TYPE_COUNT (" + std::to_string(GGUF_TYPE_COUNT) + ")");
    if (array_element && type == GGUF_TYPE_ARRAY)
        reject(field, std::to_string(type), "nested arrays are not allowed");
}

void read_value(Reader& r, uint32_t type, const std::string& field, uint32_t* u32_value = nullptr) {
    if (type == GGUF_TYPE_STRING) {
        r.skip_string(field);
        return;
    }
    const size_t n = scalar_size(type);
    if (n == 0) reject(field + " type", std::to_string(type), "a scalar or string type");
    if (u32_value && type == GGUF_TYPE_UINT32) {
        *u32_value = r.u32(field);
    } else {
        r.skip(n, field);
    }
}

void read_array(Reader& r, uint32_t type, uint64_t count, const std::string& field) {
    if (type == GGUF_TYPE_STRING) {
        // Each string needs its own length word, so reject impossible counts before looping.
        const uint64_t header_bytes = checked_mul(count, uint64_t{8}, field + " string-header byte count");
        if (count > kMaxArrayElements)
            reject(field + " count", std::to_string(count), std::to_string(kMaxArrayElements));
        if (header_bytes > r.remaining())
            reject(field + " string-header byte count", std::to_string(header_bytes), std::to_string(r.remaining()) + " remaining bytes");
        for (uint64_t i = 0; i < count; ++i) r.skip_string_at(field, i);
        return;
    }
    const size_t element_size = scalar_size(type);
    if (element_size == 0) reject(field + " element type", std::to_string(type), "a scalar or string type");
    const uint64_t bytes = checked_mul(count, element_size, field + " byte count");
    if (count > kMaxArrayElements)
        reject(field + " count", std::to_string(count), std::to_string(kMaxArrayElements));
    if (bytes > r.remaining()) reject(field + " byte count", std::to_string(bytes), std::to_string(r.remaining()) + " remaining bytes");
    r.skip(bytes, field);
}

}  // namespace

void gguf_preflight(const std::string& path) {
    std::unique_ptr<MappedFile> file;
    try {
        file = std::make_unique<MappedFile>(path);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("GGUF preflight: ") + e.what());
    }
    gguf_preflight(file->data(), file->size());
}

void gguf_preflight(const void* data, size_t size) {
    Reader r(data, size);
    std::array<char, 4> magic{};
    r.bytes(magic.data(), magic.size(), "magic");
    if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'})
        reject("magic", std::string(magic.data(), magic.size()), "GGUF");

    const uint32_t version = r.u32("version");
    if (version != 2 && version != 3) reject("version", std::to_string(version), "2 or 3");
    const int64_t n_tensors = r.i64("n_tensors");
    const int64_t n_kv = r.i64("n_kv");
    if (n_tensors < 0) reject("n_tensors", std::to_string(n_tensors), "non-negative");
    if (n_kv < 0) reject("n_kv", std::to_string(n_kv), "non-negative");
    // These coarse limits prevent attacker-controlled loop counts even before individual reads.
    if (static_cast<uint64_t>(n_tensors) > r.remaining() / 32)
        reject("n_tensors", std::to_string(n_tensors), std::to_string(r.remaining() / 32) + " minimum-size entries");
    if (static_cast<uint64_t>(n_kv) > r.remaining() / 12)
        reject("n_kv", std::to_string(n_kv), std::to_string(r.remaining() / 12) + " minimum-size entries");

    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    std::unordered_set<std::string> keys;
    for (int64_t i = 0; i < n_kv; ++i) {
        const std::string label = "key/value[" + std::to_string(i) + "]";
        const std::string key = r.string(label + " key");
        if (key.empty()) reject(label + " key length", "0", "at least 1");
        if (!keys.insert(key).second) reject(label + " key", key, "unique keys");
        const uint32_t stored_type = r.u32("key '" + key + "' type");
        valid_gguf_type(stored_type, "key '" + key + "' type", false);
        if (key == GGUF_KEY_GENERAL_ALIGNMENT && stored_type != GGUF_TYPE_UINT32)
            reject("general.alignment type", std::to_string(stored_type), "GGUF_TYPE_UINT32 (4)");
        if (stored_type == GGUF_TYPE_ARRAY) {
            const uint32_t element_type = r.u32("key '" + key + "' array element type");
            valid_gguf_type(element_type, "key '" + key + "' array element type", true);
            const uint64_t count = r.u64("key '" + key + "' array count");
            read_array(r, element_type, count, "key '" + key + "' array");
        } else {
            uint32_t value = 0;
            read_value(r, stored_type, "key '" + key + "' value", key == GGUF_KEY_GENERAL_ALIGNMENT ? &value : nullptr);
            if (key == GGUF_KEY_GENERAL_ALIGNMENT) alignment = value;
        }
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        reject("general.alignment", std::to_string(alignment), "a non-zero power of two");

    struct TensorInfo { std::string name; uint64_t offset; uint64_t nbytes; };
    std::vector<TensorInfo> tensors;
    std::unordered_set<std::string> names;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const std::string label = "tensor[" + std::to_string(i) + "]";
        const std::string name = r.string(label + " name", GGML_MAX_NAME - 1);
        if (name.size() >= GGML_MAX_NAME)
            reject(label + " name length", std::to_string(name.size()), "less than GGML_MAX_NAME (" + std::to_string(GGML_MAX_NAME) + ")");
        // ggml copies names up to the first NUL, so an embedded NUL would make two names equal there
        if (name.find('\0') != std::string::npos) reject(label + " name", "a name with an embedded NUL", "no NUL bytes");
        if (!names.insert(name).second) reject(label + " name", name, "unique tensor names");
        const uint32_t n_dims = r.u32("tensor '" + name + "' n_dims");
        if (n_dims < 1 || n_dims > GGML_MAX_DIMS)
            reject("tensor '" + name + "' n_dims", std::to_string(n_dims), "1..GGML_MAX_DIMS (" + std::to_string(GGML_MAX_DIMS) + ")");
        std::array<uint64_t, GGML_MAX_DIMS> ne{1, 1, 1, 1};
        uint64_t elements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            const int64_t extent = r.i64("tensor '" + name + "' ne[" + std::to_string(d) + "]");
            if (extent < 0) reject("tensor '" + name + "' ne[" + std::to_string(d) + "]", std::to_string(extent), "non-negative");
            ne[d] = static_cast<uint64_t>(extent);
            elements = checked_mul(elements, ne[d], "tensor '" + name + "' element count");
            if (elements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                reject("tensor '" + name + "' element count", std::to_string(elements), "INT64_MAX");
        }
        const uint32_t raw_type = r.u32("tensor '" + name + "' type");
        if (raw_type >= GGML_TYPE_COUNT)
            reject("tensor '" + name + "' type", std::to_string(raw_type), "less than GGML_TYPE_COUNT (" + std::to_string(GGML_TYPE_COUNT) + ")");
        // Only form the enum after its representation is known to be valid (U2).
        const ggml_type type = static_cast<ggml_type>(raw_type);
        const ggml_type_traits* traits = ggml_get_type_traits(type);
        if (!traits || traits->blck_size <= 0)
            reject("tensor '" + name + "' block size", traits ? std::to_string(traits->blck_size) : "missing", "greater than 0");
        const uint64_t block = static_cast<uint64_t>(traits->blck_size);
        if (ne[0] % block != 0)
            reject("tensor '" + name + "' ne[0]", std::to_string(ne[0]), "a multiple of block size " + std::to_string(block));
        // ggml_nbytes and ggml_row_size multiply type_size by ne[0] before dividing by the block size
        if (ne[0] > 0 && traits->type_size > std::numeric_limits<uint64_t>::max() / ne[0])
            reject("tensor '" + name + "' ne[0] * type size", std::to_string(ne[0]) + " * " + std::to_string(traits->type_size), "UINT64_MAX");
        const uint64_t blocks = elements / block;
        const uint64_t nbytes = checked_mul(blocks, traits->type_size, "tensor '" + name + "' byte size");
        const uint64_t offset = r.u64("tensor '" + name + "' offset");
        if (offset % alignment != 0)
            reject("tensor '" + name + "' offset", std::to_string(offset), "aligned to " + std::to_string(alignment));
        tensors.push_back({name, offset, nbytes});
    }

    const uint64_t data_start = tensors.empty() ? r.pos() : padded(r.pos(), alignment, "data section start");
    if (data_start > r.size()) reject("data section start", std::to_string(data_start), std::to_string(r.size()) + " file bytes");
    const uint64_t data_bytes = r.size() - data_start;
    uint64_t expected_offset = 0;
    for (const TensorInfo& tensor : tensors) {
        if (tensor.offset != expected_offset)
            reject("tensor '" + tensor.name + "' offset", std::to_string(tensor.offset), "expected " + std::to_string(expected_offset));
        const uint64_t end = checked_add(tensor.offset, tensor.nbytes, "tensor '" + tensor.name + "' data end");
        if (end > data_bytes)
            reject("tensor '" + tensor.name + "' data end", std::to_string(end), std::to_string(data_bytes) + " data-section bytes");
        expected_offset = checked_add(expected_offset, padded(tensor.nbytes, alignment, "tensor '" + tensor.name + "' padded size"),
                                      "GGUF data layout size");
    }
    // ggml reads the padded layout, the last tensor's padding included, when it loads the data
    if (expected_offset > data_bytes)
        reject("padded data section", std::to_string(expected_offset), std::to_string(data_bytes) + " data-section bytes");
}

}  // namespace statim
