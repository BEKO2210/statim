#!/usr/bin/env bash
# Build the release binaries inside the manylinux_2_28 image (AlmaLinux 8, glibc 2.28, GCC 14), so
# they run on every distribution with glibc 2.28 or newer: RHEL/Alma/Rocky 8 and 9, Debian 10+,
# Ubuntu 20.04+ (READINESS P1 #55). A build on the CI runner's own Ubuntu 24.04 needed glibc 2.38.
#
# The compiler is GCC 13 (gcc-toolset-13), the same major version as Ubuntu 24.04's g++ that built
# every release so far. The image's default GCC 14 made the CPU engine about 5% slower on every
# perf-gate cell; GCC 13 measured within noise of the 0.9.5 release (bench/perf_gate.py --quick).
#
#   tools/release/build_manylinux.sh <cpu|vulkan> <build-dir> [extra cmake args...]
#
# Runs inside the container (see .github/workflows/release.yml). The vulkan flavor needs glslc,
# which AlmaLinux 8 does not package: it is built from shaderc at a pinned commit, the same
# release (v2023.8) as the Ubuntu 24.04 glslc that built 0.9.3, so ggml enables the same shader
# features. ggml also needs the SPIRV-Headers CMake package; EPEL's is from 2021, so the copy that
# shaderc pins is installed next to glslc. Set SHADERC_PREFIX to reuse an earlier build (CI caches it).
set -euo pipefail
flavor=${1:?usage: build_manylinux.sh <cpu|vulkan> <build-dir> [cmake args...]}
build=${2:?build dir}
shift 2
case "$flavor" in cpu) vulkan=OFF ;; vulkan) vulkan=ON ;; *) echo "unknown flavor $flavor" >&2; exit 2 ;; esac

SHADERC_COMMIT=e6edd6d48fa5bdd9d176794c6810fae7f8e938e1  # google/shaderc tag v2023.8
SHADERC_PREFIX=${SHADERC_PREFIX:-/opt/shaderc-v2023.8}

dnf -q install -y gcc-toolset-13-gcc-c++
# AlmaLinux 8's ninja-build (1.8) rejects ggml-vulkan's multi-output rules; ninja from PyPI, pinned
# by version and hash.
if ! /opt/ninja/bin/ninja --version >/dev/null 2>&1; then
  /opt/python/cp312-cp312/bin/python -m venv /opt/ninja
  echo "ninja==1.13.2 --hash=sha256:65a24341b5ac09fcadcc37082660be40a94174e51a937fabf6e2cae26225fa2c" > /tmp/ninja-req.txt
  /opt/ninja/bin/pip install -q --require-hashes --only-binary :all: -r /tmp/ninja-req.txt
fi
export PATH=/opt/ninja/bin:$PATH
ninja --version
export CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
"$CXX" --version | head -1
git config --global --add safe.directory '*'

extra=()
if [ "$vulkan" = ON ]; then
  dnf -q install -y vulkan-headers vulkan-loader-devel
  if [ ! -x "$SHADERC_PREFIX/bin/glslc" ] || [ ! -d "$SHADERC_PREFIX/share/cmake/SPIRV-Headers" ]; then
    src=$(mktemp -d)
    git -C "$src" init -q
    git -C "$src" fetch -q --depth 1 https://github.com/google/shaderc "$SHADERC_COMMIT"
    git -C "$src" checkout -q FETCH_HEAD
    (cd "$src" && python3 utils/git-sync-deps)  # glslang, SPIRV-Tools, SPIRV-Headers at shaderc's pinned commits
    cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SHADERC_PREFIX" \
      -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON -DSHADERC_SKIP_COPYRIGHT_CHECK=ON \
      -DSPIRV_SKIP_TESTS=ON -DSPIRV_SKIP_EXECUTABLES=ON -DENABLE_GLSLANG_BINARIES=OFF
    cmake --build "$src/build" --target glslc_exe
    install -D -m 755 "$src/build/glslc/glslc" "$SHADERC_PREFIX/bin/glslc"
    cmake -S "$src/third_party/spirv-headers" -B "$src/spirv-headers-build" -DCMAKE_INSTALL_PREFIX="$SHADERC_PREFIX"
    cmake --install "$src/spirv-headers-build"
    rm -rf "$src"
  fi
  "$SHADERC_PREFIX/bin/glslc" --version | head -1
  extra+=(-DVulkan_GLSLC_EXECUTABLE="$SHADERC_PREFIX/bin/glslc" -DCMAKE_PREFIX_PATH="$SHADERC_PREFIX")
  # ggml includes <spirv/unified1/spirv.hpp> by path (on Ubuntu it is in /usr/include) and does not
  # link the SPIRV-Headers target, so its include directory goes on the compiler's search path.
  export CPATH="$SHADERC_PREFIX/include${CPATH:+:$CPATH}"
fi

cmake -S . -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTATIM_NATIVE=OFF \
  -DSTATIM_VULKAN="$vulkan" \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
  "${extra[@]}" "$@"
cmake --build "$build" --target statim statim-quantize
