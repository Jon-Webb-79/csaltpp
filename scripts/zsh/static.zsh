#!/usr/bin/env zsh
set -euo pipefail

# ------------------------------------------------------------
# Install static csalt++ library (tests OFF)
# - Default prefix: /usr/local
# - Default build type: Release
# - Builds csaltpp as STATIC via BUILD_SHARED_LIBS=OFF
# - Elevates only for the install step if needed
#
# Usage:
#   ./scripts/zsh/static.zsh
#   ./scripts/zsh/static.zsh --prefix /opt/csaltpp
#   ./scripts/zsh/static.zsh --relwithdebinfo
#   ./scripts/zsh/static.zsh --generator "Unix Makefiles"
#   ./scripts/zsh/static.zsh --clean
# ------------------------------------------------------------

# Defaults
PREFIX="${CMAKE_INSTALL_PREFIX:-/usr/local}"
BUILD_DIR="build/static"
BUILD_TYPE="Release"
GEN="${CMAKE_GENERATOR:-}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)        PREFIX="$2"; shift 2 ;;
    --release)       BUILD_TYPE="Release"; shift ;;
    --relwithdebinfo|--rel) BUILD_TYPE="RelWithDebInfo"; shift ;;
    --debug)         BUILD_TYPE="Debug"; shift ;;  # allowed for dev static builds
    --generator)     GEN="$2"; shift 2 ;;
    --clean)         CLEAN=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--prefix DIR] [--release|--relwithdebinfo|--debug] [--generator NAME] [--clean]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${(%):-%N}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJ_ROOT/csalt++"
OUT_DIR="$PROJ_ROOT/$BUILD_DIR"

# Jobs
if command -v nproc >/dev/null 2>&1; then JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then JOBS=$(sysctl -n hw.ncpu)
else JOBS=4; fi

# Prefer Ninja if not specified
if [[ -z "$GEN" ]]; then
  if command -v ninja >/dev/null 2>&1; then GEN="Ninja"; else GEN=""; fi
fi

echo "==> Source: $SRC_DIR"
echo "==> Build:  $OUT_DIR ($BUILD_TYPE)"
echo "==> Prefix: $PREFIX"
echo "==> Gen:    ${GEN:-<auto>}"
echo "==> Jobs:   $JOBS"

# Clean build dir if requested
if [[ "${CLEAN:-0}" -eq 1 ]]; then rm -rf "$OUT_DIR"; fi
mkdir -p "$OUT_DIR"

# Configure: static-only, tests OFF
# NOTE:
# - If csalt++ is header-only, this will still install headers/inl and CMake package files.
# - When you add .cpp sources, BUILD_SHARED_LIBS=OFF will install libcsaltpp.a.
CMAKE_ARGS=(
  -S "$SRC_DIR"
  -B "$OUT_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DBUILD_SHARED_LIBS=OFF
  -DCSALTXX_BUILD_TESTS=OFF
  -DCSALTXX_BUILD_STATIC=OFF   # keep OFF to avoid duplicate target/name collisions
)
[[ -n "$GEN" ]] && CMAKE_ARGS+=(-G "$GEN")

cmake "${CMAKE_ARGS[@]}"

# Build
cmake --build "$OUT_DIR" --config "$BUILD_TYPE" -- -j "$JOBS"

# Install (elevate only if needed)
if [[ -w "$PREFIX" ]]; then
  cmake --install "$OUT_DIR" --config "$BUILD_TYPE"
else
  echo "==> Installing with sudo into $PREFIX"
  sudo cmake --install "$OUT_DIR" --config "$BUILD_TYPE"
fi

echo "==> Static install complete."
echo "    Headers: $PREFIX/include/csaltpp/*.hpp (and possibly *.h)"
echo "    SIMD:    $PREFIX/include/csaltpp/simd/*.inl"
echo "    CMake:   $PREFIX/lib/cmake/csaltpp (find_package(csaltpp))"
if [[ -f "$PREFIX/lib/libcsaltpp.a" ]]; then
  echo "    Library: $PREFIX/lib/libcsaltpp.a"
else
  echo "    Library: (none built; header-only install)"
fi

