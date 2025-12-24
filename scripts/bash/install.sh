#!/usr/bin/zsh
set -euo pipefail

# ------------------------------------------------------------
# Install csalt++ (header-only today; shared lib later when .cpp exist)
# - Default prefix: /usr/local
# - Default build type: Release
# - Uses Ninja if available (fallback to default generator)
# - Elevates only for the install step if needed
#
# Usage:
#   ./scripts/zsh/install.zsh
#   ./scripts/zsh/install.zsh --prefix /opt/csaltpp
#   ./scripts/zsh/install.zsh --relwithdebinfo
#   ./scripts/zsh/install.zsh --generator "Unix Makefiles"
#   ./scripts/zsh/install.zsh --clean
# ------------------------------------------------------------

# Defaults
PREFIX="${CMAKE_INSTALL_PREFIX:-/usr/local}"
BUILD_DIR="build/install"
BUILD_TYPE="Release"
GEN="${CMAKE_GENERATOR:-}"
RUN_LDCONFIG=1

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)        PREFIX="$2"; shift 2 ;;
    --release)       BUILD_TYPE="Release"; shift ;;
    --relwithdebinfo|--rel) BUILD_TYPE="RelWithDebInfo"; shift ;;
    --debug)         BUILD_TYPE="Debug"; shift ;;  # allowed, but install usually uses Release
    --generator)     GEN="$2"; shift 2 ;;
    --clean)         CLEAN=1; shift ;;
    --no-ldconfig)   RUN_LDCONFIG=0; shift ;;
    -h|--help)
      echo "Usage: $0 [--prefix DIR] [--release|--relwithdebinfo|--debug] [--generator NAME] [--clean] [--no-ldconfig]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Resolve paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${(%):-%N}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJ_ROOT/csalt++"
OUT_DIR="$PROJ_ROOT/$BUILD_DIR"

# Decide jobs
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

# Clean configure dir if requested
if [[ "${CLEAN:-0}" -eq 1 ]]; then rm -rf "$OUT_DIR"; fi
mkdir -p "$OUT_DIR"

# Configure: tests OFF, default to shared if/when sources exist
# (Header-only library ignores BUILD_SHARED_LIBS, but install still copies headers/inl.)
CMAKE_ARGS=(
  -S "$SRC_DIR"
  -B "$OUT_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DBUILD_SHARED_LIBS=ON
  -DCSALTXX_BUILD_TESTS=OFF
  -DCSALTXX_BUILD_STATIC=OFF
)
[[ -n "$GEN" ]] && CMAKE_ARGS+=(-G "$GEN")

cmake "${CMAKE_ARGS[@]}"

# Build everything needed for install/export
cmake --build "$OUT_DIR" --config "$BUILD_TYPE" -- -j "$JOBS"

# Install (elevate only if needed)
if [[ -w "$PREFIX" ]]; then
  cmake --install "$OUT_DIR" --config "$BUILD_TYPE"
else
  echo "==> Installing with sudo into $PREFIX"
  sudo cmake --install "$OUT_DIR" --config "$BUILD_TYPE"
fi

# Refresh linker cache on Linux for shared libs (only meaningful if a .so was installed)
if [[ "$RUN_LDCONFIG" -eq 1 && "$OSTYPE" == "linux-gnu"* ]]; then
  # Heuristic: if we installed a shared lib, ldconfig is useful. Header-only installs won't.
  if ls "$PREFIX/lib"/libcsaltpp.so* >/dev/null 2>&1; then
    if [[ $EUID -ne 0 ]]; then
      echo "==> Running ldconfig (sudo)"
      sudo ldconfig
    else
      ldconfig
    fi
  else
    echo "==> No libcsaltpp.so* detected under $PREFIX/lib; skipping ldconfig"
  fi
fi

echo "==> Install complete."
echo "    Headers: $PREFIX/include/csaltpp/*.hpp (and possibly *.h)"
echo "    SIMD:    $PREFIX/include/csaltpp/simd/*.inl"
echo "    CMake:   $PREFIX/lib/cmake/csaltpp (find_package(csaltpp))"
if ls "$PREFIX/lib"/libcsaltpp.so* >/dev/null 2>&1; then
  echo "    Library: $PREFIX/lib/libcsaltpp.so (plus soname symlinks)"
fi

