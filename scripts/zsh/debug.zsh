#!/usr/bin/zsh
set -euo pipefail

# ------------------------------------------------------------
# Debug builder for csalt++ tests
# - Builds Debug with CSALTXX_BUILD_TESTS=ON
# - Produces: build/debug/bin/unit_tests
# - Runs unit_tests (use --no-run to skip)
#
# Usage:
#   ./scripts/zsh/debug_cpp.zsh
#   ./scripts/zsh/debug_cpp.zsh --no-run
#   ./scripts/zsh/debug_cpp.zsh --clean
#   ./scripts/zsh/debug_cpp.zsh --generator "Unix Makefiles"
# ------------------------------------------------------------

# Defaults
BUILD_DIR="build/debug"
BUILD_TYPE="Debug"
RUN_AFTER_BUILD=1
GEN="${CMAKE_GENERATOR:-}"   # allow override from env

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-run) RUN_AFTER_BUILD=0; shift ;;
    --clean)  CLEAN=1; shift ;;
    --generator) GEN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--no-run] [--clean] [--generator NAME]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${(%):-%N}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJ_ROOT/csalt++"
OUT_DIR="$PROJ_ROOT/$BUILD_DIR"
BIN_DIR="$OUT_DIR/bin"
UNIT_EXE="$BIN_DIR/unit_tests"

# Jobs
if command -v nproc >/dev/null 2>&1; then JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then JOBS=$(sysctl -n hw.ncpu)
else JOBS=4; fi

# Prefer Ninja if no generator provided
if [[ -z "$GEN" ]]; then
  if command -v ninja >/dev/null 2>&1; then GEN="Ninja"; else GEN=""; fi
fi

echo "==> Source: $SRC_DIR"
echo "==> Build:  $OUT_DIR ($BUILD_TYPE)"
echo "==> Gen:    ${GEN:-<auto>}"
echo "==> Jobs:   $JOBS"

# Clean build dir if requested
if [[ "${CLEAN:-0}" -eq 1 ]]; then
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

# Configure for Debug tests (shared is fine here if you later add .cpp)
CMAKE_ARGS=(
  -S "$SRC_DIR"
  -B "$OUT_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DBUILD_SHARED_LIBS=ON
  -DCSALTXX_BUILD_TESTS=ON
)
[[ -n "$GEN" ]] && CMAKE_ARGS+=(-G "$GEN")

cmake "${CMAKE_ARGS[@]}"

# Build just the unit_tests target
cmake --build "$OUT_DIR" --config "$BUILD_TYPE" --target unit_tests -- -j "$JOBS"

# Verify location (CMakeLists sets RUNTIME_OUTPUT_DIRECTORY to bin/)
if [[ -x "$UNIT_EXE" ]]; then
  echo "==> Built: $UNIT_EXE"
else
  # Fallback search (in case a generator ignores RUNTIME_OUTPUT_DIRECTORY)
  FOUND=$(find "$OUT_DIR" -type f -name unit_tests -perm -u+x 2>/dev/null | head -n1 || true)
  if [[ -n "$FOUND" ]]; then
    UNIT_EXE="$FOUND"
    echo "==> Built: $UNIT_EXE"
  else
    echo "ERROR: unit_tests executable not found."
    exit 2
  fi
fi

# Optionally run it now
if [[ "$RUN_AFTER_BUILD" -eq 1 ]]; then
  echo "==> Running unit_tests"
  "$UNIT_EXE"
fi

echo "==> Done. You can run later via: $UNIT_EXE"

