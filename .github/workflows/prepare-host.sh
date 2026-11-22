#!/usr/bin/env bash
# Provision fresh GitHub-hosted runners. This script is called before licensing.
set -euo pipefail
case "${1:-}" in fast|licensed) mode=$1 ;; *) exit 2 ;; esac
shim=/tmp/madgraph-ci-tools
mkdir -p "$shim"
export PATH="$shim:$PATH"
elevate=()
if [[ $(id -u) != 0 ]]; then elevate=(sudo); fi

if [[ $(uname -s) == Darwin ]]; then
    # The macOS image supplies versioned binaries for its Homebrew GCC formulas.
    if ! command -v gfortran >/dev/null; then
        compiler=$(command -v gfortran-14 || command -v gfortran-15 || true)
        if [[ -z $compiler ]]; then
            brew install gcc@14
            compiler="$(brew --prefix gcc@14)/bin/gfortran-14"
        fi
        ln -sf "$compiler" "$shim/gfortran"
    fi
else
    "${elevate[@]}" apt-get -qq update
    "${elevate[@]}" apt-get install -y --no-install-recommends \
        gfortran g++ make bash perl ca-certificates curl python3-venv
    if grep -qi microsoft /proc/sys/kernel/osrelease; then
        # The Windows host's Python cannot build or run Linux MadGraph processes.
        python3 -m venv /opt/madgraph-ci/bootstrap
        /opt/madgraph-ci/bootstrap/bin/pip install --no-cache-dir uv==0.9.16
        export UV_PYTHON_INSTALL_DIR=/opt/madgraph-ci/python
        /opt/madgraph-ci/bootstrap/bin/uv venv --seed --python 3.11.14 /opt/madgraph-ci/venv
        export PATH="/opt/madgraph-ci/venv/bin:$PATH"
    fi
fi
python -m pip install --no-cache-dir 'numpy==1.26.4' 'scipy==1.13.1'
for tool in python gfortran c++ make bash perl; do command -v "$tool"; done
if [[ $mode == fast ]]; then exit 0; fi

if [[ $(uname -s) == Darwin ]]; then
    # A read-only mount avoids storing a second full copy of the Engine.
    # Published checksum from Homebrew's wolfram-engine 15.0.0.0 cask.
    image="$RUNNER_TEMP/WolframEngine.dmg"
    mountpoint="$RUNNER_TEMP/wolfram-engine"
    mkdir -p "$mountpoint"
    curl --fail --location --retry 3 --output "$image" \
        https://files.wolframcdn.com/packages/Homebrew/15.0.0.0/WolframEngine_15.0.0_MAC.dmg
    printf '%s  %s\n' ca3f0252b0df22d87d139dee1371a2a53be9784106b5974787dd024ef6ef959b "$image" | shasum -a 256 -c -
    hdiutil attach -readonly -nobrowse -mountpoint "$mountpoint" "$image"
    app="$mountpoint/Wolfram Engine.app/Contents"
    kernel="$app/MacOS/WolframKernel"
    if [[ ! -x $kernel ]]; then kernel="$app/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel"; fi
else
    # Release unused Android tooling only on the disposable native Ubuntu image.
    if [[ ${RUNNER_OS:-} == Linux ]] && [[ -d /usr/local/lib/android/sdk ]]; then
        "${elevate[@]}" rm -rf -- /usr/local/lib/android/sdk
    fi
    package=$(mktemp --suffix=.deb)
    curl --fail --location --retry 3 --output "$package" https://wolfr.am/wolfram-engine.deb
    version=$(dpkg-deb --field "$package" Version)
    case "$version" in
        15.0.0|15.0.0[-.+]*) ;;
        *) echo "Expected Wolfram Engine 15.0.0, received $version" >&2; exit 1 ;;
    esac
    chmod 644 "$package"
    "${elevate[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$package"
    rm -f -- "$package"
    "${elevate[@]}" apt-get clean
    kernel=/opt/Wolfram/WolframEngine/15.0/Executables/WolframKernel
    if [[ ! -x $kernel ]]; then kernel=/usr/local/Wolfram/WolframEngine/15.0/Executables/WolframKernel; fi
fi
test -x "$kernel"
ln -sf "$kernel" "$shim/WolframKernel"
df -h .
