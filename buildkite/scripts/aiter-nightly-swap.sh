#!/bin/bash
# Install the AITER nightly over this build's ci_base, under a tag only this
# step writes. Exit codes: aiter_nightly.py's EXIT_*.

set -euo pipefail

VLLM_CI_BRANCH="${VLLM_CI_BRANCH:?VLLM_CI_BRANCH is required}"
BUILD_TAG="rocm/vllm-dev:ci_base-aiter-nightly-build-${BUILDKITE_BUILD_ID:?BUILDKITE_BUILD_ID is required}"

context="$(mktemp -d)"
trap 'rm -rf "${context}"' EXIT
errors="${context}/errors"

fail() {
    local status="$1" title="$2" detail="$3"
    echo "+++ :x: AITER nightly: ${title}"
    echo "${detail}" >&2
    printf ':x: **AITER nightly: %s**\n\n```\n%s\n```\n' "${title}" "${detail}" \
        | buildkite-agent annotate --style error --context aiter-nightly
    exit "${status}"
}

ci_base="$(buildkite-agent meta-data get rocm-ci-base-image)"
# A retry keeps its first choice.
pinned="$(buildkite-agent meta-data get aiter-nightly-wheel-url --default "${AITER_NIGHTLY_WHEEL_URL:-}")"

curl -fsSL -o "${context}/aiter_nightly.py" \
    "https://raw.githubusercontent.com/vllm-project/ci-infra/${VLLM_CI_BRANCH}/buildkite/scripts/aiter_nightly.py"

# Here, not inside docker run, which would hide the pull's progress.
echo "--- :docker: Pulling ${ci_base}"
docker pull "${ci_base}"

# aiter_nightly.py runs inside the image, fed in on stdin. It prints the
# result, or for a known failure (exit 10-13) its message.
# -P: the image's working directory could shadow the installed aiter.
echo "--- :mag: Selecting the AITER nightly for ${ci_base}"
set +e
out="$(docker run --rm -i --entrypoint python3 "${ci_base}" \
    -P - select ${pinned:+--wheel-url "${pinned}"} \
    < "${context}/aiter_nightly.py" 2> "${errors}")"
status=$?
set -e
cat "${errors}" >&2
case "${status}" in
    0) ;;
    10) fail 10 "no wheel for this image" "${out}" ;;
    11) fail 11 "the newest wheel is stale" "${out}" ;;
    *) fail "${status}" "could not select a wheel" "$(tail -n 20 "${errors}")" ;;
esac
wheel_url="${out}"
buildkite-agent meta-data set aiter-nightly-wheel-url "${wheel_url}"

cat > "${context}/Dockerfile" <<'EOF'
ARG CI_BASE_IMAGE
FROM ${CI_BASE_IMAGE}
ARG AITER_WHEEL_URL
COPY aiter_nightly.py /opt/aiter-nightly/
RUN python3 /opt/aiter-nightly/aiter_nightly.py install --wheel-url "$AITER_WHEEL_URL"
EOF

echo "--- :docker: Installing ${wheel_url##*/}"
# The default builder keeps the image local for the check below.
if ! docker buildx build \
    --builder default \
    --provenance=false \
    --progress plain \
    --build-arg "CI_BASE_IMAGE=${ci_base}" \
    --build-arg "AITER_WHEEL_URL=${wheel_url}" \
    -t "${BUILD_TAG}" \
    "${context}"; then
    fail 12 "the wheel does not install" \
        "Installing ${wheel_url##*/} over ${ci_base} failed; see this step's log."
fi

echo "--- :hourglass_flowing_sand: Loading AITER's prebuilt modules in vLLM's image"
set +e
out="$(docker run --rm -i --entrypoint python3 "${BUILD_TAG}" \
    -P - check \
    < "${context}/aiter_nightly.py" 2> "${errors}")"
status=$?
set -e
cat "${errors}" >&2
case "${status}" in
    0) ;;
    13) fail 13 "prebuilt modules do not load in vLLM's image" "${out}" ;;
    # A segfault prints no message; the last module loaded is the clue.
    *) fail "${status}" "the module check crashed" "$(tail -n 20 "${errors}")" ;;
esac
note="${out}"

docker push "${BUILD_TAG}"
digest="$(docker buildx imagetools inspect "${BUILD_TAG}" | awk '$1 == "Digest:" { print $2; exit }')"
if [[ ! "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Could not resolve the pushed digest of ${BUILD_TAG}" >&2
    exit 1
fi
# image-build-amd points the ci_base handoff at this.
buildkite-agent meta-data set aiter-nightly-ci-base "${BUILD_TAG}@${digest}"
if [[ -n "${AITER_NIGHTLY_WHEEL_URL:-}" ]]; then
    note+=$'\n\n'"Wheel set by AITER_NIGHTLY_WHEEL_URL, not the newest nightly."
fi
echo "${note}" | buildkite-agent annotate --style info --context aiter-nightly
echo "--- :white_check_mark: ${BUILD_TAG}@${digest}"
