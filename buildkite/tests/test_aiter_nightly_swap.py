import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "aiter-nightly-swap.sh"
BASH = shutil.which("bash")

STOCK = "rocm/vllm-dev:ci_base-build-42@sha256:" + "a" * 64
TAG = "rocm/vllm-dev:ci_base-aiter-nightly-build-42"
DIGEST = "sha256:" + "b" * 64
WHEEL = "https://example.com/amd_aiter-0.1.25+rocm7.2.3.bcb56d9.d20260928-cp312-cp312-linux_x86_64.whl"

# Meta-data is a file per key; docker calls are logged.
STUBS = {
    "buildkite-agent": r"""#!/bin/bash
set -euo pipefail
meta="$STATE/meta"
case "$1 ${2:-}" in
  "meta-data get")
    if [[ -f "$meta/$3" ]]; then cat "$meta/$3"
    elif [[ "${4:-}" == "--default" ]]; then printf '%s' "$5"
    else echo "no meta-data $3" >&2; exit 1; fi ;;
  "meta-data set") printf '%s' "$4" > "$meta/$3" ;;
  annotate*) style="$3"; cat > "$STATE/annotation.$style" ;;
  *) echo "unexpected: $*" >&2; exit 1 ;;
esac
""",
    "docker": r"""#!/bin/bash
set -euo pipefail
echo "$*" >> "$STATE/docker.log"
case "$1 ${2:-}" in
  "buildx build")
    cp "${@: -1}/Dockerfile" "$STATE/Dockerfile"
    exit "$(cat "$STATE/build.status")" ;;
  "buildx imagetools") echo "Name: x"; echo "Digest: $DIGEST" ;;
  "push "*|"pull "*) ;;
  "run "*)
    for command in select check; do
      if [[ " $* " == *" - $command"* ]]; then
        cat "$STATE/$command.out"
        cat "$STATE/$command.err" >&2
        exit "$(cat "$STATE/$command.status")"
      fi
    done
    echo "unexpected: $*" >&2; exit 1 ;;
  *) echo "unexpected: $*" >&2; exit 1 ;;
esac
""",
    "curl": r"""#!/bin/bash
while [[ $# -gt 0 ]]; do [[ "$1" == "-o" ]] && { echo "# script" > "$2"; shift; }; shift; done
""",
}


@pytest.fixture
def agent(tmp_path):
    if not BASH:
        pytest.skip("bash is not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in STUBS.items():
        stub = bin_dir / name
        stub.write_text(body, newline="\n")
        stub.chmod(0o755)
    (tmp_path / "meta").mkdir()

    class Agent:
        def set_meta(self, key, value):
            (tmp_path / "meta" / key).write_text(value)

        def meta(self, key):
            path = tmp_path / "meta" / key
            return path.read_text() if path.exists() else None

        def aiter_nightly(self, command, status=0, out="", err=""):
            """What `aiter_nightly.py <command>` does inside the image."""
            for suffix, value in (("status", status), ("out", out), ("err", err)):
                (tmp_path / f"{command}.{suffix}").write_text(str(value))

        def build_fails(self):
            (tmp_path / "build.status").write_text("1")

        def annotation(self, style):
            path = tmp_path / f"annotation.{style}"
            return path.read_text() if path.exists() else None

        def dockerfile(self):
            return (tmp_path / "Dockerfile").read_text()

        def run(self, **build_env):
            env = dict(
                os.environ,
                PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                STATE=str(tmp_path),
                DIGEST=DIGEST,
                BUILDKITE_BUILD_ID="42",
                VLLM_CI_BRANCH="my-branch",
            )
            env.pop("AITER_NIGHTLY_WHEEL_URL", None)
            env.update(build_env)
            (tmp_path / "docker.log").write_text("")
            result = subprocess.run(
                [BASH, str(SCRIPT)], env=env, capture_output=True, text=True
            )
            result.docker = (tmp_path / "docker.log").read_text()
            return result

    agent = Agent()
    agent.set_meta("rocm-ci-base-image", STOCK)
    agent.aiter_nightly("select", out=WHEEL + "\n")
    agent.aiter_nightly("check", out="AITER nightly: main at bcb56d9\n")
    (tmp_path / "build.status").write_text("0")
    return agent


def test_passing_run_builds_checks_and_records_its_own_tag(agent):
    result = agent.run()

    assert result.returncode == 0, result.stderr
    assert f"--build-arg CI_BASE_IMAGE={STOCK}" in result.docker
    assert f"--build-arg AITER_WHEEL_URL={WHEEL}" in result.docker
    assert f"push {TAG}" in result.docker
    assert agent.meta("aiter-nightly-wheel-url") == WHEEL
    assert agent.meta("aiter-nightly-ci-base") == f"{TAG}@{DIGEST}"
    assert agent.annotation("info") == "AITER nightly: main at bcb56d9\n"
    # image-build-amd moves the handoff, not this step.
    assert agent.meta("rocm-ci-base-image") == STOCK


def test_selects_and_checks_inside_the_images_they_concern(agent):
    docker = agent.run().docker
    select = next(line for line in docker.splitlines() if " - select" in line)
    check = next(line for line in docker.splitlines() if " - check" in line)
    assert STOCK in select
    # Pulled first, so its progress shows in the log.
    assert docker.index(f"pull {STOCK}") < docker.index(" - select")
    assert f"{TAG} -P - check" in check
    # The check runs before anything is pushed.
    assert docker.index(" - check") < docker.index("push ")


OLDER = "https://example.com/amd_aiter-0.1.24+rocm7.2.3.8253efc.d20260927-cp312-cp312-linux_x86_64.whl"


def _select_line(result):
    return next(line for line in result.docker.splitlines() if " - select" in line)


def test_first_attempt_selects_from_the_index(agent):
    assert "--wheel-url" not in _select_line(agent.run())


def test_a_wheel_set_on_the_build_is_installed_instead_of_the_newest(agent):
    agent.aiter_nightly("select", out=OLDER + "\n")
    result = agent.run(AITER_NIGHTLY_WHEEL_URL=OLDER)

    assert _select_line(result).endswith(f"- select --wheel-url {OLDER}")
    assert f"--build-arg AITER_WHEEL_URL={OLDER}" in result.docker
    assert agent.meta("aiter-nightly-wheel-url") == OLDER
    assert "Wheel set by AITER_NIGHTLY_WHEEL_URL" in agent.annotation("info")


def test_the_newest_nightly_is_not_labelled_as_chosen(agent):
    agent.run()
    assert "AITER_NIGHTLY_WHEEL_URL" not in agent.annotation("info")


def test_retry_reselects_the_first_attempts_wheel(agent):
    agent.set_meta("aiter-nightly-wheel-url", WHEEL)
    assert _select_line(agent.run()).endswith(f"- select --wheel-url {WHEEL}")


def test_installs_exactly_the_selected_wheel(agent):
    agent.run()
    assert (
        'RUN python3 /opt/aiter-nightly/aiter_nightly.py install --wheel-url "$AITER_WHEEL_URL"'
        in agent.dockerfile()
    )


@pytest.mark.parametrize(
    ("command", "status", "title"),
    [
        ("select", 10, "no wheel for this image"),
        ("select", 11, "the newest wheel is stale"),
        ("check", 13, "prebuilt modules do not load in vLLM's image"),
    ],
)
def test_each_failure_has_its_own_exit_code_and_annotation(
    agent, command, status, title
):
    agent.aiter_nightly(
        command,
        status=status,
        out="what went wrong\nmore detail\n",
        err="loading module_x.so\n",
    )

    result = agent.run()

    assert result.returncode == status
    annotation = agent.annotation("error")
    assert f"**AITER nightly: {title}**" in annotation
    assert "what went wrong\nmore detail" in annotation
    assert "push " not in result.docker
    assert agent.meta("aiter-nightly-ci-base") is None


@pytest.mark.parametrize(
    ("command", "status", "title"),
    [
        ("select", 1, "could not select a wheel"),
        ("check", 139, "the module check crashed"),
    ],
)
def test_a_crash_keeps_its_exit_code(agent, command, status, title):
    agent.aiter_nightly(command, status=status, err="Traceback\n")

    result = agent.run()

    assert result.returncode == status
    assert f"**AITER nightly: {title}**" in agent.annotation("error")
    assert "push " not in result.docker


def test_a_crash_is_annotated_with_the_last_output(agent):
    # A segfault prints no "AITER nightly:" line; the last module loaded is the clue.
    agent.aiter_nightly(
        "check", status=139, err="loading module_a.so\nloading module_b.so\n"
    )
    agent.run()
    assert "loading module_b.so" in agent.annotation("error")


def test_install_failure_has_its_own_exit_code_and_annotation(agent):
    agent.build_fails()

    result = agent.run()

    assert result.returncode == 12
    annotation = agent.annotation("error")
    assert "**AITER nightly: the wheel does not install**" in annotation
    assert WHEEL.rsplit("/", 1)[-1] in annotation
    assert "push " not in result.docker
