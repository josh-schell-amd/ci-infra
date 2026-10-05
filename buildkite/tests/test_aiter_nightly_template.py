import re
from pathlib import Path

import minijinja
import pytest
import yaml

BUILDKITE_DIR = Path(__file__).parents[1]
TEMPLATE = (BUILDKITE_DIR / "test-template-amd.j2").read_text()
BOOTSTRAP = (BUILDKITE_DIR / "bootstrap-amd.sh").read_text()
PRODUCTION = ["amdexperimental", "amdproduction"]
# test-amd.yaml tags every step with its GPU architecture.
ARCH_TAG = {
    "mi250": "amdgfx90anightly",
    "mi300": "amdgfx942nightly",
    "mi355": "amdgfx950nightly",
}


def _step(
    label, agent_pool="mi300_1", deps=(), commands=("pytest -v -s kernels",), **extra
):
    step = {
        "label": label,
        "agent_pool": agent_pool,
        "mirror_hardwares": PRODUCTION + [ARCH_TAG[agent_pool.split("_")[0]]],
        "dind": False,
        "source_file_dependencies": list(deps),
        "commands": list(commands),
    }
    step.update(extra)
    return step


STEPS = [
    _step(
        ":amd: (MI300) MoE Kernels Shard %N", deps=["vllm/_aiter_ops.py"], parallelism=5
    ),
    _step(":amd: (MI355) MoRI EP Numerics", agent_pool="mi355_2", optional=True),
    # Never mentions AITER: the nightly runs it anyway.
    _step(":amd: (MI300) Plain Kernels", deps=["vllm/model_executor"]),
    # AITER ships no kernels for gfx90a (MI250).
    _step(":amd: (MI250) Pipeline + Context Parallelism", agent_pool="mi250_4"),
    # No architecture tag: nothing says AITER builds for it.
    _step(":amd: (MI300) Untagged", mirror_hardwares=PRODUCTION),
    # amdexperimental only: not mirrored to the amdproduction nightly run.
    _step(
        ":amd: (MI300) Experimental Only",
        mirror_hardwares=["amdexperimental", "amdgfx942nightly"],
    ),
]


def _context(**overrides):
    context = {
        "branch": "main",
        "list_file_diff": "",
        "run_all": "0",
        "nightly": "0",
        "torch_nightly": "0",
        "mirror_hw": "amdproduction",
        "fail_fast": "false",
        "vllm_use_precompiled": "1",
        "vllm_merge_base_commit": "abc123",
        "cov_enabled": "0",
        "vllm_ci_branch": "my-branch",
        "rocm_base_refresh_skip": "0",
        "rocm_base_refresh_force": "0",
        "steps": STEPS,
    }
    context.update(overrides)
    return context


def _render(**overrides):
    rendered = minijinja.Environment().render_str(TEMPLATE, **_context(**overrides))
    return yaml.safe_load(rendered)[0]["steps"]


def _by_key(steps):
    return {step["key"]: step for step in steps if "key" in step}


def _test_labels(steps):
    return {step["label"] for step in steps if step.get("label", "").startswith("mi")}


@pytest.fixture
def nightly_steps():
    return _render(aiter_nightly="1")


def test_bootstrap_passes_every_variable_the_template_needs():
    passed = set(re.findall(r"^\s*-D (\w+)=", BOOTSTRAP, re.M))
    rendered_from_yaml = {"steps"}
    assert passed == set(_context(aiter_nightly="0")) - rendered_from_yaml


def test_bootstrap_aiter_nightly_defaults_off_and_disables_docs_only_skip():
    assert re.search(r'AITER_NIGHTLY:-\}" ]]; then\s+AITER_NIGHTLY=0', BOOTSTRAP)
    assert re.search(
        r'if \[\[ "\$\{AITER_NIGHTLY\}" == "1" \]\]; then\s+DOCS_ONLY_DISABLE=1',
        BOOTSTRAP,
    )


def test_flag_off_matches_flag_absent():
    assert _render(aiter_nightly="0") == _render()


def test_flag_off_has_no_nightly_step_or_overrides():
    steps = _render(aiter_nightly="0")
    by_key = _by_key(steps)
    assert "aiter-nightly-amd" not in by_key
    assert by_key["image-build-amd"]["depends_on"] == "ensure-ci-base-amd"
    assert (
        by_key["image-build-amd"]["commands"][0]
        == "bash .buildkite/scripts/rocm/build-test-image.sh"
    )
    assert (
        "rocm/vllm-dev:ci_base-build-$BUILDKITE_BUILD_ID"
        in by_key["verify-native-ci-base-amd"]["commands"][0]
    )


def test_nightly_installs_aiter_between_ci_base_and_test_image(nightly_steps):
    steps = _by_key(nightly_steps)
    swap = steps["aiter-nightly-amd"]
    assert swap["depends_on"] == "ensure-ci-base-amd"
    assert swap["agents"] == {"queue": "amd-cpu"}
    assert swap["env"]["VLLM_CI_BRANCH"] == "my-branch"
    assert (
        "/ci-infra/my-branch/buildkite/scripts/aiter-nightly-swap.sh"
        in swap["commands"][0]
    )
    assert steps["image-build-amd"]["depends_on"] == "aiter-nightly-amd"
    assert steps["verify-native-ci-base-amd"]["depends_on"] == "aiter-nightly-amd"


def test_nightly_native_jobs_use_the_tag_only_the_swap_writes(nightly_steps):
    tag = "rocm/vllm-dev:ci_base-aiter-nightly-build-$BUILDKITE_BUILD_ID"
    verify = _by_key(nightly_steps)["verify-native-ci-base-amd"]
    assert verify["commands"] == [f'docker manifest inspect "{tag}"']
    job = next(
        s for s in nightly_steps if s.get("label", "").endswith("MoE Kernels Shard %N")
    )
    assert job["env"]["VLLM_CI_BASE_IMAGE"] == tag
    assert "ci_base-build-$BUILDKITE_BUILD_ID" not in yaml.safe_dump(nightly_steps)


def test_nightly_test_image_points_the_handoff_at_the_nightly_first(nightly_steps):
    commands = _by_key(nightly_steps)["image-build-amd"]["commands"]
    # $$ defers the substitution from pipeline upload to the job.
    assert commands[0] == (
        "buildkite-agent meta-data set rocm-ci-base-image "
        '"$$(buildkite-agent meta-data get aiter-nightly-ci-base)"'
    )
    assert commands[1] == "bash .buildkite/scripts/rocm/build-test-image.sh"


def test_nightly_keeps_its_layers_out_of_main_caches(nightly_steps):
    env = _by_key(nightly_steps)["image-build-amd"]["env"]
    assert env["ROCM_CACHE_BRANCH_NAME"] == "aiter-nightly"
    assert env["ROCM_CONTENT_CACHE_EXPORT_MODE"] == "never"


def test_nightly_runs_every_production_step_on_an_aiter_gpu(nightly_steps):
    assert _test_labels(nightly_steps) == {
        "mi300_1: :amd: (MI300) MoE Kernels Shard %N",
        "mi355_2: :amd: (MI355) MoRI EP Numerics",
        "mi300_1: :amd: (MI300) Plain Kernels",
    }


def test_nightly_excludes_gpus_aiter_does_not_build_for(nightly_steps):
    labels = _test_labels(nightly_steps)
    assert not any("MI250" in label for label in labels)
    assert not any("Untagged" in label for label in labels)


def test_stock_pipeline_ignores_architecture_tags():
    labels = _test_labels(_render(aiter_nightly="0"))
    assert "mi250_4: :amd: (MI250) Pipeline + Context Parallelism" in labels
    assert "mi300_1: :amd: (MI300) Untagged" in labels


def test_nightly_respects_mirror_hardwares(nightly_steps):
    assert not any(
        "Experimental Only" in label for label in _test_labels(nightly_steps)
    )


def test_nightly_keeps_step_sharding(nightly_steps):
    job = next(
        s for s in nightly_steps if s.get("label", "").endswith("MoE Kernels Shard %N")
    )
    assert job["parallelism"] == 5


def test_nightly_steps_are_never_gated(nightly_steps):
    assert not [step for step in nightly_steps if "block" in step]
    optional = next(
        s for s in nightly_steps if s.get("label", "").endswith("MoRI EP Numerics")
    )
    assert optional["depends_on"] == ["image-build-amd", "verify-native-ci-base-amd"]
