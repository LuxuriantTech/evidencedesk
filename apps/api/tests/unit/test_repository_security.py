from pathlib import Path

from scripts.check_supply_chain_refs import collect_violations

ROOT = Path(__file__).resolve().parents[4]


def test_compose_publishes_local_services_on_loopback_only() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    published_ports = [
        line.strip().strip('"')
        for line in compose.splitlines()
        if line.strip().startswith('- "') and line.strip().endswith('"')
    ]

    assert len(published_ports) == 4
    assert all(port.startswith('- "127.0.0.1:') for port in published_ports)


def test_repository_external_executable_references_are_immutable() -> None:
    assert collect_violations(ROOT) == []


def test_supply_chain_check_rejects_mutable_action_and_image_refs(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "ci.yml").write_text(
        "steps:\n  - uses: actions/checkout@v4\nservices:\n  redis:\n    image: redis:latest\n",
        encoding="utf-8",
    )

    violations = collect_violations(tmp_path)

    assert any("actions/checkout@v4" in violation for violation in violations)
    assert any("redis:latest" in violation for violation in violations)


def test_supply_chain_check_requires_forced_build_for_local_image_aliases(
    tmp_path: Path,
) -> None:
    (tmp_path / "compose.yaml").write_text(
        "services:\n"
        "  builder:\n"
        "    image: example-api:local\n"
        "    build: .\n"
        "    pull_policy: build\n"
        "  consumer:\n"
        "    image: example-api:local\n",
        encoding="utf-8",
    )

    violations = collect_violations(tmp_path)

    assert any(
        "example-api:local" in violation and "forced build" in violation
        for violation in violations
    )


def test_supply_chain_check_includes_local_8b_overlay_images(tmp_path: Path) -> None:
    overlay = tmp_path / "compose.local-8b.yaml"
    overlay.write_text(
        "services:\n  local-8b:\n    image: registry.example/model:latest\n",
        encoding="utf-8",
    )

    violations = collect_violations(tmp_path)

    assert any(
        "compose.local-8b.yaml" in item and "mutable external image" in item
        for item in violations
    )


def test_supply_chain_check_includes_local_generation_dockerfile_and_from_options(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "scripts" / "local_generation" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text(
        "FROM --platform=linux/amd64 ubuntu:24.04 AS runtime\n",
        encoding="utf-8",
    )

    violations = collect_violations(tmp_path)

    assert any("scripts/local_generation/Dockerfile" in item for item in violations)
    assert any("ubuntu:24.04" in item and "mutable external image" in item for item in violations)


def test_local_overlay_alias_is_allowed_only_with_forced_build(tmp_path: Path) -> None:
    base_compose = tmp_path / "compose.yaml"
    base_compose.write_text(
        "services:\n  api:\n    image: evidencedesk-api:local\n"
        "    build: .\n    pull_policy: build\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "compose.local-8b.yaml"
    overlay.write_text(
        "services:\n  local-8b:\n    image: evidencedesk-local-8b:opt-in\n"
        "    build: .\n    pull_policy: build\n",
        encoding="utf-8",
    )

    assert collect_violations(tmp_path) == []

    overlay.write_text(
        "services:\n  local-8b:\n    image: evidencedesk-local-8b:opt-in\n"
        "    build: .\n",
        encoding="utf-8",
    )
    violations = collect_violations(tmp_path)
    assert any(
        "evidencedesk-local-8b:opt-in" in item and "forced build" in item
        for item in violations
    )


def test_web_runtime_image_declares_a_non_root_user() -> None:
    dockerfile = (ROOT / "infra" / "docker" / "web.Dockerfile").read_text(
        encoding="utf-8"
    )
    user_directives = [
        line.split(maxsplit=1)[1].strip()
        for line in dockerfile.splitlines()
        if line.strip().upper().startswith("USER ")
    ]

    assert user_directives
    assert user_directives[-1].lower() not in {"0", "root"}


def test_web_runtime_image_prepares_a_writable_nginx_pid_file() -> None:
    dockerfile = (ROOT / "infra" / "docker" / "web.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "touch /run/nginx.pid" in dockerfile
    assert "chown nginx:nginx /run/nginx.pid" in dockerfile
