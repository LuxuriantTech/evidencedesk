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
