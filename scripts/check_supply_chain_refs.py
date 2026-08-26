from __future__ import annotations

import re
from pathlib import Path

_ACTION_SHA = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _clean_ref(raw: str) -> str:
    return raw.split("#", 1)[0].strip().strip("'\"")


def _compose_local_images(compose_path: Path) -> tuple[set[str], set[str]]:
    if not compose_path.exists():
        return set(), set()
    text = compose_path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^  (?=[a-zA-Z0-9_-]+:\s*$)", text)
    built: set[str] = set()
    for block in blocks:
        image = re.search(r"(?m)^    image:\s*(\S+)", block)
        build = re.search(r"(?m)^    build:\s*(?:$|\S+)", block)
        if image and build:
            built.add(_clean_ref(image.group(1)))
    usage_forced: dict[str, list[bool]] = {ref: [] for ref in built}
    for block in blocks:
        image = re.search(r"(?m)^    image:\s*(\S+)", block)
        if image is None:
            continue
        ref = _clean_ref(image.group(1))
        if ref not in built:
            continue
        build = re.search(r"(?m)^    build:\s*(?:$|\S+)", block)
        pull_policy = re.search(r"(?m)^    pull_policy:\s*(\S+)", block)
        usage_forced[ref].append(
            build is not None
            and pull_policy is not None
            and _clean_ref(pull_policy.group(1)) == "build"
        )
    forced = {ref for ref, states in usage_forced.items() if states and all(states)}
    return built, forced


def _check_image_ref(
    path: Path,
    line_number: int,
    ref: str,
    local_images: set[str],
    forced_local_images: set[str],
) -> str | None:
    if ref in forced_local_images:
        return None
    if ref in local_images:
        return f"{path}:{line_number}: local image alias lacks forced build: {ref}"
    if "${" in ref or not _IMAGE_DIGEST.search(ref):
        return f"{path}:{line_number}: mutable external image reference: {ref}"
    return None


def collect_violations(root: Path) -> list[str]:
    root = root.resolve()
    violations: list[str] = []
    local_images, forced_local_images = _compose_local_images(root / "compose.yaml")
    yaml_paths = [root / "compose.yaml"]
    yaml_paths.extend(sorted((root / ".github" / "workflows").glob("*.y*ml")))

    for path in yaml_paths:
        if not path.exists():
            continue
        relative = path.relative_to(root)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            uses = re.search(r"\buses:\s*(\S+)", line)
            if uses:
                ref = _clean_ref(uses.group(1))
                if ref.startswith("./"):
                    pass
                elif ref.startswith("docker://"):
                    issue = _check_image_ref(
                        relative,
                        line_number,
                        ref.removeprefix("docker://"),
                        local_images,
                        forced_local_images,
                    )
                    if issue:
                        violations.append(issue)
                elif "@" not in ref or not _ACTION_SHA.fullmatch(ref.rsplit("@", 1)[1]):
                    violations.append(
                        f"{relative}:{line_number}: mutable GitHub Action reference: {ref}"
                    )

            image = re.match(r"\s*image:\s*(\S+)", line)
            if image:
                ref = _clean_ref(image.group(1))
                issue = _check_image_ref(
                    relative, line_number, ref, local_images, forced_local_images
                )
                if issue:
                    violations.append(issue)

    dockerfiles = sorted((root / "infra" / "docker").glob("*Dockerfile"))
    for path in dockerfiles:
        relative = path.relative_to(root)
        stage_names: set[str] = set()
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            match = re.match(r"FROM\s+\S+\s+AS\s+(\S+)", line, re.IGNORECASE)
            if match:
                stage_names.add(match.group(1))
        for line_number, line in enumerate(lines, start=1):
            syntax = re.match(r"#\s*syntax=(\S+)", line)
            if syntax:
                ref = _clean_ref(syntax.group(1))
                issue = _check_image_ref(
                    relative, line_number, ref, local_images, forced_local_images
                )
                if issue:
                    violations.append(issue)
            from_ref = re.match(r"FROM\s+(\S+)", line, re.IGNORECASE)
            if from_ref:
                ref = _clean_ref(from_ref.group(1))
                if ref not in stage_names:
                    issue = _check_image_ref(
                        relative, line_number, ref, local_images, forced_local_images
                    )
                    if issue:
                        violations.append(issue)
            copy_ref = re.search(r"COPY\s+--from=(\S+)", line, re.IGNORECASE)
            if copy_ref:
                ref = _clean_ref(copy_ref.group(1))
                if ref not in stage_names:
                    issue = _check_image_ref(
                        relative, line_number, ref, local_images, forced_local_images
                    )
                    if issue:
                        violations.append(issue)
    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = collect_violations(root)
    if violations:
        print("\n".join(violations))
        return 1
    print("Supply-chain executable references are pinned to immutable revisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
