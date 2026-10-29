"""Script 33: verifica la trazabilidad Git del candidato congelado.

Comprueba que el snapshot científico y el tag apunten a los commits, trees y
archives registrados en FASE5_CANDIDATO_CHECKSUMS.json o en el mensaje del tag.
Es un verificador de solo lectura: no modifica archivos.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "auditoria" / "fase5_candidato_final" / "FASE5_CANDIDATO_CHECKSUMS.json"


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE
    ).strip()


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, stderr=subprocess.PIPE
    )


def archive_sha(ref: str) -> str:
    return hashlib.sha256(git_bytes("archive", "--format=tar", ref)).hexdigest()


def tag_field(message: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in message.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def verify_source(meta: dict, errors: list[str]) -> None:
    ref = meta["source_commit"]
    try:
        commit = git("rev-parse", f"{ref}^{{commit}}")
        tree = git("rev-parse", f"{ref}^{{tree}}")
    except subprocess.CalledProcessError:
        errors.append(f"Snapshot científico: no existe {ref}")
        return

    if commit != meta["source_commit"]:
        errors.append(f"Snapshot científico: commit {commit} != {meta['source_commit']}")
    if tree != meta["source_tree"]:
        errors.append(f"Snapshot científico: tree {tree} != {meta['source_tree']}")

    digest = archive_sha(ref)
    if digest != meta["source_archive_sha256"]:
        errors.append(
            f"Snapshot científico: archive {digest} != {meta['source_archive_sha256']}"
        )

    print("Snapshot científico:")
    print(f"  commit:  {commit}")
    print(f"  tree:    {tree}")
    print(f"  archive: {digest}")


def verify_tag(meta: dict, errors: list[str]) -> str:
    tag_name = meta.get("tag_name") or meta["tag_previsto"]
    if meta.get("tag_name") and meta["tag_name"] != meta["tag_previsto"]:
        errors.append(
            f"Tag: tag_name {meta['tag_name']} != tag_previsto {meta['tag_previsto']}"
        )

    try:
        commit = git("rev-parse", f"{tag_name}^{{commit}}")
        tree = git("rev-parse", f"{tag_name}^{{tree}}")
        message = git("tag", "-l", "--format=%(contents)", tag_name)
    except subprocess.CalledProcessError:
        errors.append(f"Tag {tag_name}: no existe")
        return tag_name

    digest = archive_sha(tag_name)
    expected_commit = meta.get("tag_commit") or tag_field(message, "tag_commit")
    expected_tree = meta.get("tag_tree") or tag_field(message, "tag_tree")
    expected_archive = meta.get("tag_archive_sha256") or tag_field(
        message, "tag_archive_sha256"
    )

    for key, expected in [
        ("tag_commit", expected_commit),
        ("tag_tree", expected_tree),
        ("tag_archive_sha256", expected_archive),
    ]:
        if not expected:
            errors.append(f"Tag {tag_name}: falta {key} en JSON y en el mensaje")

    if expected_commit and commit != expected_commit:
        errors.append(f"Tag {tag_name}: commit {commit} != {expected_commit}")
    if expected_tree and tree != expected_tree:
        errors.append(f"Tag {tag_name}: tree {tree} != {expected_tree}")
    if expected_archive and digest != expected_archive:
        errors.append(f"Tag {tag_name}: archive {digest} != {expected_archive}")

    for key in ("source_archive_sha256", "tag_archive_sha256"):
        expected = meta.get(key) or tag_field(message, key)
        if expected and expected not in message:
            errors.append(f"Tag {tag_name}: mensaje sin {key}")

    print(f"Tag {tag_name}:")
    print(f"  commit:  {commit}")
    print(f"  tree:    {tree}")
    print(f"  archive: {digest}")
    return tag_name


def main() -> int:
    if not JSON_PATH.exists():
        print("Falta FASE5_CANDIDATO_CHECKSUMS.json", file=sys.stderr)
        return 2

    meta = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    required = [
        "source_commit",
        "source_tree",
        "source_archive_sha256",
        "tag_previsto",
    ]
    missing = [key for key in required if key not in meta]
    if missing:
        print(f"Faltan campos de trazabilidad: {missing}", file=sys.stderr)
        return 2

    errors: list[str] = []
    verify_source(meta, errors)
    tag_name = verify_tag(meta, errors)

    if errors:
        print(f"\nERRORES DE TRAZABILIDAD: {len(errors)}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"\nOK: snapshot científico y {tag_name} coinciden con los hashes registrados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
