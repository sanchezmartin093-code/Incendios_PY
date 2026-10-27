"""Script 33: verifica la trazabilidad Git del candidato congelado.

Comprueba que el snapshot científico y el tag r2 apunten a los commits, trees
y archives registrados en FASE5_CANDIDATO_CHECKSUMS.json. Es un verificador
externo al corpus r2: no modifica archivos.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "auditoria" / "FASE5_CANDIDATO_CHECKSUMS.json"


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


def verify_ref(
    label: str,
    ref: str,
    expected_commit: str,
    expected_tree: str,
    expected_archive: str,
    errors: list[str],
) -> None:
    try:
        commit = git("rev-parse", f"{ref}^{{commit}}")
        tree = git("rev-parse", f"{ref}^{{tree}}")
    except subprocess.CalledProcessError:
        errors.append(f"{label}: no existe la referencia {ref}")
        return

    if commit != expected_commit:
        errors.append(f"{label}: commit {commit} != {expected_commit}")
    if tree != expected_tree:
        errors.append(f"{label}: tree {tree} != {expected_tree}")

    digest = archive_sha(ref)
    if digest != expected_archive:
        errors.append(f"{label}: archive {digest} != {expected_archive}")

    print(f"{label}:")
    print(f"  commit:  {commit}")
    print(f"  tree:    {tree}")
    print(f"  archive: {digest}")


def main() -> int:
    if not JSON_PATH.exists():
        print("Falta FASE5_CANDIDATO_CHECKSUMS.json", file=sys.stderr)
        return 2

    meta = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    required = [
        "source_commit",
        "source_tree",
        "source_archive_sha256",
        "tag_name",
        "tag_commit",
        "tag_tree",
        "tag_archive_sha256",
    ]
    missing = [key for key in required if key not in meta]
    if missing:
        print(f"Faltan campos de trazabilidad: {missing}", file=sys.stderr)
        return 2

    errors: list[str] = []
    verify_ref(
        "Snapshot científico",
        meta["source_commit"],
        meta["source_commit"],
        meta["source_tree"],
        meta["source_archive_sha256"],
        errors,
    )
    verify_ref(
        f"Tag {meta['tag_name']}",
        meta["tag_name"],
        meta["tag_commit"],
        meta["tag_tree"],
        meta["tag_archive_sha256"],
        errors,
    )

    try:
        message = git("tag", "-l", "--format=%(contents)", meta["tag_name"])
    except subprocess.CalledProcessError:
        message = ""
    for key in ("source_archive_sha256", "tag_archive_sha256"):
        if meta[key] not in message:
            errors.append(f"El mensaje del tag no registra {key}")

    if errors:
        print(f"\nERRORES DE TRAZABILIDAD: {len(errors)}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"\nOK: snapshot científico y {meta['tag_name']} coinciden con los hashes registrados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
