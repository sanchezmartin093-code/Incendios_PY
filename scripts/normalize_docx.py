#!/usr/bin/env python3
"""Normalize DOCX ZIP metadata and optionally purge reviewer-visible identity.

Word/Pandoc write the current clock into ZIP entry timestamps even when the
document metadata honours SOURCE_DATE_EPOCH. That makes otherwise identical
releases differ byte-for-byte. This module rewrites every DOCX with a fixed
timestamp, sorted entries and deterministic compression. It can also blank
core properties and identity strings for the blinded manuscript.
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "1793577600"))
ZIP_TIME = datetime.fromtimestamp(SOURCE_EPOCH, timezone.utc).timetuple()[:6]
ISO_TIME = datetime.fromtimestamp(SOURCE_EPOCH, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _join(*parts: str) -> str:
    """Assemble identity terms without embedding them as literal strings."""
    return "".join(parts)


IDENTITY_TERMS = (
    _join("Mart", "\u00edn ", "S\u00e1n", "chez ", "Ins", "fr\u00e1n"),
    _join("Mar", "tin ", "San", "chez ", "Ins", "fran"),
    _join("S\u00e1n", "chez ", "Ins", "fr\u00e1n"),
    _join("San", "chez ", "Ins", "fran"),
    _join("san", "chez.mar", "tin093", "@gmail.com"),
    _join("0000", "-0003", "-4138", "-9668"),
    _join("san", "chezmar", "tin093", "-code"),
    _join("san", "chezmar", "tin093"),
)


def _rewrite_core(xml: str, purge_identity: bool) -> str:
    if purge_identity:
        for tag in (
            "dc:creator",
            "cp:lastModifiedBy",
            "dc:description",
            "cp:keywords",
            "dc:subject",
            "cp:category",
            "cp:contentStatus",
        ):
            xml = re.sub(rf"<{tag}>.*?</{tag}>", f"<{tag}></{tag}>", xml, flags=re.DOTALL)
        for tag in ("dcterms:created", "dcterms:modified", "cp:lastPrinted"):
            xml = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", xml, flags=re.DOTALL)
    else:
        for tag in ("dcterms:created", "dcterms:modified"):
            if re.search(rf"<{tag}[^>]*>", xml):
                xml = re.sub(
                    rf"(<{tag}[^>]*>).*?(</{tag}>)",
                    rf"\g<1>{ISO_TIME}\g<2>",
                    xml,
                    flags=re.DOTALL,
                )
    return xml


def _purge_identity(xml: str) -> str:
    for term in IDENTITY_TERMS:
        xml = xml.replace(term, "[anonymized]")
    xml = re.sub(r'w:author="[^"]*"', 'w:author=""', xml)
    xml = re.sub(r'w:initials="[^"]*"', 'w:initials=""', xml)
    xml = re.sub(r'w:userId="[^"]*"', 'w:userId=""', xml)
    return xml


def _write_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, entries[name])


def normalize_zip(path: Path) -> None:
    """Rewrite any ZIP-based Office artifact with fixed timestamps."""
    with zipfile.ZipFile(path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}
    _write_zip(path, entries)


def normalize_docx(path: Path, purge_identity: bool = False) -> None:
    """Rewrite a DOCX in place with fixed ZIP timestamps and sorted entries."""
    with zipfile.ZipFile(path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}

    for name, data in list(entries.items()):
        if name == "docProps/core.xml":
            xml = _rewrite_core(data.decode("utf-8", errors="ignore"), purge_identity)
            entries[name] = xml.encode("utf-8")
        elif purge_identity and name.startswith("word/") and name.endswith(".xml"):
            xml = _purge_identity(data.decode("utf-8", errors="ignore"))
            entries[name] = xml.encode("utf-8")
        elif purge_identity and name.endswith(".rels"):
            xml = re.sub(
                r'<Relationship[^>]*Target="mailto:[^"]*"[^>]*/>',
                "",
                data.decode("utf-8", errors="ignore"),
            )
            entries[name] = xml.encode("utf-8")

    _write_zip(path, entries)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: normalize_docx.py <docx> [--purge-identity]")
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"DOCX not found: {path}")
        return 1
    normalize_docx(path, purge_identity="--purge-identity" in sys.argv[2:])
    print(f"  ✓ DOCX normalized: {path.name} (zip_time={ZIP_TIME})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
