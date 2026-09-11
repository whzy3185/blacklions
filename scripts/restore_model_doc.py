# -*- coding: utf-8 -*-
"""Restore the exact current modeling Markdown from committed payload chunks."""
from pathlib import Path
import base64
import gzip
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "docs" / "_payload"
OUT = ROOT / "docs" / "B题_高上限建模与算法总方案.md"

EXPECTED_B64_SHA256 = "4f134faaa81ea5afb4926867f8de704607e4a966ac487598183728312bba82a1"
EXPECTED_GZ_SHA256 = "75fe3d4f2d0a1f6f75e922464a80e53c89d76fcb6f48fc12157f3c905b05d535"
EXPECTED_MD_SHA256 = "864789bb1d616e3d50507b83854554b789e19d86d5518a7ffcb5fe4f961a8589"
EXPECTED_MD_SIZE = 66772


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parts = sorted(PAYLOAD.glob("B题_高上限建模与算法总方案.md.gz.b64.part*"))
    if len(parts) != 3:
        raise RuntimeError(f"expected 3 payload parts, found {len(parts)}")

    b64 = "".join(p.read_text(encoding="utf-8").strip() for p in parts).encode("ascii")
    if sha256(b64) != EXPECTED_B64_SHA256:
        raise RuntimeError("base64 payload SHA-256 mismatch")

    gz = base64.b64decode(b64, validate=True)
    if sha256(gz) != EXPECTED_GZ_SHA256:
        raise RuntimeError("gzip payload SHA-256 mismatch")

    md = gzip.decompress(gz)
    if len(md) != EXPECTED_MD_SIZE:
        raise RuntimeError(f"Markdown size mismatch: {len(md)}")
    if sha256(md) != EXPECTED_MD_SHA256:
        raise RuntimeError("Markdown SHA-256 mismatch")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(md)
    print(f"restored: {OUT}")
    print(f"sha256:   {sha256(md)}")
    print(f"bytes:    {len(md)}")


if __name__ == "__main__":
    main()
