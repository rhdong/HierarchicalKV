#!/usr/bin/env bash
# Verify and copy the already-public, audited Company A statistics snapshot.
# This is not a sanitizer for new or original production logs.
# Usage: bash sanitize.sh <published-snapshot.log> <new-output.log>
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash sanitize.sh <published-snapshot.log> <new-output.log>" >&2
  exit 2
fi
python3 - "$1" "$2" <<'PYTHON'
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

EXPECTED_SHA256 = "379557f217500026b41c9d2a41e3c76ad984eb101ac47d4770e8d298ba881646"
source, destination = map(Path, sys.argv[1:])
temporary = None
try:
    if destination.exists() or destination.is_symlink():
        raise ValueError("output already exists; refusing to overwrite")
    if not destination.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    digest = hashlib.sha256()
    with source.open("rb") as input_file:
        if not stat.S_ISREG(os.fstat(input_file.fileno()).st_mode):
            raise ValueError("input must be a regular file")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".company-a-snapshot-", delete=False
        ) as output_file:
            temporary = Path(output_file.name)
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("input is not the audited public snapshot; refusing to publish")
    # link() atomically creates exactly this name and fails if it exists,
    # including a symlink or directory created after the initial check.
    os.link(temporary, destination, follow_symlinks=False)
except (OSError, ValueError) as error:
    print(f"FAIL: {error}", file=sys.stderr)
    sys.exit(1)
finally:
    if temporary is not None:
        temporary.unlink(missing_ok=True)
print(f"OK: verified public snapshot SHA256={EXPECTED_SHA256}; copied to {destination}")
PYTHON
