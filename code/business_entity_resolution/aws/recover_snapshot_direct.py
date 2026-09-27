#!/usr/bin/env python3
"""Extract selected files from an encrypted EBS snapshot using AWS CLI only.

The snapshot is exposed as a lazy, read-only block device to the Dissect XFS
parser. Only filesystem metadata and the requested files are downloaded; this
does not restore an EC2 instance, use S3, or materialize the 200-GiB disk image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

BLOCK_SIZE = 512 * 1024
DEFAULT_FILES = (
    "/var/log/devcore-v2-split.log",
    "/opt/devcore-v2/results/feature-shards/001/feature-shard-metrics.json",
    "/opt/devcore-v2/results/feature-shards/001/train_features-shard-001-of-006.parquet",
    "/opt/devcore-v2/results/feature-shards/001/train_targets-shard-001-of-006.parquet",
    "/opt/devcore-v2/results/feature-shard-metrics.json",
    "/opt/devcore-v2/results/train_features-shard-001-of-006.parquet",
    "/opt/devcore-v2/results/train_targets-shard-001-of-006.parquet",
)


def aws_json(args: list[str], env: dict[str, str] | None = None) -> dict:
    command_env = (env or os.environ).copy()
    command_env["AWS_PAGER"] = ""
    result = subprocess.run(
        ["aws", *args, "--output", "json"],
        check=False,
        capture_output=True,
        text=True,
        env=command_env,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def export_login_credentials() -> tuple[dict[str, str], datetime]:
    """Export one short-lived profile credential set for parallel CLI workers.

    Without this, concurrent AWS CLI processes can all refresh the same local
    login cache at once. Exporting serially keeps worker subprocesses off that
    cache while still using AWS CLI for every AWS API operation.
    """
    env = os.environ.copy()
    env["AWS_PAGER"] = ""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
        env.pop(name, None)
    profile = env.get("AWS_PROFILE", "default")
    result = subprocess.run(
        ["aws", "configure", "export-credentials", "--profile", profile, "--format", "process"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Could not export AWS CLI credentials")
    exported = json.loads(result.stdout)
    credential_env = {
        "AWS_ACCESS_KEY_ID": exported["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": exported["SecretAccessKey"],
        "AWS_SESSION_TOKEN": exported["SessionToken"],
    }
    expiry = datetime.fromisoformat(exported["Expiration"].replace("Z", "+00:00"))
    return credential_env, expiry


class SnapshotStream(io.RawIOBase):
    """Read-only seekable stream backed by EBS Direct APIs via AWS CLI."""

    def __init__(
        self,
        snapshot_id: str,
        region: str,
        size: int,
        scratch: Path,
        cache_blocks: int = 96,
        workers: int = 8,
    ) -> None:
        super().__init__()
        self.snapshot_id = snapshot_id
        self.region = region
        self.size = size
        self.scratch = scratch
        self.cache_blocks = cache_blocks
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.requests = 0
        self.downloaded_bytes = 0
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.aws_env = os.environ.copy()
        self.aws_env["AWS_PAGER"] = ""
        self.credential_expiry = datetime.min.replace(tzinfo=timezone.utc)
        self.next_refresh_attempt = datetime.min.replace(tzinfo=timezone.utc)
        self._refresh_credentials()

        # Listing once avoids one ListSnapshotBlocks API call for every block
        # read. The CLI follows NextToken pages automatically; block tokens
        # remain valid until the response's ExpiryTime.
        page = aws_json(
            [
                "ebs",
                "list-snapshot-blocks",
                "--region",
                self.region,
                "--snapshot-id",
                self.snapshot_id,
                "--max-results",
                "10000",
            ],
            env=self.aws_env,
        )
        if page.get("BlockSize") != BLOCK_SIZE:
            raise RuntimeError(f"Unexpected EBS block size: {page.get('BlockSize')}")
        self.block_tokens = {
            item["BlockIndex"]: item["BlockToken"] for item in page.get("Blocks", [])
        }
        print(f"Indexed {len(self.block_tokens):,} populated snapshot blocks.", flush=True)

    def _refresh_credentials(self) -> None:
        previous_expiry = self.credential_expiry
        credential_env, expiry = export_login_credentials()
        self.aws_env.update(credential_env)
        self.credential_expiry = expiry
        now = datetime.now(timezone.utc)
        if expiry <= previous_expiry:
            self.next_refresh_attempt = now + timedelta(minutes=2)
        else:
            self.next_refresh_attempt = expiry - timedelta(minutes=5)
        print(f"AWS CLI worker credentials valid until {expiry.astimezone(timezone.utc).isoformat()}.", flush=True)

    def _ensure_credentials(self) -> None:
        now = datetime.now(timezone.utc)
        if now + timedelta(minutes=5) >= self.credential_expiry and now >= self.next_refresh_attempt:
            # Reads are issued in batches, so this refresh runs on the main
            # thread between batches and never races the GetSnapshotBlock calls.
            self._refresh_credentials()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            new_position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new_position < 0:
            raise ValueError("negative seek position")
        self.position = new_position
        return new_position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def _fetch_block(self, index: int) -> bytes:
        token = self.block_tokens.get(index)
        if token is None:
            return bytes(BLOCK_SIZE)

        output_path = self.scratch / f"block-{index}.bin"
        env = self.aws_env.copy()
        command = [
            "aws",
            "ebs",
            "get-snapshot-block",
            "--region",
            self.region,
            "--snapshot-id",
            self.snapshot_id,
            "--block-index",
            str(index),
            "--block-token",
            token,
            str(output_path),
        ]
        retryable = (
            "Could not connect to the endpoint URL",
            "ConnectTimeoutError",
            "ReadTimeoutError",
            "Read timeout on endpoint URL",
            "EndpointConnectionError",
            "RequestThrottledException",
            "ThrottlingException",
            "InternalServerException",
            "ServiceUnavailable",
        )
        for attempt in range(4):
            output_path.unlink(missing_ok=True)
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
            if result.returncode == 0:
                break
            error = result.stderr.strip() or result.stdout.strip()
            if attempt == 3 or not any(marker in error for marker in retryable):
                raise RuntimeError(f"GetSnapshotBlock({index}) failed: {error}")
            delay = 2**attempt
            print(
                f"Transient EBS read error for block {index}; retrying in {delay}s "
                f"({attempt + 1}/4).",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        try:
            metadata = json.loads(result.stdout)
            block = output_path.read_bytes()
        finally:
            output_path.unlink(missing_ok=True)
        if len(block) != BLOCK_SIZE:
            raise IOError(f"block {index} had {len(block)} bytes, expected {BLOCK_SIZE}")
        checksum = base64.b64decode(metadata["Checksum"])
        if hashlib.sha256(block).digest() != checksum:
            raise IOError(f"SHA-256 validation failed for EBS block {index}")
        return block

    def _get_block(self, index: int) -> bytes:
        cached = self.cache.get(index)
        if cached is not None:
            self.cache.move_to_end(index)
            return cached
        block = self._fetch_block(index)
        self.cache[index] = block
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return block

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        self._ensure_credentials()
        first = self.position // BLOCK_SIZE
        last = (self.position + size - 1) // BLOCK_SIZE
        missing = [index for index in range(first, last + 1) if index not in self.cache]
        futures = {self.executor.submit(self._fetch_block, index): index for index in missing}
        for future in as_completed(futures):
            index = futures[future]
            block = future.result()
            self.cache[index] = block
            self.cache.move_to_end(index)
            if index in self.block_tokens:
                self.requests += 1
                self.downloaded_bytes += len(block)
                if self.requests % 100 == 0:
                    print(
                        f"Read {self.requests} EBS blocks "
                        f"({self.downloaded_bytes / 1024**2:.1f} MiB)",
                        file=sys.stderr,
                        flush=True,
                    )
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)

        chunks: list[bytes] = []
        remaining = size
        while remaining:
            block_index, within = divmod(self.position, BLOCK_SIZE)
            count = min(remaining, BLOCK_SIZE - within)
            block = self._get_block(block_index)
            chunks.append(block[within : within + count])
            self.position += count
            remaining -= count
        return b"".join(chunks)


def open_xfs(image: SnapshotStream):
    from dissect.volume.disk import Disk
    from dissect.xfs import XFS

    errors: list[str] = []
    try:
        disk = Disk(image)
        for partition in disk.partitions:
            try:
                stream = partition.open()
                filesystem = XFS(stream)
                filesystem.get("/")
                print(
                    f"Using XFS partition {partition.number} "
                    f"(offset {partition.offset}, size {partition.size})"
                )
                return filesystem, stream
            except Exception as exc:  # try remaining partitions if this isn't XFS
                errors.append(f"partition {partition.number}: {exc}")
    except Exception as exc:
        errors.append(f"partition table: {exc}")

    try:
        filesystem = XFS(image)
        filesystem.get("/")
        return filesystem, image
    except Exception as exc:
        errors.append(f"whole disk: {exc}")
    raise RuntimeError("Could not open XFS filesystem: " + " | ".join(errors))


def extract(
    snapshot_id: str,
    region: str,
    output_dir: Path,
    paths: list[str],
    resume: bool = False,
) -> int:
    from dissect.xfs.exceptions import FileNotFoundError as XFSFileNotFoundError

    snapshot = aws_json(
        [
            "ec2",
            "describe-snapshots",
            "--region",
            region,
            "--snapshot-ids",
            snapshot_id,
            "--query",
            "Snapshots[0]",
        ]
    )
    if snapshot.get("State") != "completed":
        raise RuntimeError(
            f"Snapshot {snapshot_id} is {snapshot.get('State')} "
            f"({snapshot.get('Progress', 'no progress reported')}); run again after it completes."
        )
    if not snapshot.get("Encrypted"):
        print("Snapshot is unencrypted.")
    else:
        print("Snapshot is encrypted; the configured AWS identity must have KMS decrypt access.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ebs-direct-", dir=output_dir) as temp:
        scratch = Path(temp)
        image = SnapshotStream(
            snapshot_id,
            region,
            int(snapshot["VolumeSize"]) * 1024**3,
            scratch,
        )
        try:
            filesystem, stream = open_xfs(image)
            found = 0
            for remote_path in paths:
                try:
                    entry = filesystem.get(remote_path)
                    if entry.filetype == stat.S_IFDIR:
                        print(f"Directory, skipped: {remote_path}")
                        continue
                    file_obj: BinaryIO = entry.open()
                    destination = output_dir / Path(remote_path.lstrip("/"))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    offset = destination.stat().st_size if resume and destination.is_file() else 0
                    if offset >= 8 and destination.suffix == ".parquet":
                        with destination.open("rb") as existing:
                            head = existing.read(4)
                            existing.seek(-4, io.SEEK_END)
                            tail = existing.read(4)
                        if head == b"PAR1" and tail == b"PAR1":
                            print(f"Already complete, skipping: {destination} ({offset:,} bytes)", flush=True)
                            file_obj.close()
                            found += 1
                            continue
                    if offset:
                        file_obj.seek(offset, io.SEEK_SET)
                        print(f"Resuming {remote_path} at byte {offset:,}.", flush=True)
                    with file_obj, destination.open("ab" if offset else "wb") as output:
                        while True:
                            chunk = file_obj.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            output.flush()
                    found += 1
                    size = destination.stat().st_size
                    note = ""
                    if destination.suffix == ".parquet":
                        with destination.open("rb") as parquet:
                            head = parquet.read(4)
                            parquet.seek(max(0, size - 4))
                            tail = parquet.read(4)
                        note = f" | parquet magic {head!r} ... {tail!r}"
                    print(f"Extracted {remote_path} -> {destination} ({size:,} bytes){note}", flush=True)
                except XFSFileNotFoundError:
                    print(f"Not present: {remote_path}", flush=True)
            close = getattr(stream, "close", None)
            if close:
                close()
        finally:
            image.executor.shutdown(wait=True, cancel_futures=True)
        print(
            f"Finished: {found} file(s), {image.requests} EBS data blocks, "
            f"{image.downloaded_bytes / 1024**2:.1f} MiB downloaded."
        )
        if found == 0:
            print("No requested artifacts were found; inspect the worker log and filesystem paths.")
        return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/v2-recovered/direct-ebs/shard-001"),
    )
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="Worker filesystem path to extract; may be repeated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing partial files from their current byte offset.",
    )
    args = parser.parse_args()
    try:
        found = extract(
            args.snapshot_id,
            args.region,
            args.output_dir,
            args.paths or list(DEFAULT_FILES),
            args.resume,
        )
    except Exception as exc:
        print(f"Recovery failed: {exc}", file=sys.stderr)
        return 1
    return 0 if found else 2


if __name__ == "__main__":
    raise SystemExit(main())
