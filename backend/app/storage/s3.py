"""S3-compatible storage (SeaweedFS locally; AWS S3, Cloudflare R2, Backblaze B2 or MinIO in production)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from .base import CHUNK, ObjectNotFound


class S3Storage:
    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str],
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(retries={"max_attempts": 5, "mode": "standard"}, s3={"addressing_style": "path"}),
        )

    async def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        def _ensure():
            try:
                self.client.head_bucket(Bucket=self.bucket)
            except ClientError:
                self.client.create_bucket(Bucket=self.bucket)

        await asyncio.to_thread(_ensure)

    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self.client.put_object, Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )

    async def put_file(self, key: str, path: Path, content_type: str) -> None:
        await asyncio.to_thread(
            self.client.upload_file, str(path), self.bucket, key, ExtraArgs={"ContentType": content_type}
        )

    async def download(self, key: str, dest: Path) -> None:
        from botocore.exceptions import ClientError

        try:
            await asyncio.to_thread(self.client.download_file, self.bucket, key, str(dest))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                raise ObjectNotFound(key) from exc
            raise

    async def size(self, key: str) -> Optional[int]:
        from botocore.exceptions import ClientError

        try:
            head = await asyncio.to_thread(self.client.head_object, Bucket=self.bucket, Key=key)
            return int(head["ContentLength"])
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def iter_chunks(self, key: str) -> Iterator[bytes]:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"]
        try:
            yield from body.iter_chunks(CHUNK)
        finally:
            body.close()

    async def delete_prefix(self, prefix: str) -> None:
        def _delete():
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                if keys:
                    self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})

        await asyncio.to_thread(_delete)

    async def ping(self) -> None:
        await asyncio.to_thread(self.client.head_bucket, Bucket=self.bucket)
