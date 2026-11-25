"""Job manager for the Modal deployment (see modal_app.py).

The API container never touches the GPU. It stores the job record in a
modal.Dict and spawns `MeshWorker.run` on a GPU container, which writes its
files to a modal.Volume and updates the record as it goes. Downloads stream
straight from the Volume.
"""

from __future__ import annotations

import io
import itertools
import logging
from typing import Callable, Iterator, Optional

from PIL import Image

from .backends import GenerateOptions
from .jobs import JobRecord, JobStatus, new_job_id

log = logging.getLogger(__name__)

SpawnFn = Callable[[str, bytes, dict], None]


def volume_path(job_id: str, name: str) -> str:
    return f"jobs/{job_id}/{name}"


class ModalJobManager:
    def __init__(self, jobs_dict, volume, spawn: SpawnFn) -> None:
        self.jobs = jobs_dict
        self.volume = volume
        self.spawn = spawn

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def submit(self, image: Image.Image, options: GenerateOptions) -> JobRecord:
        job = JobRecord(id=new_job_id(), options=options.to_dict())
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        self.jobs[job.id] = job.to_dict()
        try:
            self.spawn(job.id, buf.getvalue(), job.options)
        except Exception as exc:
            log.exception("Could not start job %s", job.id)
            job.status, job.error = JobStatus.failed, f"Could not start the GPU worker: {exc}"
            self.jobs[job.id] = job.to_dict()
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        data = self.jobs.get(job_id)
        return JobRecord.from_dict(data) if data else None

    def read_file(self, job_id: str, name: str) -> Optional[Iterator[bytes]]:
        try:
            chunks = iter(self.volume.read_file(volume_path(job_id, name)))
            first = next(chunks)
        except StopIteration:
            return iter([b""])
        except Exception:  # modal raises NotFoundError / FileNotFoundError for a missing file
            return None
        return itertools.chain([first], chunks)

    def health(self) -> dict:
        # GPU containers scale from zero, so "loaded" means "can be started".
        return {"model_loaded": True, "model_error": None, "queue_size": None}
