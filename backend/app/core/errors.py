from __future__ import annotations

from typing import Optional
from fastapi import HTTPException

class SpectraSRError(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        recoverable: bool = False,
        suggested_action: Optional[str] = None
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail={
                "code": code,
                "message": message,
                "recoverable": recoverable,
                "suggested_action": suggested_action or "Check server logs or try again."
            }
        )

class SceneNotFoundError(SpectraSRError):
    def __init__(self, scene_id: str) -> None:
        super().__init__(
            status_code=404,
            code="SCENE_NOT_FOUND",
            message=f"Scene '{scene_id}' could not be located in cache or CDSE.",
            recoverable=True,
            suggested_action="Verify the scene_id or use one of the pre-cached demo scenes."
        )

class JobNotFoundError(SpectraSRError):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            status_code=404,
            code="JOB_NOT_FOUND",
            message=f"Processing job '{job_id}' does not exist.",
            recoverable=False,
            suggested_action="Verify the job_id or initiate a new processing run."
        )

class ArtifactNotFoundError(SpectraSRError):
    def __init__(self, artifact_type: str, job_id: str) -> None:
        super().__init__(
            status_code=404,
            code="ARTIFACT_NOT_FOUND",
            message=f"Artifact of type '{artifact_type}' for job '{job_id}' is not yet available.",
            recoverable=True,
            suggested_action="Wait until the job transitions to 'COMPLETED'."
        )
