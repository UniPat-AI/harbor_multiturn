from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ResumeOutputMode(str, Enum):
    COPY_TO_OUTPUT_JOBS_DIR = "copy_to_output_jobs_dir"
    IN_PLACE_WITH_BACKUP = "in_place_with_backup"
    IN_PLACE_WITHOUT_BACKUP = "in_place_without_backup"


@dataclass(frozen=True)
class ResumeOutputPlan:
    output_mode: ResumeOutputMode
    jobs_dir: Path
    job_name: str | None
    resume_source_dir: Path
    resume_trial_name_for_config: str | None
    trial_parent: Path
    resume_trial_name: str

    @property
    def mutates_source_trial(self) -> bool:
        return self.output_mode in {
            ResumeOutputMode.IN_PLACE_WITH_BACKUP,
            ResumeOutputMode.IN_PLACE_WITHOUT_BACKUP,
        }

    @property
    def uses_backup(self) -> bool:
        return self.output_mode == ResumeOutputMode.IN_PLACE_WITH_BACKUP

    def backup_dir(self, timestamp: str) -> Path:
        if not self.uses_backup:
            raise ValueError("Resume output plan does not create a backup")
        return self.trial_parent / f"{self.resume_trial_name}__resumed_{timestamp}"


def plan_resume_output(
    *,
    resolved_resume: Path,
    jobs_dir: Path | None,
    output_jobs_dir: Path | None,
    no_resume_backup: bool,
) -> ResumeOutputPlan:
    trial_parent = resolved_resume.parent
    inferred_jobs_dir = trial_parent.parent
    inferred_job_name = trial_parent.name
    resume_trial_name = resolved_resume.name

    if output_jobs_dir is not None:
        return ResumeOutputPlan(
            output_mode=ResumeOutputMode.COPY_TO_OUTPUT_JOBS_DIR,
            jobs_dir=output_jobs_dir,
            job_name=None,
            resume_source_dir=resolved_resume,
            resume_trial_name_for_config=None,
            trial_parent=trial_parent,
            resume_trial_name=resume_trial_name,
        )

    if jobs_dir is not None and jobs_dir.resolve() != inferred_jobs_dir.resolve():
        raise ValueError(
            f"-o ({jobs_dir.resolve()}) conflicts with --resume-trial path "
            f"(inferred jobs_dir: {inferred_jobs_dir})"
        )

    return ResumeOutputPlan(
        output_mode=(
            ResumeOutputMode.IN_PLACE_WITHOUT_BACKUP
            if no_resume_backup
            else ResumeOutputMode.IN_PLACE_WITH_BACKUP
        ),
        jobs_dir=inferred_jobs_dir,
        job_name=inferred_job_name,
        resume_source_dir=resolved_resume,
        resume_trial_name_for_config=resume_trial_name,
        trial_parent=trial_parent,
        resume_trial_name=resume_trial_name,
    )
