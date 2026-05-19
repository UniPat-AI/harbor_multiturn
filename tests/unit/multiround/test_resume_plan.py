from pathlib import Path

import pytest

from harbor.multiround.resume_plan import (
    ResumeOutputMode,
    plan_resume_output,
)


def test_plan_resume_output_copies_to_explicit_output_jobs_dir(tmp_path: Path):
    resume_trial = tmp_path / "jobs" / "demo-job" / "trial-a"
    output_jobs_dir = tmp_path / "new-jobs"

    plan = plan_resume_output(
        resolved_resume=resume_trial,
        jobs_dir=None,
        output_jobs_dir=output_jobs_dir,
        no_resume_backup=False,
    )

    assert plan.output_mode == ResumeOutputMode.COPY_TO_OUTPUT_JOBS_DIR
    assert plan.jobs_dir == output_jobs_dir
    assert plan.job_name is None
    assert plan.resume_source_dir == resume_trial
    assert plan.resume_trial_name_for_config is None
    assert plan.mutates_source_trial is False
    assert plan.uses_backup is False


def test_plan_resume_output_infers_in_place_jobs_dir_and_job_name(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    resume_trial = jobs_dir / "demo-job" / "trial-a"

    plan = plan_resume_output(
        resolved_resume=resume_trial,
        jobs_dir=jobs_dir,
        output_jobs_dir=None,
        no_resume_backup=False,
    )

    assert plan.output_mode == ResumeOutputMode.IN_PLACE_WITH_BACKUP
    assert plan.jobs_dir == jobs_dir
    assert plan.job_name == "demo-job"
    assert plan.resume_source_dir == resume_trial
    assert plan.resume_trial_name_for_config == "trial-a"
    assert plan.trial_parent == jobs_dir / "demo-job"
    assert plan.resume_trial_name == "trial-a"
    assert plan.mutates_source_trial is True
    assert plan.uses_backup is True
    assert plan.backup_dir("20260519_120000") == (
        jobs_dir / "demo-job" / "trial-a__resumed_20260519_120000"
    )


def test_plan_resume_output_marks_no_backup_in_place_resume(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    resume_trial = jobs_dir / "demo-job" / "trial-a"

    plan = plan_resume_output(
        resolved_resume=resume_trial,
        jobs_dir=None,
        output_jobs_dir=None,
        no_resume_backup=True,
    )

    assert plan.output_mode == ResumeOutputMode.IN_PLACE_WITHOUT_BACKUP
    assert plan.jobs_dir == jobs_dir
    assert plan.job_name == "demo-job"
    assert plan.resume_source_dir == resume_trial
    assert plan.resume_trial_name_for_config == "trial-a"
    assert plan.mutates_source_trial is True
    assert plan.uses_backup is False
    with pytest.raises(ValueError, match="does not create a backup"):
        plan.backup_dir("20260519_120000")


def test_plan_resume_output_rejects_conflicting_jobs_dir(tmp_path: Path):
    resume_trial = tmp_path / "jobs" / "demo-job" / "trial-a"

    with pytest.raises(ValueError, match="conflicts with --resume-trial path"):
        plan_resume_output(
            resolved_resume=resume_trial,
            jobs_dir=tmp_path / "other-jobs",
            output_jobs_dir=None,
            no_resume_backup=False,
        )
