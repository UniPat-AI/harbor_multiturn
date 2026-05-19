from harbor.models.job.config import JobConfig
from harbor.models.trial.config import VerifierConfig


def test_verifier_config_defaults_to_success_snapshot_policy():
    assert VerifierConfig().multiround_state_cache_policy == "success"
    assert VerifierConfig().multiround_state_retention_policy == "latest"


def test_job_config_inherits_success_snapshot_policy_default():
    assert JobConfig().verifier.multiround_state_cache_policy == "success"
    assert JobConfig().verifier.multiround_state_retention_policy == "latest"
