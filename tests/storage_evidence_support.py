"""Synthetic complete storage receipts for offline CI gate regressions."""

from scripts.ci_storage import (
    CleanupState,
    CollectionReceipt,
    ContainerSample,
    StackEvidence,
    StorageIdentity,
    StorageSample,
)


def complete_stack(project: str = "a" * 12) -> StackEvidence:
    """No Docker or collection setup happens while constructing this fixture."""
    return StackEvidence(
        identity=StorageIdentity(tested_sha="a" * 40, run_id="123", run_attempt="2"),
        project="insightpilot-test-milvus-" + project,
        completed=True,
        collections=[
            CollectionReceipt(name="step31_owned", owner="test", state=CleanupState.ABSENT)
        ],
        samples=[
            StorageSample(
                phase=phase,
                containers=[
                    ContainerSample(
                        service=service,
                        memory_bytes=100,
                        limit_bytes=1000,
                        oom_killed=False,
                        running=True,
                        restarts=0,
                    )
                    for service in ("milvus", "etcd", "minio")
                ],
            )
            for phase in ("startup", "final")
        ],
    )
