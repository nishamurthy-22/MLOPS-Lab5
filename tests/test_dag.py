"""Checks that the DAG loads and has the expected structure.

Needs Airflow installed, so it is skipped in the local venv. Airflow's own
DAG File Processing Stats confirm the same thing inside Docker.
"""

import pytest

pytest.importorskip("airflow", reason="Airflow is not installed in this environment")


def test_dag_structure():
    from dry_bean_dag import dry_bean_dag as dag

    assert dag.dag_id == "dry_bean_clustering"
    assert set(dag.params.keys()) == {"k_min", "k_max", "test_size"}
    assert len(dag.tasks) == 7
    downstream = {t.task_id: t.downstream_task_ids for t in dag.tasks}
    assert downstream["train_model"] == {"evaluate_model", "profile_clusters"}
    assert downstream["write_report"] == set()