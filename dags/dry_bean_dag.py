"""Airflow DAG: validate, cluster, and evaluate the Dry Bean dataset with K-Means."""

from datetime import datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task

from src import tasks

default_args = {
    "owner": "Nisha Murthy Dinesh",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="dry_bean_clustering",
    description="Validate, cluster, and evaluate the Dry Bean dataset with K-Means",
    schedule=None,  # triggered manually
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "clustering", "dry-bean"],
    params={
        "k_min": Param(2, type="integer", minimum=2, description="Smallest k to try"),
        "k_max": Param(15, type="integer", minimum=3, maximum=30, description="Largest k to try"),
        "test_size": Param(0.2, type="number", minimum=0.1, maximum=0.5,
                           description="Share of beans held out for testing"),
    },
)
def dry_bean_clustering():

    @task
    def validate_data():
        run_dir = tasks.make_run_dir(get_current_context()["run_id"])
        return {"run_dir": run_dir, **tasks.validate(run_dir)}

    @task
    def prepare_data(validation):
        params = get_current_context()["params"]
        run_dir = validation["run_dir"]
        return {"run_dir": run_dir, **tasks.prepare(run_dir, params["test_size"])}

    @task
    def select_k(prepared):
        params = get_current_context()["params"]
        run_dir = prepared["run_dir"]
        return {"run_dir": run_dir, **tasks.select_k(run_dir, params["k_min"], params["k_max"])}

    @task
    def train_model(choice):
        run_dir = choice["run_dir"]
        return {"run_dir": run_dir, "k": choice["k"], "model_path": tasks.train(run_dir, choice["k"])}

    @task
    def evaluate_model(trained):
        return tasks.evaluate(trained["run_dir"])

    @task
    def profile_clusters(trained):
        return tasks.profile(trained["run_dir"])

    @task
    def write_report(validation, prepared, choice, metrics, clusters):
        return tasks.report(validation["run_dir"], validation, prepared, choice, metrics, clusters)

    validation = validate_data()
    prepared = prepare_data(validation)
    choice = select_k(prepared)
    trained = train_model(choice)
    # evaluate_model and profile_clusters run in parallel after training
    metrics = evaluate_model(trained)
    clusters = profile_clusters(trained)
    write_report(validation, prepared, choice, metrics, clusters)


dry_bean_dag = dry_bean_clustering()

if __name__ == "__main__":
    dry_bean_dag.test()