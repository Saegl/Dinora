import pathlib


def upload_dataset(dataset_dir: pathlib.Path, wandb_label: str) -> None:
    # TODO: check that `dataset_dir` has report.json
    import wandb

    wandb.init(project="dinora-chess")

    artifact = wandb.Artifact(wandb_label, type="dataset")
    artifact.add_dir(dataset_dir)  # type: ignore

    wandb.log_artifact(artifact)
