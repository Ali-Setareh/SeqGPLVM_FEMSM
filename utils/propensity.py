from pathlib import Path 

def propensity_dir(root: Path | str, model_name: str, train_id: str) -> Path:
    return Path(root) / "propensity" / model_name / train_id