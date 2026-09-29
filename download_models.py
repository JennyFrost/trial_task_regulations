from pathlib import Path
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS = {
    # "embedding_model": "deepvk/USER-bge-m3",
    "bge_m3": "deepvk/USER-bge-m3",
    "e5": "intfloat/multilingual-e5-base",
    # "reranker_model": "qilowoq/bge-reranker-v2-m3-en-ru",
    # "reranker_model_bert": "ARGA100/ru-reranker-modernbert-small"

}


def download_model(name:str, repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name}: {repo_id}")
    print(f"Target: {local_dir}")

    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"Downloaded {name}: {repo_id}")


def main() -> None:
    for name, repo_id in MODELS.items():
        download_model(name=name, repo_id=repo_id, local_dir=MODELS_DIR / name)
    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()