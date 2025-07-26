import kagglehub
import shutil
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("miadul/prostate-cancer-risk-and-lifestyle-synthetic-dataset")

print("Path to dataset files:", path)

target_dir = Path('/Users/halilgoktas/Development/prostat_risk/data')
target_dir.mkdir(exist_ok=True)

for file in Path(path).iterdir():
    shutil.copy(file, target_dir / file.name)