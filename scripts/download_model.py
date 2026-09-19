"""Script to download the MediaPipe Hand Landmarker model asset."""

import os
import pathlib
import sys
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
EXPECTED_MIN_SIZE_BYTES = 7_000_000  # ~7.8 MB


def download_model(target_path: pathlib.Path) -> bool:
    """Download the model task asset if not already present."""
    if target_path.exists() and target_path.stat().st_size >= EXPECTED_MIN_SIZE_BYTES:
        print(f"Model already exists at: {target_path} ({target_path.stat().st_size / 1_048_576:.2f} MB)")
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand_landmarker.task from:\n  {MODEL_URL}")
    print(f"Destination: {target_path}")

    def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            sys.stdout.write(f"\r  Progress: {percent:5.1f}% ({mb_down:4.1f} / {mb_total:4.1f} MB)")
            sys.stdout.flush()

    try:
        temp_path = target_path.with_suffix(".tmp")
        urllib.request.urlretrieve(MODEL_URL, temp_path, reporthook=progress_callback)
        sys.stdout.write("\n")

        if temp_path.stat().st_size < EXPECTED_MIN_SIZE_BYTES:
            raise RuntimeError(
                f"Downloaded file is smaller than expected: "
                f"{temp_path.stat().st_size} bytes."
            )

        temp_path.replace(target_path)
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"\nFailed to download model: {e}", file=sys.stderr)
        return False


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parent.parent
    target_path = project_root / "assets" / "hand_landmarker.task"
    success = download_model(target_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
