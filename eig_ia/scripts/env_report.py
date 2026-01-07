import argparse
import json
import os
import platform
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    report = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception as exc:
        report["torch"] = "not installed"
        report["cuda_available"] = False
        report["cuda_device"] = str(exc)

    try:
        import transformers

        report["transformers"] = transformers.__version__
    except Exception:
        report["transformers"] = "not installed"

    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit = "unknown"
    report["git_commit"] = commit

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "env.txt"), "w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
