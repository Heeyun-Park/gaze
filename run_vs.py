import os
import json
import subprocess
import pandas as pd
from datetime import datetime

DATA_BASE = "./data_occluded_{}"
OUTPUT_DIR = "./outputs"

SCRIPT_GAZE360 = "models/gaze360/run.py"
SCRIPT_SWITCH = "models/switchgaze/run.py"

TRAIN_PATTERN = "train_occlusion_{}.txt"
VAL_PATTERN   = "val_occlusion_{}.txt"
TEST_PATTERN  = "test_occlusion_{}.txt"

OCCLUSION_RATIOS = ["0", "25", "50", "75"]
OCCLUSION_RATIOS = ["0"]

CONFIG_GAZE360 = {
    "epochs": 20,
    "batch_size": 46,
    "num_workers": 4,
    "lr": 1e-4,
    "backbone": "resnet18",
    # "resume": "outputs/Gaze360_occlusion_25_20251119_1748/checkpoint/best_model.pth.tar"
}

CONFIG_SWITCH = {
    "epochs": 40,
    "batch_size": 64,
    "num_workers": 8,
    "lr": 1e-4,
    "backbone": "resnet18",
    # "resume": "outputs/SwitchGaze_occlusion_0_20251119_0140/checkpoint/best_model.pth.tar"
}

def run_experiment(model_type, ratio, config):
    script = SCRIPT_SWITCH if model_type == "switchgaze" else SCRIPT_GAZE360

    data_path = DATA_BASE.format(ratio)
    train_list = TRAIN_PATTERN.format(ratio)
    val_list   = VAL_PATTERN.format(ratio)
    test_list  = TEST_PATTERN.format(ratio)

    print(f"\n=== Running {model_type.upper()} (ratio={ratio}%) ===")

    cmd = [
        "python", script,
        "--data_path", data_path,
        "--train_list", train_list,
        "--val_list", val_list,
        "--test_list", test_list,
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--num_workers", str(config["num_workers"]),
        "--lr", str(config["lr"]),
        "--backbone", config["backbone"],
        "--save_csv",
        # "--resume", config["resume"]
    ]

    env = os.environ.copy()
    main_root = os.path.abspath(".")
    env["PYTHONPATH"] = (
        main_root if "PYTHONPATH" not in env else f"{main_root}:{env['PYTHONPATH']}"
    )

    subprocess.run(cmd, check=True, env=env)
    print(f"[OK] {model_type.upper()} ratio={ratio} completed")

def collect_results():
    logs = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f == "log.json":
                path = os.path.join(root, f)
                with open(path, "r") as j:
                    data = json.load(j)
                    data["exp_dir"] = root
                    logs.append(data)
    return logs

def build_result_table(logs):
    df = pd.DataFrame(logs)
    df["ratio"] = df["train_list"].str.extract(r"(\d+)\.txt").astype(int)
    df["best_val_err"] = df["best_val_err"].round(3)
    df["test_err"] = df["test_err"].round(3)

    table = df.pivot(index="ratio", columns="model", values="test_err").sort_index()
    table = table.rename_axis("Occlusion Ratio (%)").reset_index()
    table["Occlusion Ratio (%)"] = table["Occlusion Ratio (%)"].astype(str) + "%"

    print(table.to_string(index=False))
    table.to_csv(f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
    return table


if __name__ == "__main__":
    print("=== Running experiments on occluded datasets ===")
    for r in OCCLUSION_RATIOS:
        # run_experiment("gaze360", r, CONFIG_GAZE360)
        run_experiment("switchgaze", r, CONFIG_SWITCH)

    logs = collect_results()
    build_result_table(logs)
