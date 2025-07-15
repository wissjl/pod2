import os
import subprocess

current_script_dir = os.path.dirname(os.path.abspath(__file__))
models = ["garch_lstm"]
input_files = [f"df{i}.csv" for i in range(1, 13)]
output_dir = os.path.join(current_script_dir, "output_dir")
os.makedirs(output_dir, exist_ok=True)

run_model_script_path = os.path.join(current_script_dir, "scripts", "run_model.py")

for model in models:
    for input_file in input_files:
        cmd = [
            "python", run_model_script_path,
            "--model", model,
            "--input_file", input_file,
            "--output_dir", output_dir
        ]
        subprocess.run(cmd, check=True, cwd=current_script_dir)