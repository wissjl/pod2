# project/scripts/run_model.py

import argparse
import os
import sys

# Get the directory where run_model.py itself is located.
# This will be 'project/scripts/' when run via subprocess with cwd=project_root.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'scripts' directory to sys.path.
# This is crucial for Python to find the modules for imports that refer to files within 'scripts'.
sys.path.append(script_dir)

# --- CRITICAL CHANGE HERE: Change relative imports to absolute imports ---
# Now that 'script_dir' is in sys.path, Python can directly find these modules.
from automated_lstm_vf import run_lstm_model
from automated_cnn_lstm import run_cnn_lstm_model
from automated_garch_lstm import run_garch_lstm_model
# -----------------------------------------------------------------------

# Mapping string names (from command line) to the actual model execution functions.
model_map = {
    "lstm": run_lstm_model,
    "cnn_lstm": run_cnn_lstm_model,
    "garch_lstm": run_garch_lstm_model
}

if __name__ == "__main__":
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run a specified machine learning model with given input and output paths."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to run: lstm | cnn_lstm | garch_lstm"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Name of the input CSV file (e.g., df1.csv, eth_df1.csv)."
             "Assumed to be in the 'input_files' directory relative to the project root."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the model outputs. Can be an absolute or relative path."
    )
    args = parser.parse_args()

    # 2. Validate model name and prepare paths
    model_name = args.model.lower()

    # Determine the current working directory.
    # When called by run_all_jobs.py with `cwd=current_script_dir`, os.getcwd() will be
    # the 'project/' root directory (e.g., 'cloudhoussem/').
    project_root_cwd = os.getcwd()

    # Construct the full path to the input file.
    # It expects input files to be in 'input_files/' directory relative to the project root.
    input_path = os.path.join(project_root_cwd, "input_files", args.input_file)

    # Construct the full path for the specific output file/directory for this model run.
    # args.output_dir is typically an absolute path passed from run_all_jobs.py.
    output_path = os.path.join(args.output_dir, f"{model_name}_{args.input_file.replace('.csv','')}")

    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Execute the specified model
    if model_name in model_map:
        print(f"[{model_name.upper()}] Processing input: {args.input_file}")
        print(f"Saving output to: {output_path}")
        model_map[model_name](input_path, output_path)
        print(f"[{model_name.upper()}] Finished processing {args.input_file}.")
    else:
        print(f"Error: Model '{model_name}' not recognized.")
        print(f"Please choose from: {list(model_map.keys())}")
        sys.exit(1)