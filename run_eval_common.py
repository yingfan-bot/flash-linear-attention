#!/usr/bin/env python3

import argparse
import subprocess
import os
from typing import List

def parse_args():
   parser = argparse.ArgumentParser(description="Launch RULER experiments")
   parser.add_argument(
       "--model_name", type=str, help="Model namåe to use"
   )
   parser.add_argument(
       "--path", type=str, help="Model path to use"
   )
   parser.add_argument(
       "--niah_single_1", action="store_true", help="Run niah_single_1 task"
   )
   parser.add_argument(
       "--niah_single_2", action="store_true", help="Run niah_single_2 task"
   )
   parser.add_argument(
       "--niah_single_3", action="store_true", help="Run niah_single_3 task"
   )
   parser.add_argument(
       "--niah_multikey_1", action="store_true", help="Run niah_multikey_1 task"
   )
   parser.add_argument(
       "--niah_multiquery", action="store_true", help="Run niah_multiquery task"
   )
   parser.add_argument(
       "--niah_multivalue", action="store_true", help="Run niah_multivalue task"
   )
   parser.add_argument(
       "--length",
       type=str,
       help='Comma-separated list of sequence lengths (e.g., "8192,16384")',
   )
   parser.add_argument(
       "--device", type=int, default=0, help="Device to run the experiment on"
   )

   return parser.parse_args()


def run_experiment(model_name: str, task_name: str, path: str, device: int):
  
   output_dir = f"../RULER/{model_name}/{task_name}"

   # Create output directory if it doesn't exist
   os.makedirs(output_dir, exist_ok=True)

   # Construct the command
   # Using main_process_port=0 tells accelerate to find an available port automatically
   port = 12345 + device
   cmd = [
       "accelerate",
       "launch",
       "--num_processes=1",
       "--main_process_port="
       + str(port),  # This will automatically find an available port
       "-m",
       "evals.harness",
       "--output_path",
       output_dir,
       "--tasks",
       task_name,
       "--model_args",
       f"pretrained={path},use_cache=False,dtype=bfloat16,max_length=32768,trust_remote_code=True",
       "--batch_size",
       "1",
       "--show_config",
       "--trust_remote_code",
   ]

   # Set environment variable
   env = os.environ.copy()
   env["CC"] = "/usr/bin/gcc"

   # Run the command
   print(f"Running experiment for {model_name} on {task_name}")
   subprocess.run(cmd, env=env)


def main():
   args = parse_args()

   # Get list of tasks to run
   tasks = ["wikitext", "lambada_openai", "arc_easy", "arc_challenge", "winogrande", "hellaswag", "piqa", "social_iqa"]

   # Run experiments
   for task in tasks:
       run_experiment(args.model_name, task, args.path, args.device)


if __name__ == "__main__":
   main()
   
