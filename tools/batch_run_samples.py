#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
from pathlib import Path
import glob
import shlex

def parse_cfg_file(cfg_path):
    params = []
    
    if not os.path.exists(cfg_path):
        return params
        
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = ""
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith(' \\'):
                line = line[:-2].strip()
            content += line + " "
    
    if content.strip():
        try:
            parsed_args = shlex.split(content)
            params.extend(parsed_args)
        except ValueError as e:
            pass
    
    return params

def merge_configs(default_cfg_path, sample_cfg_path):
    """Merge default and sample-specific configurations."""
    default_params = parse_cfg_file(default_cfg_path)
    sample_params = parse_cfg_file(sample_cfg_path)
    
    # Merge parameters, sample-specific parameters will override default ones.
    all_params = default_params + sample_params
    
    return all_params

def run_inference(gpu_id, params, output_dir=None, seed=None):
    """Run the inference command"""
    cmd = [
        'python', 'infer.py'
    ]
    
    cmd.extend(params)
    
    if output_dir:
        output_added = False
        for i, arg in enumerate(cmd):
            if arg == '--output_path' and i + 1 < len(cmd):
                cmd[i + 1] = output_dir
                output_added = True
                break
        
        if not output_added:
            cmd.extend(['--output_path', output_dir])
    
    if seed is not None:
        seed_added = False
        for i, arg in enumerate(cmd):
            if arg == '--seed' and i + 1 < len(cmd):
                cmd[i + 1] = str(seed)
                seed_added = True
                break
        
        if not seed_added:
            cmd.extend(['--seed', str(seed)])
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"Executing command: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(shlex.quote(arg) for arg in cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, env=env, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch run inference for samples.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--config_dir", type=str, default="assets/config", help="Directory containing configuration files")
    parser.add_argument("--output_base", type=str, default="assets/output", help="Base directory for output")
    parser.add_argument("--sample_pattern", type=str, default="*.cfg", help="Pattern to match sample configuration files")
    parser.add_argument("--default_config", type=str, default="default.cfg", help="Filename for the default configuration")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    default_cfg_path = config_dir / args.default_config
    
    if not config_dir.exists():
        print("Error: Config directory not found: {}".format(config_dir))
        sys.exit(1)
    
    # Find all sample config files
    sample_cfg_files = glob.glob(str(config_dir / args.sample_pattern))
    sample_cfg_files = [f for f in sample_cfg_files if not f.endswith(args.default_config)]
    
    if not sample_cfg_files:
        print("Error: No sample config files found in {}".format(config_dir))
        sys.exit(1)
    
    print("Found {} sample config files.".format(len(sample_cfg_files)))
    print("Default config: {}".format(default_cfg_path))
    print("Using GPU: {}".format(args.gpu_id))
    if args.seed is not None:
        print("Random seed: {}".format(args.seed))
    print("=" * 80)
    
    success_count = 0
    total_count = len(sample_cfg_files)
    
    for cfg_file in sorted(sample_cfg_files):
        cfg_path = Path(cfg_file)
        sample_name = cfg_path.stem
        
        print("\nProcessing sample: {}".format(sample_name))
        print("Config file: {}".format(cfg_path))
        
        # Merge configurations
        merged_params = merge_configs(str(default_cfg_path), str(cfg_path))
        
        if not merged_params:
            print("Warning: Sample {} has no valid configuration parameters, skipping.".format(sample_name))
            continue
        
        # Create output directory
        output_dir = "{}/{}".format(args.output_base, sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run inference
        success = run_inference(args.gpu_id, merged_params, output_dir, args.seed)
        
        if success:
            success_count += 1
            print("✓ Sample {} processed successfully.".format(sample_name))
        else:
            print("✗ Sample {} processing failed.".format(sample_name))
        
        print("=" * 80)
    
    print("\nBatch processing finished!")
    print("Success: {}/{}".format(success_count, total_count))
    print("Failed: {}/{}".format(total_count - success_count, total_count))

if __name__ == "__main__":
    main()
