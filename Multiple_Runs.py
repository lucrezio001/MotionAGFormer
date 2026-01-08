#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import re
from datetime import datetime

# --- DEFAULT CONFIG ---

DATASET_CONFIGS = {
    'h36m': {
        'script': 'train.py',
        'config': 'configs/h36m/MotionAGFormer-xsmall.yaml',
        'checkpoint': 'checkpoint/motionagformer-xs-h36m.pth.tr',
        'output_dir': 'results/h36m'
    },
    '3dhp': {
        'script': 'train_3dhp.py',
        'config': 'configs/mpi/MotionAGFormer-xsmall.yaml',
        'checkpoint': 'checkpoint/motionagformer-xs-mpi.pth.tr',
        'output_dir': 'results/3dhp'
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Script per eseguire test multipli con diverse confidence.")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['h36m', '3dhp'],
                        help='Il dataset da usare (h36m o 3dhp)')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Percorso del file .yaml (opzionale, sovrascrive il default)')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Percorso del file pesi .pth.tr (opzionale, sovrascrive il default)')
    
    parser.add_argument('--gpu', type=str, default="0",
                        help='ID della GPU da usare (default: 0)')

    return parser.parse_args()

def run_experiment(dataset_name, script_path, config_path, ckpt_path, output_dir, confidence_value, gpu_id):
    """Esegue un singolo esperimento."""
    
    os.makedirs(output_dir, exist_ok=True)
    python_exe = sys.executable
    
    cmd = [
        python_exe, "-W", "ignore", script_path,
        "--eval-only",
        "--checkpoint", os.path.dirname(ckpt_path), # train.py folder
        "--checkpoint-file", os.path.basename(ckpt_path), # train.py file name
        "--config", config_path,
        "--fixed-conf", str(confidence_value)
    ]
    
    print(f"Running {dataset_name} | Conf: {confidence_value}...")
    
    output_file = os.path.join(output_dir, f"confidence_{confidence_value}.txt")
    
    # venv for uv
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env
        )
        
        # log 
        with open(output_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== ERROR LOG ===\n")
                f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ Success (Conf {confidence_value})")
            return True
        else:
            print(f"✗ Failed (Conf {confidence_value})")
            if len(result.stderr) > 0:
                print(f"Error snippet: {result.stderr[-300:]}") 
            return False
            
    except Exception as e:
        print(f"✗ System Error: {e}")
        return False

def extract_metrics(content, dataset_name):
    """Estrae le metriche specifiche in base al dataset usando Regex."""
    metrics = {}
    
    if dataset_name == 'h36m':
        mpjpe = re.search(r"Protocol #1 Error \(MPJPE\): ([0-9.]+)", content)
        p_mpjpe = re.search(r"Protocol #2 Error \(P-MPJPE\): ([0-9.]+)", content)
        acc = re.search(r"Acceleration error: ([0-9.]+)", content)
        
        metrics['MPJPE'] = float(mpjpe.group(1)) if mpjpe else None
        metrics['P-MPJPE'] = float(p_mpjpe.group(1)) if p_mpjpe else None
        metrics['Acc'] = float(acc.group(1)) if acc else None
        
    elif dataset_name == '3dhp':
        mpjpe = re.search(r"MPJPE: ([0-9.]+)", content)
        pck = re.search(r"PCK: ([0-9.]+)", content)
        auc = re.search(r"AUC: ([0-9.]+)", content)
        
        metrics['MPJPE'] = float(mpjpe.group(1)) if mpjpe else None
        metrics['PCK'] = float(pck.group(1)) if pck else None
        metrics['AUC'] = float(auc.group(1)) if auc else None
        
    return metrics

def create_summary(dataset_name, output_dir, confidence_values):
    summary_path = os.path.join(output_dir, "summary.txt")
    
    with open(summary_path, "w") as summary:
        summary.write(f"CONFIDENCE EXPERIMENTS SUMMARY ({dataset_name.upper()})\n")
        summary.write("=" * 60 + "\n\n")
        summary.write(f"Date: {datetime.now()}\n\n")
        
        results = []
        
        for conf in confidence_values:
            filename = os.path.join(output_dir, f"confidence_{conf}.txt")
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    content = f.read()
                    
                metrics = extract_metrics(content, dataset_name)
                
                if dataset_name == 'h36m':
                    mpjpe = metrics.get('MPJPE')
                    p_mpjpe = metrics.get('P-MPJPE')
                    acc = metrics.get('Acc')
                    
                    if mpjpe is not None:
                        line = f"Conf {conf:3.1f}: MPJPE={mpjpe:>6.2f}, P-MPJPE={p_mpjpe:>6.2f}, Acc={acc:>6.2f}"
                        results.append((conf, mpjpe))
                    else:
                        line = f"Conf {conf:3.1f}: FAILED (Metrics not found)"
                        
                elif dataset_name == '3dhp':
                    mpjpe = metrics.get('MPJPE')
                    pck = metrics.get('PCK')
                    auc = metrics.get('AUC')
                    
                    if mpjpe is not None:
                        line = f"Conf {conf:3.1f}: MPJPE={mpjpe:>6.2f}, PCK={pck:>6.2f}, AUC={auc:>6.2f}"
                        results.append((conf, mpjpe))
                    else:
                        line = f"Conf {conf:3.1f}: FAILED (Metrics not found)"
                
                summary.write(line + "\n")
            else:
                summary.write(f"Conf {conf:3.1f}: FILE NOT FOUND\n")
        
        # Best result analysis (Minimizing MPJPE)
        if results:
            best = min(results, key=lambda x: x[1])
            summary.write("\n" + "="*60 + "\n")
            summary.write(f"BEST CONFIDENCE: {best[0]} (MPJPE: {best[1]:.2f})\n")

    print(f"Summary created: {summary_path}")

def main():
    args = parse_args()
    
    # load config
    cfg = DATASET_CONFIGS[args.dataset]
    
    # new args 
    script_to_run = cfg['script']
    config_file = args.config if args.config else cfg['config']
    checkpoint_file = args.checkpoint if args.checkpoint else cfg['checkpoint']
    output_directory = cfg['output_dir']
    
    print(f"--- STARTING EXPERIMENT ---")
    print(f"Dataset:    {args.dataset.upper()}")
    print(f"Script:     {script_to_run}")
    print(f"Config:     {config_file}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Output:     {output_directory}")
    print("-" * 30)
    
    # list of confidence value to test
    confidence_values = [0.1]
    
    success_count = 0
    for conf in confidence_values:
        if run_experiment(args.dataset, script_to_run, config_file, checkpoint_file, output_directory, conf, args.gpu):
            success_count += 1
            
    print(f"\nCompleted: {success_count}/{len(confidence_values)} successful runs.")
    create_summary(args.dataset, output_directory, confidence_values)

if __name__ == "__main__":
    main()