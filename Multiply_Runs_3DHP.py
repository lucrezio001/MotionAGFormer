#!/usr/bin/env python3
import subprocess
import os
import sys
from datetime import datetime
import re


def run_confidence_experiment(confidence_value):
    """Run single experiment with given confidence value"""
    
    os.makedirs("results/3dhp", exist_ok=True)
    
    # Use the same Python executable that's running this script
    python_exe = sys.executable
    
    cmd = [
        python_exe, "-W", "ignore", "train.py",
        "--eval-only",
        "--checkpoint", "checkpoint", 
        "--checkpoint-file", "motionagformer-xs-mpi.pth.tr",
        "--config", "configs/mpi/MotionAGFormer-xsmall.yaml",
        "--fixed-conf", str(confidence_value)
    ]
    
    print(f"Using Python: {python_exe}")
    print(f"Running confidence {confidence_value}...")
    
    output_file = f"results/3dhp/confidence_{confidence_value}.txt"
    
    # Env python path
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU to avoid DataParallel mismatch
    
    try:
        with open(output_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, 
                                  text=True, timeout=1800, env=env)
        
        if result.returncode == 0:
            print(f"✓ Confidence {confidence_value} SUCCESS")
            return True
        else:
            print(f"✗ Confidence {confidence_value} FAILED")
            print("!" * 20 + " ERROR LOG " + "!" * 20)
            with open(output_file, "r") as f:
                print(f.read())
            print("!" * 51)
            return False
            
    except Exception as e:
        print(f"✗ Confidence {confidence_value} ERROR: {e}")
        return False

def create_summary():
    """Create summary of all experiments"""
    
    confidence_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    with open("results/3dhp/summary.txt", "w") as summary:
        summary.write("CONFIDENCE EXPERIMENTS SUMMARY\n")
        summary.write("=" * 60 + "\n\n")
        summary.write(f"Date: {datetime.now()}\n\n")
        
        results = []
        
        for conf in confidence_values:
            filename = f"results/3dhp/confidence_{conf}.txt"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    content = f.read()
                    
                    # Extract metrics
                    mpjpe_match = re.search(r"Protocol #1 Error \(MPJPE\): ([0-9.]+)", content)
                    p_mpjpe_match = re.search(r"Protocol #2 Error \(P-MPJPE\): ([0-9.]+)", content)
                    acc_match = re.search(r"Acceleration error: ([0-9.]+)", content)
                    
                    mpjpe = mpjpe_match.group(1) if mpjpe_match else "FAILED"
                    p_mpjpe = p_mpjpe_match.group(1) if p_mpjpe_match else "FAILED"
                    acc = acc_match.group(1) if acc_match else "FAILED"
                    
                    line = f"Conf {conf:3.1f}: MPJPE={mpjpe:>8}, P-MPJPE={p_mpjpe:>8}, Acc={acc:>8}"
                    summary.write(line + "\n")
                    
                    if mpjpe != "FAILED":
                        results.append((conf, float(mpjpe), float(p_mpjpe)))
            else:
                summary.write(f"Conf {conf:3.1f}: FILE NOT FOUND\n")
        
        # Best/Worst analysis
        if results:
            best = min(results, key=lambda x: x[1])
            worst = max(results, key=lambda x: x[1])
            
            summary.write("\n" + "="*60 + "\n")
            summary.write(f"BEST:  Confidence {best[0]} -> MPJPE {best[1]:.2f}\n")
            summary.write(f"WORST: Confidence {worst[0]} -> MPJPE {worst[1]:.2f}\n")
    
    print("Summary created: results/3dhp/summary.txt")

def main():
    confidence_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    print("Starting confidence experiments...")
    print(f"Testing: {confidence_values}")
    
    success = 0
    for conf in confidence_values:
        if run_confidence_experiment(conf):
            success += 1
    
    print(f"\nCompleted: {success}/{len(confidence_values)} successful")
    create_summary()
    print("Check results/3dhp directory for all outputs!")

if __name__ == "__main__":
    main()