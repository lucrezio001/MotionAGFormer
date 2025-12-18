#!/usr/bin/env python3
"""
Confidence Analysis Plot Generator for MotionAGFormer
Plots MPJPE, P-MPJPE, and Acceleration Error vs Fixed Confidence Values
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_confidence_analysis():
    # Experimental data
    confidence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    mpjpe = [85.96820605755394, 78.16472523010354, 70.5549414779417, 62.15661190930386, 
             55.08909790737663, 51.23915490310516, 49.51274665795492, 48.658957073995005, 
             48.16512807061996, 47.933310131235615, 48.23721023262347, 51.0553167731881, 
             56.027098891380014, 62.77889267095095, 71.10313800605621]

    p_mpjpe = [69.19641937213594, 63.27439938712639, 57.433979136893846, 50.6495985742123, 
               44.71539512880126, 41.53419110765716, 40.215250621738186, 39.721095780764095, 
               39.55343108631735, 39.592871560255695, 40.0349855615901, 42.019232142013585, 
               45.02282505220709, 48.56565541880073, 52.422455451476566]

    acceleration = [1.1634051254844207, 1.156221835479387, 1.1207153348789545, 1.0523108640460603, 
                   0.9754839708897466, 0.9289919076951765, 0.9247072961040111, 0.9605753580877388, 
                   1.0367304531053578, 1.1306972524871655, 1.2041195581807875, 1.3366359592006931, 
                   1.5076584862455138, 1.6577663182276037, 1.79261846742814]

    # Default confidence reference values
    default_mpjpe = 45.14923548348352
    default_p_mpjpe = 36.89173169232892
    default_acceleration = 0.8405583363551942

    runif_mpjpe = 62.10465406930499
    runif_p_mpjpe = 50.416696280520824
    runif_acceleration = 1.04457634818685

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MotionAGFormer Performance for Fixed Confidence Values', fontsize=16, fontweight='bold')

    # Plot 1: MPJPE
    axes[0].plot(confidence, mpjpe, 'o-', linewidth=2, markersize=6, color='blue', label='Fixed Confidence')
    axes[0].axhline(y=default_mpjpe, color='red', linestyle='--', linewidth=2, 
                   label=f'Default Confidence ({default_mpjpe:.2f} mm)')
    axes[0].axhline(y=runif_mpjpe, color='blue', linestyle='--', linewidth=2, 
                   label=f'random uniform Confidence ({default_mpjpe:.2f} mm)')
    axes[0].set_xlabel('Fixed Confidence Value')
    axes[0].set_ylabel('MPJPE (mm)')
    axes[0].set_title('Protocol #1 Error (MPJPE)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0.05, 1.55)

    # Highlight optimal range for MPJPE
    optimal_idx_mpjpe = np.argmin(mpjpe)
    axes[0].plot(confidence[optimal_idx_mpjpe], mpjpe[optimal_idx_mpjpe], 
                'go', markersize=10, markerfacecolor='lightgreen', markeredgecolor='green',
                label=f'Best Fixed: {confidence[optimal_idx_mpjpe]} ({mpjpe[optimal_idx_mpjpe]:.2f} mm)')
    axes[0].legend()

    # Plot 2: P-MPJPE
    axes[1].plot(confidence, p_mpjpe, 'o-', linewidth=2, markersize=6, color='orange', label='Fixed Confidence')
    axes[1].axhline(y=default_p_mpjpe, color='red', linestyle='--', linewidth=2,
                   label=f'Default Confidence ({default_p_mpjpe:.2f} mm)')
    axes[1].axhline(y=runif_p_mpjpe, color='blue', linestyle='--', linewidth=2, 
                   label=f'random uniform Confidence ({default_mpjpe:.2f} mm)')
    axes[1].set_xlabel('Fixed Confidence Value')
    axes[1].set_ylabel('P-MPJPE (mm)')
    axes[1].set_title('Protocol #2 Error (P-MPJPE)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0.05, 1.55)

    # Highlight optimal range for P-MPJPE
    optimal_idx_pmpjpe = np.argmin(p_mpjpe)
    axes[1].plot(confidence[optimal_idx_pmpjpe], p_mpjpe[optimal_idx_pmpjpe], 
                'go', markersize=10, markerfacecolor='lightgreen', markeredgecolor='green',
                label=f'Best Fixed: {confidence[optimal_idx_pmpjpe]} ({p_mpjpe[optimal_idx_pmpjpe]:.2f} mm)')
    axes[1].legend()

    # Plot 3: Acceleration Error
    axes[2].plot(confidence, acceleration, 'o-', linewidth=2, markersize=6, color='green', label='Fixed Confidence')
    axes[2].axhline(y=default_acceleration, color='red', linestyle='--', linewidth=2,
                   label=f'Default Confidence ({default_acceleration:.3f} mm/s²)')
    axes[2].axhline(y=runif_acceleration, color='blue', linestyle='--', linewidth=2, 
                   label=f'random uniform Confidence ({default_mpjpe:.2f} mm)')
    axes[2].set_xlabel('Fixed Confidence Value')
    axes[2].set_ylabel('Acceleration Error (mm/s²)')
    axes[2].set_title('Acceleration Error', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(0.05, 1.55)

    # Highlight optimal range for Acceleration
    optimal_idx_acc = np.argmin(acceleration)
    axes[2].plot(confidence[optimal_idx_acc], acceleration[optimal_idx_acc], 
                'go', markersize=10, markerfacecolor='lightgreen', markeredgecolor='green',
                label=f'Best Fixed: {confidence[optimal_idx_acc]} ({acceleration[optimal_idx_acc]:.3f} mm/s²)')
    axes[2].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the plot
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('confidence_analysis.pdf', bbox_inches='tight')
    print("Plots saved as 'confidence_analysis.png' and 'confidence_analysis.pdf'")

    plt.show()

def print_analysis():
    """Print detailed analysis of results"""

    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS RESULTS")
    print("="*60)

    # Data
    confidence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    mpjpe = [85.97, 78.16, 70.55, 62.16, 55.09, 51.24, 49.51, 48.66, 48.17, 47.93, 48.24, 51.06, 56.03, 62.78, 71.10]
    p_mpjpe = [69.20, 63.27, 57.43, 50.65, 44.72, 41.53, 40.22, 39.72, 39.55, 39.59, 40.03, 42.02, 45.02, 48.57, 52.42]
    acceleration = [1.163, 1.156, 1.121, 1.052, 0.975, 0.929, 0.925, 0.961, 1.037, 1.131, 1.204, 1.337, 1.508, 1.658, 1.793]

    # Default values
    default_mpjpe = 45.15
    default_p_mpjpe = 36.89
    default_acceleration = 0.841

    # Find optimal values
    best_mpjpe_idx = np.argmin(mpjpe)
    best_p_mpjpe_idx = np.argmin(p_mpjpe) 
    best_acc_idx = np.argmin(acceleration)

    print(f"\nDEFAULT CONFIDENCE PERFORMANCE:")
    print(f"  MPJPE:         {default_mpjpe:.2f} mm")
    print(f"  P-MPJPE:       {default_p_mpjpe:.2f} mm") 
    print(f"  Acceleration:  {default_acceleration:.3f} mm/s²")

    print(f"\nBEST FIXED CONFIDENCE PERFORMANCE:")
    print(f"  MPJPE:         {mpjpe[best_mpjpe_idx]:.2f} mm at confidence {confidence[best_mpjpe_idx]}")
    print(f"  P-MPJPE:       {p_mpjpe[best_p_mpjpe_idx]:.2f} mm at confidence {confidence[best_p_mpjpe_idx]}")
    print(f"  Acceleration:  {acceleration[best_acc_idx]:.3f} mm/s² at confidence {confidence[best_acc_idx]}")

    print(f"\nDEFAULT vs BEST FIXED DIFFERENCE:")
    print(f"  MPJPE:         {mpjpe[best_mpjpe_idx] - default_mpjpe:+.2f} mm ({(mpjpe[best_mpjpe_idx] - default_mpjpe)/default_mpjpe*100:+.1f}%)")
    print(f"  P-MPJPE:       {p_mpjpe[best_p_mpjpe_idx] - default_p_mpjpe:+.2f} mm ({(p_mpjpe[best_p_mpjpe_idx] - default_p_mpjpe)/default_p_mpjpe*100:+.1f}%)")
    print(f"  Acceleration:  {acceleration[best_acc_idx] - default_acceleration:+.3f} mm/s² ({(acceleration[best_acc_idx] - default_acceleration)/default_acceleration*100:+.1f}%)")

    print(f"\nKEY INSIGHTS:")
    print(f"  • Default confidence OUTPERFORMS all fixed values")
    print(f"  • Optimal fixed range: 0.7-1.0 for balanced performance")
    print(f"  • Performance degrades significantly with extreme values (<0.5 or >1.1)")
    print(f"  • Variable confidence contains valuable uncertainty information")

    print("="*60 + "\n")

def create_summary_table():
    """Create a formatted table of all results"""

    confidence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    mpjpe = [85.97, 78.16, 70.55, 62.16, 55.09, 51.24, 49.51, 48.66, 48.17, 47.93, 48.24, 51.06, 56.03, 62.78, 71.10]
    p_mpjpe = [69.20, 63.27, 57.43, 50.65, 44.72, 41.53, 40.22, 39.72, 39.55, 39.59, 40.03, 42.02, 45.02, 48.57, 52.42]
    acceleration = [1.163, 1.156, 1.121, 1.052, 0.975, 0.929, 0.925, 0.961, 1.037, 1.131, 1.204, 1.337, 1.508, 1.658, 1.793]

    print("\nDETAILED RESULTS TABLE:")
    print("-" * 70)
    print(f"{'Conf':>6} | {'MPJPE':>8} | {'P-MPJPE':>8} | {'Accel':>8} | {'Notes':>15}")
    print("-" * 70)

    for i, conf in enumerate(confidence):
        notes = ""
        if i == np.argmin(mpjpe):
            notes += "Best MPJPE"
        if i == np.argmin(p_mpjpe):
            notes += "Best P-MPJPE"
        if i == np.argmin(acceleration):
            notes += "Best Accel"

        print(f"{conf:>6.1f} | {mpjpe[i]:>8.2f} | {p_mpjpe[i]:>8.2f} | {acceleration[i]:>8.3f} | {notes:>15}")

    print("-" * 70)
    print(f"{'DEF':>6} | {45.15:>8.2f} | {36.89:>8.2f} | {0.841:>8.3f} | {'Default':>15}")
    print("-" * 70)

def main():
    """Main function to run all analysis"""
    print("Generating confidence analysis plots...")

    # Generate plots
    plot_confidence_analysis()

    # Print analysis
    print_analysis()

    # Create summary table
    create_summary_table()

    print("\nAnalysis complete! Check 'confidence_analysis.png' and 'confidence_analysis.pdf' for plots.")

if __name__ == "__main__":
    main()