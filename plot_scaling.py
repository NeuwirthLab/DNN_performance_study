import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists('scaling_metrics.csv'):
    print("Error: scaling_metrics.csv not found! Run the scaling_test_amd.slurm job first.")
    exit(1)

df = pd.read_csv('scaling_metrics.csv')

# Filter into strong and weak scaling datasets
df_strong = df[df['TestType'] == 'Strong'].sort_values('GPUs')
df_weak = df[df['TestType'] == 'Weak'].sort_values('GPUs')

# Calculate Ideal Scaling curves
# Strong Scaling Ideal: Throughput should scale linearly with GPUs
if not df_strong.empty:
    base_strong_tp = df_strong[df_strong['GPUs'] == 1]['ImgPerSec'].values[0]
    df_strong['IdealImgPerSec'] = df_strong['GPUs'] * base_strong_tp

# Weak Scaling Ideal: Throughput should scale linearly with GPUs (since batch size scales linearly)
if not df_weak.empty:
    base_weak_tp = df_weak[df_weak['GPUs'] == 1]['ImgPerSec'].values[0]
    df_weak['IdealImgPerSec'] = df_weak['GPUs'] * base_weak_tp

plt.figure(figsize=(14, 6))

# Plot 1: Strong Scaling
plt.subplot(1, 2, 1)
if not df_strong.empty:
    plt.plot(df_strong['GPUs'], df_strong['ImgPerSec'], marker='o', linewidth=2, label='Measured (AMD MI300X)')
    plt.plot(df_strong['GPUs'], df_strong['IdealImgPerSec'], linestyle='--', color='gray', label='Ideal Linear Scaling')
    
    plt.title('Strong Scaling (Fixed Global Batch = 256)', fontsize=14)
    plt.xlabel('Number of GPUs', fontsize=12)
    plt.ylabel('Throughput (Images / Second)', fontsize=12)
    plt.xticks(df_strong['GPUs'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

# Plot 2: Weak Scaling
plt.subplot(1, 2, 2)
if not df_weak.empty:
    plt.plot(df_weak['GPUs'], df_weak['ImgPerSec'], marker='s', color='orange', linewidth=2, label='Measured (AMD MI300X)')
    plt.plot(df_weak['GPUs'], df_weak['IdealImgPerSec'], linestyle='--', color='gray', label='Ideal Linear Scaling')
    
    plt.title('Weak Scaling (Fixed Local Batch = 64)', fontsize=14)
    plt.xlabel('Number of GPUs', fontsize=12)
    plt.ylabel('Throughput (Images / Second)', fontsize=12)
    plt.xticks(df_weak['GPUs'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

plt.tight_layout()
plt.savefig('scaling_plots.png', dpi=300, bbox_inches='tight')
print("Successfully generated 'scaling_plots.png'!")
