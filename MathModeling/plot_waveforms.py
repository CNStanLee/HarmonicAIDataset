import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_waveforms(npz_files, output_dir="preview"):
    """
    Load waveform data from .npz files and plot all in one figure with subplots.
    
    Args:
        npz_files: List of paths to .npz files or single path
        output_dir: Directory to save plots
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Handle single file or list of files
    if isinstance(npz_files, str):
        npz_files = [npz_files]
    
    # Load all data
    all_data = []
    for npz_file in npz_files:
        case_name = Path(npz_file).stem
        print(f"Loading {case_name}...")
        data = np.load(npz_file)
        signals = data["signals"]
        labels = data["labels"]
        
        print(f"  Signals shape: {signals.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Min signal value: {np.min(signals):.6f}")
        print(f"  Max signal value: {np.max(signals):.6f}")
        
        # Calculate THD
        H1 = labels[:, 0]
        H3 = labels[:, 1]
        H5 = labels[:, 2]
        H7 = labels[:, 3]
        thd = np.sqrt(H3**2 + H5**2 + H7**2) / H1 * 100
        print(f"  THD range: {np.min(thd):.2f}% - {np.max(thd):.2f}%")
        
        all_data.append({
            'name': case_name,
            'signals': signals,
            'labels': labels,
            'thd': thd
        })
    
    # Create combined plot with 2x2 subplots
    num_cases = len(all_data)
    samples_per_cycle = all_data[0]['signals'].shape[1]
    
    fig, axes = plt.subplots(num_cases, 2, figsize=(16, 7*num_cases))
    if num_cases == 1:
        axes = axes.reshape(1, -1)
    
    sublabels = ['(a)', '(b)', '(c)', '(d)']
    sublabel_idx = 0
    
    for case_idx, case_data in enumerate(all_data):
        case_name = case_data['name']
        signals = case_data['signals']
        labels = case_data['labels']
        thd = case_data['thd']
        
        # Sample 0: Time domain
        signal = signals[0]
        label = labels[0]
        time = np.arange(len(signal)) / samples_per_cycle
        thd_val = thd[0]
        
        ax = axes[case_idx, 0]
        ax.plot(time, signal, linewidth=1.5, color='C0')
        ax.set_xlabel("Time (cycles)", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_title(f"{case_name} Time Domain\nH1={label[0]:.1f}, H3={label[1]:.1f}, H5={label[2]:.1f}, H7={label[3]:.1f}, THD={thd_val:.2f}%", fontsize=12)
        ax.grid(True, alpha=0.3)
        # Add sublabel at the bottom
        ax.text(0.5, -0.25, sublabels[sublabel_idx], transform=ax.transAxes, 
                ha='center', va='top', fontsize=14, fontweight='bold')
        sublabel_idx += 1
        
        # Sample 0: Frequency domain
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1/samples_per_cycle)
        magnitude = np.abs(fft_result)
        
        # Normalize FFT magnitude
        magnitude = 2.0 * magnitude / len(signal)
        magnitude[0] = magnitude[0] / 2.0
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_mag = magnitude[:len(magnitude)//2]
        
        ax = axes[case_idx, 1]
        ax.stem(positive_freqs, positive_mag, basefmt=' ', linefmt='C0-', markerfmt='C0o')
        ax.set_xlabel("Frequency (multiples of fundamental)", fontsize=11)
        ax.set_ylabel("Magnitude", fontsize=11)
        ax.set_title(f"{case_name} Frequency Domain\nH1={label[0]:.1f}, H3={label[1]:.1f}, H5={label[2]:.1f}, H7={label[3]:.1f}, THD={thd_val:.2f}%", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(20, len(positive_freqs))])
        # Add sublabel at the bottom
        ax.text(0.5, -0.25, sublabels[sublabel_idx], transform=ax.transAxes, 
                ha='center', va='top', fontsize=14, fontweight='bold')
        sublabel_idx += 1
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.subplots_adjust(hspace=0.4)
    combined_file = os.path.join(output_dir, "combined_waveforms.svg")
    plt.savefig(combined_file, format='svg')
    print(f"  Saved: {combined_file}")
    plt.close()
    



if __name__ == "__main__":
    # Get the current directory
    script_dir = Path(__file__).parent
    
    # Load and plot caseA1 and caseA2
    caseA1_path = script_dir / "caseA1.npz"
    caseA2_path = script_dir / "caseA2.npz"
    output_dir = script_dir / "preview"
    
    print("=" * 60)
    print("Waveform Analysis")
    print("=" * 60)
    
    npz_files = []
    if caseA1_path.exists():
        npz_files.append(str(caseA1_path))
    else:
        print(f"Warning: {caseA1_path} not found")
    
    if caseA2_path.exists():
        npz_files.append(str(caseA2_path))
    else:
        print(f"Warning: {caseA2_path} not found")
    
    if npz_files:
        plot_waveforms(npz_files, str(output_dir))
    
    print()
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)
