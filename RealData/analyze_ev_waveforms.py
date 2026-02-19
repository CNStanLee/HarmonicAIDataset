"""
Analyze EV charging waveform data from EV-CPW Dataset.
For each vehicle, load waveform data, compute harmonics and THD,
and generate time domain and frequency domain plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def read_waveform_csv(filepath):
    """
    Read waveform CSV file from EV-CPW dataset.
    
    Returns:
        time (ms), voltage (V), current (A), samples_per_cycle, sample_period_us
    """
    with open(filepath, 'r') as f:
        # Read metadata
        line1 = f.readline()  # Trigger_Date
        line2 = f.readline()  # Trigger_Time
        line3 = f.readline()  # Samples_Per_Cycle
        line4 = f.readline()  # Microseconds_Per_Sample
        
        # Extract metadata
        samples_per_cycle = int(line3.split(',')[1])
        us_per_sample = float(line4.split(',')[1])
    
    # Read waveform data
    df = pd.read_csv(filepath, skiprows=4)
    time_ms = df.iloc[:, 0].values
    voltage = df.iloc[:, 1].values
    current = df.iloc[:, 2].values
    
    return time_ms, voltage, current, samples_per_cycle, us_per_sample


def compute_fft_harmonics(signal, samples_per_cycle, num_harmonics=20):
    """
    Compute FFT and extract harmonic magnitudes.
    
    Returns:
        frequencies, magnitudes, harmonic_values (dict)
    """
    # FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    
    # Normalize magnitude (single-sided spectrum)
    magnitude = 2.0 * np.abs(fft_result) / len(signal)
    magnitude[0] = magnitude[0] / 2.0  # DC component
    
    # Extract positive frequencies
    positive_idx = freqs >= 0
    freqs_pos = freqs[positive_idx]
    mag_pos = magnitude[positive_idx]
    
    # Extract harmonics
    # Find fundamental frequency (largest peak in low frequency range)
    low_freq_mask = (freqs_pos > 0) & (freqs_pos < 0.2)
    if np.any(low_freq_mask):
        fundamental_idx = np.argmax(mag_pos[low_freq_mask])
        fundamental_freq_idx = np.where(low_freq_mask)[0][fundamental_idx]
    else:
        fundamental_freq_idx = 1
    
    harmonics = {}
    for h in range(1, num_harmonics + 1):
        harmonic_idx = fundamental_freq_idx * h
        if harmonic_idx < len(mag_pos):
            harmonics[h] = mag_pos[harmonic_idx]
        else:
            harmonics[h] = 0.0
    
    return freqs_pos, mag_pos, harmonics


def compute_thd(harmonics, fundamental_idx=1):
    """
    Compute Total Harmonic Distortion (THD).
    
    THD = sqrt(sum(H_i^2 for i > 1)) / H_1 * 100%
    """
    if fundamental_idx not in harmonics or harmonics[fundamental_idx] == 0:
        return 0.0
    
    H1 = harmonics[fundamental_idx]
    harmonic_sum_squared = sum(harmonics[h]**2 for h in harmonics if h > fundamental_idx)
    thd = np.sqrt(harmonic_sum_squared) / H1 * 100
    
    return thd


def plot_waveform_analysis(time_ms, voltage, current, vehicle_name, 
                           waveform_name, output_dir, harmonics_v, harmonics_i,
                           freqs_v, mag_v, freqs_i, mag_i, thd_v, thd_i):
    """
    Create combined plot with time domain and frequency domain for both voltage and current.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Convert time to cycles (assuming 60Hz)
    # Find one complete cycle
    cycle_duration_ms = 1000.0 / 60.0  # ~16.67 ms for 60Hz
    num_points_per_cycle = int(len(time_ms) * cycle_duration_ms / (time_ms[-1] - time_ms[0]))
    num_points_per_cycle = min(num_points_per_cycle, len(time_ms))
    
    time_cycles = time_ms / cycle_duration_ms
    
    # (a) Voltage time domain
    ax = axes[0, 0]
    ax.plot(time_cycles[:num_points_per_cycle], voltage[:num_points_per_cycle], 
            linewidth=1.2, color='C0')
    ax.set_xlabel("Time (cycles)", fontsize=11)
    ax.set_ylabel("Voltage (V)", fontsize=11)
    ax.set_title(f"(a) Voltage Waveform - {vehicle_name}\nTHD = {thd_v:.2f}%", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # (b) Voltage frequency domain
    ax = axes[0, 1]
    harmonic_indices = list(range(1, 21))
    harmonic_mags = [harmonics_v.get(h, 0) for h in harmonic_indices]
    ax.bar(harmonic_indices, harmonic_mags, width=0.6, color='C0', alpha=0.7)
    ax.set_xlabel("Harmonic Order", fontsize=11)
    ax.set_ylabel("Magnitude (V)", fontsize=11)
    ax.set_title(f"(b) Voltage Harmonics\nH1={harmonics_v[1]:.1f}V", fontsize=12)
    ax.set_xticks(harmonic_indices)
    ax.grid(True, alpha=0.3, axis='y')
    
    # (c) Current time domain
    ax = axes[1, 0]
    ax.plot(time_cycles[:num_points_per_cycle], current[:num_points_per_cycle], 
            linewidth=1.2, color='C1')
    ax.set_xlabel("Time (cycles)", fontsize=11)
    ax.set_ylabel("Current (A)", fontsize=11)
    ax.set_title(f"(c) Current Waveform - {vehicle_name}\nTHD = {thd_i:.2f}%", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # (d) Current frequency domain
    ax = axes[1, 1]
    harmonic_mags_i = [harmonics_i.get(h, 0) for h in harmonic_indices]
    ax.bar(harmonic_indices, harmonic_mags_i, width=0.6, color='C1', alpha=0.7)
    ax.set_xlabel("Harmonic Order", fontsize=11)
    ax.set_ylabel("Magnitude (A)", fontsize=11)
    ax.set_title(f"(d) Current Harmonics\nH1={harmonics_i[1]:.1f}A", fontsize=12)
    ax.set_xticks(harmonic_indices)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / f"{waveform_name}_analysis.svg"
    plt.savefig(output_path, format='svg')
    plt.close()
    
    return str(output_path)


def analyze_vehicle(vehicle_dir, base_output_dir):
    """
    Analyze all waveforms for a single vehicle.
    
    Returns:
        vehicle_name, thd_results (list of dicts)
    """
    vehicle_name = vehicle_dir.name
    waveforms_dir = vehicle_dir / "Waveforms"
    
    if not waveforms_dir.exists():
        print(f"  Warning: No Waveforms directory found for {vehicle_name}")
        return vehicle_name, []
    
    # Create output directory
    output_dir = vehicle_dir / "Analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Get all waveform files
    waveform_files = sorted(waveforms_dir.glob("Waveform_*.csv"))
    
    if len(waveform_files) == 0:
        print(f"  Warning: No waveform files found for {vehicle_name}")
        return vehicle_name, []
    
    print(f"\n{vehicle_name}:")
    print(f"  Found {len(waveform_files)} waveform files")
    
    thd_results = []
    
    # Analyze first 3 waveforms (or all if less than 3)
    for i, waveform_file in enumerate(waveform_files[:3]):
        waveform_name = waveform_file.stem
        print(f"  Processing {waveform_name}...")
        
        try:
            # Read waveform data
            time_ms, voltage, current, samples_per_cycle, us_per_sample = read_waveform_csv(waveform_file)
            
            # Compute FFT and harmonics
            freqs_v, mag_v, harmonics_v = compute_fft_harmonics(voltage, samples_per_cycle)
            freqs_i, mag_i, harmonics_i = compute_fft_harmonics(current, samples_per_cycle)
            
            # Compute THD
            thd_v = compute_thd(harmonics_v)
            thd_i = compute_thd(harmonics_i)
            
            # Plot
            plot_path = plot_waveform_analysis(
                time_ms, voltage, current, vehicle_name, waveform_name,
                output_dir, harmonics_v, harmonics_i,
                freqs_v, mag_v, freqs_i, mag_i, thd_v, thd_i
            )
            
            # Store results
            thd_results.append({
                'waveform': waveform_name,
                'thd_voltage': thd_v,
                'thd_current': thd_i,
                'h1_voltage': harmonics_v[1],
                'h1_current': harmonics_i[1],
                'h3_voltage': harmonics_v.get(3, 0),
                'h3_current': harmonics_i.get(3, 0),
                'h5_voltage': harmonics_v.get(5, 0),
                'h5_current': harmonics_i.get(5, 0),
                'h7_voltage': harmonics_v.get(7, 0),
                'h7_current': harmonics_i.get(7, 0),
                'plot_path': plot_path
            })
            
            print(f"    THD(V): {thd_v:.2f}%, THD(I): {thd_i:.2f}%")
            print(f"    Saved: {plot_path}")
            
        except Exception as e:
            print(f"    Error processing {waveform_name}: {e}")
            continue
    
    return vehicle_name, thd_results


def main():
    # Dataset path
    dataset_path = Path(__file__).parent / "EV_CPW" / "EV_CPW" / "EV-CPW Dataset"
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    print("=" * 80)
    print("EV-CPW Dataset Waveform Analysis")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all vehicle directories
    vehicle_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    vehicle_dirs = sorted(vehicle_dirs, key=lambda x: x.name)
    
    print(f"\nFound {len(vehicle_dirs)} vehicles")
    
    # Analyze each vehicle
    all_results = {}
    
    for vehicle_dir in vehicle_dirs:
        vehicle_name, thd_results = analyze_vehicle(vehicle_dir, dataset_path)
        all_results[vehicle_name] = thd_results
    
    # Generate summary log
    print("\n" + "=" * 80)
    print("SUMMARY: THD Analysis Results")
    print("=" * 80)
    
    log_path = dataset_path / "THD_Analysis_Summary.txt"
    with open(log_path, 'w') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("EV-CPW Dataset - THD Analysis Summary\n")
        log_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 80 + "\n\n")
        
        for vehicle_name in sorted(all_results.keys()):
            results = all_results[vehicle_name]
            
            if len(results) == 0:
                summary = f"\n{vehicle_name}:\n  No valid waveform data\n"
            else:
                # Compute statistics
                thd_v_values = [r['thd_voltage'] for r in results]
                thd_i_values = [r['thd_current'] for r in results]
                
                avg_thd_v = np.mean(thd_v_values)
                avg_thd_i = np.mean(thd_i_values)
                
                summary = f"\n{vehicle_name}:\n"
                summary += f"  Number of waveforms analyzed: {len(results)}\n"
                summary += f"  Average THD (Voltage): {avg_thd_v:.2f}%\n"
                summary += f"  Average THD (Current): {avg_thd_i:.2f}%\n"
                summary += f"  THD Range (Voltage): {min(thd_v_values):.2f}% - {max(thd_v_values):.2f}%\n"
                summary += f"  THD Range (Current): {min(thd_i_values):.2f}% - {max(thd_i_values):.2f}%\n"
                
                # Add detailed results
                summary += "\n  Detailed Results:\n"
                for r in results:
                    summary += f"    {r['waveform']}:\n"
                    summary += f"      Voltage: THD={r['thd_voltage']:.2f}%, H1={r['h1_voltage']:.1f}V, H3={r['h3_voltage']:.1f}V, H5={r['h5_voltage']:.1f}V, H7={r['h7_voltage']:.1f}V\n"
                    summary += f"      Current: THD={r['thd_current']:.2f}%, H1={r['h1_current']:.1f}A, H3={r['h3_current']:.1f}A, H5={r['h5_current']:.1f}A, H7={r['h7_current']:.1f}A\n"
            
            print(summary)
            log_file.write(summary + "\n")
    
    print("\n" + "=" * 80)
    print(f"Analysis complete!")
    print(f"Summary log saved to: {log_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
