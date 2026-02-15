# HarmonicAIDataset
## Math Modeling
The synthetic harmonic waveform generation following the settings in [1] to train the basic harmonic estimation model.

### Case A1
Synthetic waveform dataset with 64 samples per cycle (fs=3840 Hz). Each cycle includes 1/3/5/7 harmonics with nominal amplitudes 100/30/20/15, phases 152/35/0/0 degrees, +/-0.5% fundamental frequency deviation, +/-1% amplitude jitter, and 26 dB Gaussian white noise. Labels store per-cycle amplitudes for the 1/3/5/7 harmonics.

### Case A2
Same as Case A1, plus two interharmonics at 84.4 Hz (A=10, phase 0) and 385.6 Hz (A=5, phase 0). Labels still contain only the 1/3/5/7 harmonic amplitudes.

## Real Data
### EV-CPW Charging Data
The dataset sorted from [2], which comprises of charging profiles and high-resolution current/voltage AC waveforms for 12 different EV's, including popular battery EV's and plug-in hybrid EV's.

Our analysis reveals that harmonic distortion (THD) is a significant issue during single-phase charging of EVs, especially at low power outputs, with some models experiencing THD levels reaching as high as 15%.

### Collected 7kw EV charging data
In progress, the data will be collected and sorted by 28/02/2026.
### Collected 7kw PV station data
In progress, the data will be collected and sorted by 28/02/2026.
## Simulation
All simulation models were performed using Simulink and fitted and calibrated using real data or parameters.

### EV modeling
In this modeling, we have compiled three 7-kw simplified single-phase EV charging models.

### PV fine-grained modeling
In this modeling, we have compiled a fine-grained simulation model of a 100kW solar charging array.

### EV fine-grained modeling
In this modeling, the rectifier and DC-DC stages of the EV are modeled in a fine-grained manner to support fine-grained monitoring and control.

### EV-PV fine-grained co-modeling
Fine-grained PV and EV models were integrated into a unified model for simulating the grid topology of energy communities to analyze harmonic conditions and for subsequent HIL simulations and harmonic control.

## References
[1] Li, Congcong, et al. "Broad learning system using rectified adaptive moment estimation for harmonic detection and analysis." IEEE Transactions on Industrial Electronics 71.3 (2023): 2873-2882.

[2] Ziyat, Isla, et al. "EV charging profiles and waveforms dataset (EV-CPW) and associated power quality analysis." IEEE Access 11 (2023): 138445-138456.

## Acknowledgments
- This work was supported by the Sustainable Energy Authority of Ireland under Grant number 24/RDD/1170.
- Acknowledgements to Cleanwatts Digital and Coalesce for providing the opportunity to engage with the energy community and collect harmonic data from EV and PV sites.