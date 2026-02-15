%% Parameter file: LLC_resonant_Converter_v1_data.m

%% Model Parameters
Ts_Power= 50e-9;       % SPS Model sample time (s)   
%
%% LLC Data
Vdc_in=400;             % DC input voltage (V)
Vo_nom=400;             % DC nominal output voltage (V)
Po_nom=7.2e3;           % Nominal output power (W)
Lr=25e-6;               % Transformer leakage reactance (H)
Lm=150e-6;              % Magnetization inductance (H)
Cr=75e-9;               % Resonant capacitor (F)
Np=10; Ns=12; N=Np/Ns;  % Transformer ratio
Co=150e-6;              % Output filter (F)
Ro=22.22;               % Nominal load (Ohm)
Ron_FET=50e-3;          % FET resistance (ohms)
Ron_Diode=50e-3;        % FET resistance (ohms)
Vf=0.6;                 % Diode forward voltage (V)
%
%% LLC Control System Parameters
DT_LLC=400e-9;
Ts_Control=10e-6;
% Voltage Regulator (PI): 
Kp_LLC= 100;            % Proportional gain 
Ki_LLC= 150e3;          % Integral gain
fs_LCC_min= 120e3;      % Minimum switching frequency (Hz)
fs_LCC_max= 250e3;      % Maximum switching frequency (Hz)
%
%%
% Resonant tank data
Ro=Vo_nom^2/Po_nom      % LLC nominal output load (Ohm)
Rac=(8/pi^2)*N^2*Ro     % Equivalent primary AC resistance (Ohm)
Q=sqrt(Lr/Cr)/Rac       % Quality factor
fr=1/(2*pi*sqrt(Lr*Cr)) % Resonant frequency (Hz)
m=(Lr+Lm)/Lr            % Ratio of total primary reactance to resonant inductance (Lr) 
%
%%