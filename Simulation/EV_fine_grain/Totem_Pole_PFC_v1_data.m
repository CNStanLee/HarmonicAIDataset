%% Parameter file: Totem_Pole_PFC_v1_data.m

%% Model Parameters
Ts_Power= 50e-9;    % SPS Model sample time (s)
Ro=22.22;           % 7.2-kW nominal load (Ohms)
Ron_FET=50e-3;      % FET resistance (ohms)
Ron_Diode=50e-3;    % FET resistance (ohms)
Vf=0.6;             % Diode forward voltage (V)
      

%% PFC Data
Vnom_ac= 240;       % Nominal Ac voltage (V)
Fnom= 60;           % System frequency (Hz)
L_PFC= 600e-6;      % PFC Inductance (H)
RL_PFC= 20e-3;      % Inductance resistance (Ohm)
C_PFC= 2600e-6;     % Capacitance (F)

%% PFC Control System Parameters
Ts_Control= 10e-6;  % Control system time (s)
Fsw= 100e3;         % PWM Switching frequency (Hz)
DT_PFC=400e-9;

% Current Regulator (PR): 
Kp_I= 0.06;         % Proportional gain 
Kr_I= 2.5;          % Resonant gain
Zeta_I=0.2;         % Damping coefficient
Fr_I=Fnom*2;        % Resonant frequency (Hz)

% Voltage Regulator (PI): 
Kp_V= 0.25;        % Proportional gain 
Ki_V= 80;          % Integral gain
Limit_V= 100;      % Output limit
%
%%



