%% PLOT_SPECTRAL_ANALYSIS.M
% Spectral Analysis Validation: Full Load vs. Partial Load
%
% Description:
%   This script generates the Power Spectral Density (PSD) comparison 
%   between a fully loaded 5G signal and a partially loaded one.
%   It serves as visual proof of the spectral sparsity that MERA exploits.
%
% Output:
%   Saves high-resolution PNG and Vector PDF to the 'results/' folder.
%
% Author: Alexandre Barbosa de Lima
% Date: 2026

clear; clc; close all;

%% 1. CONFIGURATION
N_SYMBOLS = 200;     % Number of OFDM symbols for smooth PSD estimation
NFFT = 1024;         % FFT Size
FS = 30.72e6;        % Sampling Rate (30.72 MHz)

fprintf('Generating signals for spectral analysis...\n');

%% 2. SIGNAL GENERATION

% --- Scenario A: Full Load (~90% occupied) ---
% Acts as the baseline "saturated" spectrum.
sig_full = [];
for i = 1:N_SYMBOLS
    [s, ~] = signal.generate_5g_iq('TrafficLoad', 0.9, 'Nfft', NFFT);
    sig_full = [sig_full; s]; %#ok<AGROW>
end

% --- Scenario B: Partial Load (30% - Proposed Scenario) ---
% Represents the sparse traffic typical of the "Wave6G" use case.
sig_part = [];
for i = 1:N_SYMBOLS
    [s, ~] = signal.generate_5g_iq('TrafficLoad', 0.3, 'Nfft', NFFT);
    sig_part = [sig_part; s]; %#ok<AGROW>
end

%% 3. POWER SPECTRAL DENSITY (PSD) CALCULATION
% Welch's method is used to reduce variance and smooth the plot.
window = hamming(NFFT);
noverlap = NFFT/2;

fprintf('Calculating PSD...\n');
[pxx_full, f] = pwelch(sig_full, window, noverlap, NFFT, FS, 'centered');
[pxx_part, ~] = pwelch(sig_part, window, noverlap, NFFT, FS, 'centered');

% Convert to dB scale
pxx_full_db = 10*log10(pxx_full);
pxx_part_db = 10*log10(pxx_part);

% Normalize peak to 0 dB for easier comparison
peak_val = max(pxx_full_db);
pxx_full_db = pxx_full_db - peak_val;
pxx_part_db = pxx_part_db - peak_val;

%% 4. PLOTTING
figure('Position', [100, 100, 900, 500], 'Color', 'w');

% --- Plot 1: Full Load (Reference / Baseline) ---
% Style: SOLID BLACK Line
h1 = plot(f/1e6, pxx_full_db, '-', 'LineWidth', 1.5, 'Color', 'k'); 
hold on;

% --- Plot 2: Partial Load (MERA / Interest) ---
% Style: SOLID RED Line
h2 = plot(f/1e6, pxx_part_db, '-', 'LineWidth', 2.0, 'Color', 'r'); 

grid on;
box on; % Adds a border around the plot area

% --- Labels and Titles ---
xlabel('Frequency (MHz)', 'FontSize', 12, 'FontName', 'Arial');
ylabel('Power Spectral Density (dB)', 'FontSize', 12, 'FontName', 'Arial');
title({'Spectral Sparsity Validation', 'Comparison: Saturated vs. Sparse Traffic'}, ...
      'FontSize', 14, 'FontName', 'Arial', 'FontWeight', 'bold');

% --- Legend ---
legend([h2, h1], ...
    {'Partial Load (30%) - Target for Compression', 'Full Load (Reference)'}, ...
    'Location', 'south', 'FontSize', 11);

% --- Axis Limits (UPDATED) ---
% Ajustado conforme solicitado: -40dB a +10dB
ylim([-40 10]); 
xlim([min(f)/1e6 max(f)/1e6]);

%% 5. SAVING
% Ensure directory exists
if ~exist('results', 'dir'), mkdir('results'); end

filename = 'results/spectral_proof';

% Save as Vector PDF (Best for LaTeX/Papers)
exportgraphics(gcf, [filename '.pdf'], 'ContentType', 'vector');

% Save as High-Res PNG (For Slides)
exportgraphics(gcf, [filename '.png'], 'Resolution', 300);

fprintf('Figures saved successfully with Y-limits [-40, 10] dB.\n');