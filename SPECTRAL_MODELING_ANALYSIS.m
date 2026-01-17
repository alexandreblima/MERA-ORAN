%% ESTIMATE_SPECTRUM_YULEWALKER.M
% Parametric Spectral Estimation using Yule-Walker (Autocorrelation Method)
% Updated: Vertical scale [-50, 5] dB and Blue Welch baseline.

clear; clc; close all;

%% 1. CONFIGURATION
NFFT = 1024;
order = 80;       % AR Model Order for 5G signal representation
fs = 30.72e6;     % Standard 5G Sampling Frequency (30.72 MHz)

fprintf('Generating 30%% Load signal for Yule-Walker estimation...\n');

%% 2. SIGNAL GENERATION
% Simulating the sparse 5G signal (Wave6G project context)
[x, ~] = signal.generate_5g_iq('TrafficLoad', 0.3, 'Nfft', NFFT);

%% 3. SPECTRAL ESTIMATION
% --- Yule-Walker (Parametric) ---
fprintf('Computing Yule-Walker PSD (Order = %d)...\n', order);
[pxx_yw, f] = pyulear(x, order, NFFT, fs, 'centered');

% --- Welch (Non-Parametric) ---
[pxx_welch, ~] = pwelch(x, hamming(256), 128, NFFT, fs, 'centered');

%% 4. NORMALIZATION AND CONVERSION
% Normalize peaks to 0 dB for better comparison overlay
pxx_yw_db = 10*log10(pxx_yw / max(pxx_yw));
pxx_welch_db = 10*log10(pxx_welch / max(pxx_welch));

%% 5. PLOTTING
figure('Color', 'w', 'Position', [100, 100, 900, 500]);

% Plot Welch
plot(f/1e6, pxx_welch_db, 'k', 'LineWidth', 2, ...
    'DisplayName', 'Welch Estimate (Blue)');
hold on;

% Plot Yule-Walker in RED
plot(f/1e6, pxx_yw_db, 'm', 'LineWidth', 2, ...
    'DisplayName', sprintf('Yule-Walker AR (Order %d)', order));

grid on;
box on;
title({'Spectral Estimation Comparison', '5G Signal: Welch vs. Yule-Walker'}, ...
      'FontSize', 14, 'FontWeight', 'bold');
xlabel('Frequency (MHz)', 'FontSize', 12);
ylabel('Normalized PSD (dB)', 'FontSize', 12);

% --- Vertical Scale Adjustment ---
% Set limits between -55 dB and 5 dB
ylim([-55 5]); 
xlim([min(f)/1e6 max(f)/1e6]);

legend('Location', 'south', 'FontSize', 11);

%% 6. SAVING RESULTS (High-Res PNG and Vector PDF)
if ~exist('results', 'dir'), mkdir('results'); end

filename = 'results/yule_walker_spectral_estimation';

% Save as Vector PDF (Best for LaTeX)
exportgraphics(gcf, [filename '.pdf'], 'ContentType', 'vector');

% Save as High-Resolution PNG (300 DPI)
exportgraphics(gcf, [filename '.png'], 'Resolution', 300);

fprintf('Figures saved to results/ folder: \n 1. %s.pdf \n 2. %s.png\n', filename, filename);