%% MAIN_EXPERIMENT.M
% Rate-Distortion Experiment for 5G Fronthaul Compression
%
% This script compares three compression approaches:
%   1. Standard CPRI (Scalar Quantization) - Baseline
%   2. Daubechies-4 (Fixed Wavelet) - Baseline
%   3. Adaptive MERA (Learned Unitary Transform) - Proposed Method
%
% Key Features:
%   - Simulates 5G traffic with partial load (30%) to create spectral sparsity.
%   - Trains MERA filters on the Unitary Manifold.
%   - Implements Dynamic Bit Allocation based on channel energy.
%
% Author: Alexandre Barbosa de Lima
% Date: 2026

clear; clc; close all;

%% ==================== 1. EXPERIMENT PARAMETERS ====================
fprintf('==============================================\n');
fprintf('  Fronthaul Compression: MERA vs Baselines\n');
fprintf('==============================================\n\n');

% Dataset Configuration
NUM_TRAIN = 100;        % Number of OFDM symbols for training
NUM_TEST = 50;          % Number of OFDM symbols for testing/evaluation

% MERA Architecture
M = 4;                  % Number of channels (decimation factor)
K_VALUES = [0, 2, 4];   % Filter depths to test (K=0 is block transform, K>0 is lapped)

% Compression Targets
BITS_RANGE = 3:8;       % Target average bits per sample (I or Q)

% Training Hyperparameters
NUM_EPOCHS = 50;        % Training iterations
LEARNING_RATE = 0.1;    % Learning rate for Riemannian optimization

%% ==================== 2. DATA GENERATION ====================
% We simulate a "Low Traffic" scenario (30% Load).
% This creates a band-limited signal, which MERA learns to exploit.

fprintf('Generating Training Data (Traffic Load: 30%%)...\n');
trainData = cell(NUM_TRAIN, 1);
for i = 1:NUM_TRAIN
    % Generate OFDM symbol with 30% active subcarriers
    [iq_signal, ~] = signal.generate_5g_iq('TrafficLoad', 0.3); 
    trainData{i} = iq_signal;
end

fprintf('Generating Test Data...\n');
testData = cell(NUM_TEST, 1);
for i = 1:NUM_TEST
    [iq_signal, ~] = signal.generate_5g_iq('TrafficLoad', 0.3);
    testData{i} = iq_signal;
end
fprintf('Data generation complete.\n');

%% ==================== 3. INITIALIZE RESULTS STRUCTURE ====================
numBits = length(BITS_RANGE);
results = struct();
results.bits = BITS_RANGE;
% Storage for EVM (Error Vector Magnitude) and SNR results
results.cpri = struct('evm', zeros(numBits, 1), 'snr', zeros(numBits, 1));
results.db4  = struct('evm', zeros(numBits, 1), 'snr', zeros(numBits, 1));

for K = K_VALUES
    fieldName = sprintf('mera_K%d', K);
    results.(fieldName) = struct('evm', zeros(numBits, 1), 'snr', zeros(numBits, 1));
end

%% ==================== 4. BASELINE: STANDARD CPRI ====================
fprintf('\n=== Standard CPRI (Scalar Quantization) ===\n');
for b = 1:numBits
    nbits = BITS_RANGE(b);
    evms = zeros(NUM_TEST, 1);
    snrs = zeros(NUM_TEST, 1);
    
    for i = 1:NUM_TEST
        x = testData{i};
        % CPRI applies uniform quantization directly to time samples
        [xhat, ~] = baselines.cpri_standard(x, nbits);
        evms(i) = metrics.compute_evm(x, xhat);
        snrs(i) = metrics.compute_snr(x, xhat);
    end
    
    results.cpri.evm(b) = mean(evms);
    results.cpri.snr(b) = mean(snrs);
    fprintf('  %d bits: EVM = %5.2f%%, SNR = %5.1f dB\n', ...
        nbits, results.cpri.evm(b), results.cpri.snr(b));
end

%% ==================== 5. BASELINE: DAUBECHIES-4 ====================
fprintf('\n=== Daubechies-4 (Fixed Wavelet) ===\n');
for b = 1:numBits
    nbits = BITS_RANGE(b);
    evms = zeros(NUM_TEST, 1);
    snrs = zeros(NUM_TEST, 1);
    
    for i = 1:NUM_TEST
        x = testData{i};
        % Uses standard discrete wavelet transform
        [xhat, ~] = baselines.daubechies_compress(x, nbits, 'db4');
        evms(i) = metrics.compute_evm(x, xhat);
        snrs(i) = metrics.compute_snr(x, xhat);
    end
    
    results.db4.evm(b) = mean(evms);
    results.db4.snr(b) = mean(snrs);
    fprintf('  %d bits: EVM = %5.2f%%, SNR = %5.1f dB\n', ...
        nbits, results.db4.evm(b), results.db4.snr(b));
end

%% ==================== 6. PROPOSED: ADAPTIVE MERA ====================
for K = K_VALUES
    fprintf('\n=== MERA Architecture (Channels M=%d, Depth K=%d) ===\n', M, K);
    fieldName = sprintf('mera_K%d', K);
    
    % --- A. Training Phase ---
    % Train the filter bank to maximize energy concentration (sparsity)
    fprintf('  Training filters on Unitary Manifold...\n');
    fb = mera.FilterBank(M, K, 'random');
    
    % Using L1 or Energy Concentration Loss
    lossHistory = mera.train(fb, trainData, ...
        'NumEpochs', NUM_EPOCHS, ...
        'LearningRate', LEARNING_RATE, ...
        'BatchSize', 32, ...
        'Verbose', false);
    
    fprintf('  Final Loss: %.4f (Optimized)\n', lossHistory(end));

    % --- B. Testing Phase (Rate-Distortion Loop) ---
    for b = 1:numBits
        nbits_target = BITS_RANGE(b);
        evms = zeros(NUM_TEST, 1);
        snrs = zeros(NUM_TEST, 1);
        
        for i = 1:NUM_TEST
            x = testData{i};
            
            % 1. Analysis: Transform input to latent space (coefficients)
            Y = fb.analyze(x);
            
            % 2. Bit Allocation: Determine bits per channel based on energy
            energies = mean(abs(Y).^2, 2);
            bits_vec = quantizer.allocate_bits(energies, nbits_target, M);
            
            % 3. Quantization: Apply uniform quantization with peak scaling
            Yhat = zeros(size(Y));
            for ch = 1:M
                nb = bits_vec(ch);
                if nb > 0
                    % Peak scaling prevents clipping of OFDM peaks
                    scale_factor = max(abs(Y(ch,:))) + 1e-6;
                    Yq = quantizer.uniform(Y(ch,:), nb, scale_factor);
                    Yhat(ch,:) = quantizer.uniform_inverse(Yq, nb, scale_factor);
                else
                    % Discard channel (noise removal)
                    Yhat(ch,:) = 0; 
                end
            end
            
            % 4. Synthesis: Reconstruct signal
            xhat = fb.synthesize(Yhat);
            if length(xhat) > length(x), xhat = xhat(1:length(x)); end
            
            evms(i) = metrics.compute_evm(x, xhat);
            snrs(i) = metrics.compute_snr(x, xhat);
        end
        
        results.(fieldName).evm(b) = mean(evms);
        results.(fieldName).snr(b) = mean(snrs);
        fprintf('  Rate %d bits: EVM = %5.2f%%, SNR = %5.1f dB\n', ...
            nbits_target, results.(fieldName).evm(b), results.(fieldName).snr(b));
    end
end

%% ==================== 7. PLOTTING RESULTS ====================
fprintf('\nGenerating Rate-Distortion Plots...\n');
figure('Position', [100, 100, 1000, 400], 'Color', 'w');
colors = lines(2 + length(K_VALUES));

% --- Subplot 1: EVM (Lower is Better) ---
subplot(1, 2, 1); hold on; grid on; box on;
plot(BITS_RANGE, results.cpri.evm, '-o', 'LineWidth', 2, 'Color', 'k', 'DisplayName', 'CPRI (Baseline)');
plot(BITS_RANGE, results.db4.evm, '--s', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], 'DisplayName', 'Daubechies-4');
for idx = 1:length(K_VALUES)
    K = K_VALUES(idx);
    fieldName = sprintf('mera_K%d', K);
    plot(BITS_RANGE, results.(fieldName).evm, '-^', 'LineWidth', 2, ...
        'Color', colors(2+idx,:), 'MarkerFaceColor', colors(2+idx,:), ...
        'DisplayName', sprintf('MERA K=%d', K));
end
xlabel('Rate (bits/sample)', 'FontSize', 12); 
ylabel('EVM (%)', 'FontSize', 12);
title('Rate-Distortion: EVM (Lower is Better)', 'FontSize', 14);
legend('Location', 'northeast'); set(gca, 'YScale', 'log');

% --- Subplot 2: SNR (Higher is Better) ---
subplot(1, 2, 2); hold on; grid on; box on;
plot(BITS_RANGE, results.cpri.snr, '-o', 'LineWidth', 2, 'Color', 'k', 'DisplayName', 'CPRI');
plot(BITS_RANGE, results.db4.snr, '--s', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], 'DisplayName', 'Daubechies-4');
for idx = 1:length(K_VALUES)
    K = K_VALUES(idx);
    fieldName = sprintf('mera_K%d', K);
    plot(BITS_RANGE, results.(fieldName).snr, '-^', 'LineWidth', 2, ...
        'Color', colors(2+idx,:), 'MarkerFaceColor', colors(2+idx,:), ...
        'DisplayName', sprintf('MERA K=%d', K));
end
xlabel('Rate (bits/sample)', 'FontSize', 12); 
ylabel('SNR (dB)', 'FontSize', 12);
title('Rate-Distortion: SNR (Higher is Better)', 'FontSize', 14);
legend('Location', 'southeast');

% --- Saving Figures (Section 7) ---
filename_rd = 'results/rate_distortion';

% 1. PNG (high resolution - 300 DPI) slides/fast visualization
exportgraphics(gcf, [filename_rd '.png'], 'Resolution', 300);

% 2. PDF (vectorial) 
exportgraphics(gcf, [filename_rd '.pdf'], 'ContentType', 'vector');

fprintf('Figures R-D saved as PNG and PDF.\n');

%% ==================== 8. CONSTELLATION VISUALIZATION ====================
fprintf('Generating Constellation Plot (Visual Proof)...\n');
figure('Position', [100, 100, 1200, 400], 'Color', 'w');

% Select a sample signal
x_plot = testData{1};
nbits_plot = 4; % Critical comparison point

% 1. CPRI Output
[x_cpri, ~] = baselines.cpri_standard(x_plot, nbits_plot);

% 2. MERA Output (Using the last trained FB, assumed to be K=4 or re-instantiate)
% Ideally, we use the best model found. Here we simulate the K=2 flow for plotting.
fb_vis = mera.FilterBank(4, 2, 'random');
mera.train(fb_vis, trainData(1:50), 'NumEpochs', 20, 'Verbose', false); % Quick retrain
Y = fb_vis.analyze(x_plot);
E = mean(abs(Y).^2, 2);
bits = quantizer.allocate_bits(E, nbits_plot, 4);
Yh = zeros(size(Y));
for ch=1:4
    if bits(ch)>0
        sc = max(abs(Y(ch,:)))+1e-6; 
        Yh(ch,:) = quantizer.uniform_inverse(quantizer.uniform(Y(ch,:), bits(ch), sc), bits(ch), sc); 
    end
end
x_mera = fb_vis.synthesize(Yh);

% Plotting
subplot(1,3,1); 
plot(real(x_plot(1:1000)), imag(x_plot(1:1000)), '.k'); 
title('Original Signal (OFDM)', 'FontSize', 12); axis square;
xlabel('In-Phase (I)'); ylabel('Quadrature (Q)'); xlim([-4 4]); ylim([-4 4]);

subplot(1,3,2); 
plot(real(x_cpri(1:1000)), imag(x_cpri(1:1000)), '.b'); 
title('CPRI @ 4 bits (Noisy)', 'FontSize', 12); axis square;
xlabel('In-Phase (I)'); ylabel('Quadrature (Q)'); xlim([-4 4]); ylim([-4 4]);

subplot(1,3,3); 
plot(real(x_mera(1:1000)), imag(x_mera(1:1000)), '.r'); 
title('MERA K=2 @ 4 bits (Clean)', 'FontSize', 12); axis square;
xlabel('In-Phase (I)'); ylabel('Quadrature (Q)'); xlim([-4 4]); ylim([-4 4]);

% --- Saving Figures (Section 8) ---
filename_const = 'results/constellation_comparison';

% 1. PNG (high resolution - 300 DPI)
exportgraphics(gcf, [filename_const '.png'], 'Resolution', 300);

% 2. PDF (vectorial) 
exportgraphics(gcf, [filename_const '.pdf'], 'ContentType', 'vector');

fprintf('Constellation figures saved as PNG and PDF .\n');
fprintf('Experiment Complete.\n');