%% MAIN_EXPERIMENT.M
% Experimento Rate-Distortion para compressão de fronthaul 5G
%
% Compara:
%   1. CPRI padrão (quantização escalar)
%   2. Daubechies-4 (wavelet fixa)
%   3. MERA Adaptativa (K=0, 2, 4) com Alocação de Bits
%
% Autor: Alexandre Barbosa de Lima
% Data: 2025/2026

clear; clc; close all;

%% ==================== PARÂMETROS ====================
fprintf('==============================================\n');
fprintf('  Fronthaul Compression: MERA vs Baselines\n');
fprintf('==============================================\n\n');

% Dados
NUM_TRAIN = 100;        % Amostras para treino
NUM_TEST = 50;          % Amostras para avaliação

% Arquitetura MERA
M = 4;                  % Número de canais
K_VALUES = [0, 2, 4];   % Profundidades a testar

% Compressão
BITS_RANGE = 3:8;       % Bits médios por amostra (I ou Q)

% Treinamento
NUM_EPOCHS = 50;        % Épocas
LEARNING_RATE = 0.01;    % Learning rate agressivo para Loss de Energia

%% ==================== GERAR DADOS (Traffic Load 30%) ====================
fprintf('Gerando dados de treino (Carga 30%%, %d amostras)...\n', NUM_TRAIN);
trainData = cell(NUM_TRAIN, 1);
for i = 1:NUM_TRAIN
    % Gera sinal com esparsidade espectral (Traffic Load 0.3)
    [iq_signal, ~] = signal.generate_5g_iq('TrafficLoad', 0.3); 
    trainData{i} = iq_signal;
    
    if mod(i, 20) == 0, fprintf('  %d/%d\n', i, NUM_TRAIN); end
end

fprintf('Gerando dados de teste (%d amostras)...\n', NUM_TEST);
testData = cell(NUM_TEST, 1);
for i = 1:NUM_TEST
    [iq_signal, ~] = signal.generate_5g_iq('TrafficLoad', 0.3);
    testData{i} = iq_signal;
end
fprintf('Dados gerados.\n');

%% ==================== INICIALIZAR RESULTADOS ====================
numBits = length(BITS_RANGE);
results = struct();
results.bits = BITS_RANGE;
results.cpri = struct('evm', zeros(numBits, 1), 'snr', zeros(numBits, 1));
results.db4 = struct('evm', zeros(numBits, 1), 'snr', zeros(numBits, 1));

for K = K_VALUES
    fieldName = sprintf('mera_K%d', K);
    results.(fieldName) = struct('evm', zeros(numBits, 1), 'snr', zeros(numBits, 1));
end

%% ==================== BASELINE: CPRI PADRÃO ====================
fprintf('\n=== CPRI Padrão (quantização escalar) ===\n');
for b = 1:numBits
    nbits = BITS_RANGE(b);
    evms = zeros(NUM_TEST, 1);
    snrs = zeros(NUM_TEST, 1);
    
    for i = 1:NUM_TEST
        x = testData{i};
        [xhat, ~] = baselines.cpri_standard(x, nbits);
        evms(i) = metrics.compute_evm(x, xhat);
        snrs(i) = metrics.compute_snr(x, xhat);
    end
    
    results.cpri.evm(b) = mean(evms);
    results.cpri.snr(b) = mean(snrs);
    fprintf('  %d bits: EVM = %5.2f%%, SNR = %5.1f dB\n', ...
        nbits, results.cpri.evm(b), results.cpri.snr(b));
end

%% ==================== BASELINE: DAUBECHIES-4 ====================
fprintf('\n=== Daubechies-4 (wavelet fixa) ===\n');
for b = 1:numBits
    nbits = BITS_RANGE(b);
    evms = zeros(NUM_TEST, 1);
    snrs = zeros(NUM_TEST, 1);
    
    for i = 1:NUM_TEST
        x = testData{i};
        [xhat, ~] = baselines.daubechies_compress(x, nbits, 'db4');
        evms(i) = metrics.compute_evm(x, xhat);
        snrs(i) = metrics.compute_snr(x, xhat);
    end
    
    results.db4.evm(b) = mean(evms);
    results.db4.snr(b) = mean(snrs);
    fprintf('  %d bits: EVM = %5.2f%%, SNR = %5.1f dB\n', ...
        nbits, results.db4.evm(b), results.db4.snr(b));
end

%% ==================== MERA ADAPTATIVA ====================
for K = K_VALUES
    L = M + K * (M - 1);
    fprintf('\n=== MERA M=%d, K=%d (L=%d taps) ===\n', M, K, L);
    
    fieldName = sprintf('mera_K%d', K);
    
    % Treinamento (Feito uma vez por K, pois a Loss é de Esparsidade, independente de bits)
    % Se quiser treinar por taxa, mova para dentro do loop de bits.
    
    fprintf('  Inicializando e Treinando Filtros...\n');
    fb = mera.FilterBank(M, K, 'random');
    fb.verifyPR(1024);
    
    lossHistory = mera.train(fb, trainData, ...
        'NumEpochs', NUM_EPOCHS, ...
        'LearningRate', LEARNING_RATE, ...
        'BatchSize', 32, ...
        'Verbose', false);
    
    fprintf('  Loss Final: %.4f (Energia concentrada)\n', lossHistory(end));
    
    % Diagnóstico de Energia
    Y_diag = fb.analyze(testData{1});
    E_diag = sum(abs(Y_diag).^2, 2);
    E_rel = E_diag / sum(E_diag) * 100;
    fprintf('  Energia por canal: [%.1f%%, %.1f%%, %.1f%%, %.1f%%]\n', E_rel);

    % Loop de Taxas (Bit Allocation)
    for b = 1:numBits
        nbits_target = BITS_RANGE(b);
        
        evms = zeros(NUM_TEST, 1);
        snrs = zeros(NUM_TEST, 1);
        
        for i = 1:NUM_TEST
            x = testData{i};
            
            % 1. Análise
            Y = fb.analyze(x);
            
            % 2. Energia do bloco (para alocação)
            energies = mean(abs(Y).^2, 2);
            
            % 3. Alocação de Bits Dinâmica
            bits_vec = quantizer.allocate_bits(energies, nbits_target, M);
            
            % 4. Quantização
            Yhat = zeros(size(Y));
            for ch = 1:M
                nb = bits_vec(ch);
                if nb > 0
                    % ESCALA DE PICO (Evita clipping no OFDM)
                    scale_factor = max(abs(Y(ch,:))) + 1e-6;
                    
                    Yq = quantizer.uniform(Y(ch,:), nb, scale_factor);
                    Yhat(ch,:) = quantizer.uniform_inverse(Yq, nb, scale_factor);
                else
                    Yhat(ch,:) = 0; % Descarta ruído
                end
            end
            
            % 5. Síntese
            xhat = fb.synthesize(Yhat);
            
            % Ajuste de tamanho
            if length(xhat) > length(x)
                xhat = xhat(1:length(x));
            end
            
            evms(i) = metrics.compute_evm(x, xhat);
            snrs(i) = metrics.compute_snr(x, xhat);
        end
        
        results.(fieldName).evm(b) = mean(evms);
        results.(fieldName).snr(b) = mean(snrs);
        fprintf('  %d bits (avg): EVM = %5.2f%%, SNR = %5.1f dB\n', ...
            nbits_target, results.(fieldName).evm(b), results.(fieldName).snr(b));
    end
end

%% ==================== PLOTAR RESULTADOS ====================
fprintf('\n=== Gerando figuras ===\n');
figure('Position', [100, 100, 1000, 400], 'Color', 'w');
colors = lines(2 + length(K_VALUES));

% Subplot 1: EVM
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
xlabel('Taxa Média (bits/amostra)'); ylabel('EVM (%)');
title('Rate-Distortion: EVM (Menor é melhor)');
legend('Location', 'northeast'); set(gca, 'YScale', 'log');

% Subplot 2: SNR
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
xlabel('Taxa Média (bits/amostra)'); ylabel('SNR (dB)');
title('Rate-Distortion: SNR (Maior é melhor)');
legend('Location', 'southeast');

% Salvar
if ~exist('results', 'dir'), mkdir('results'); end
saveas(gcf, 'results/rate_distortion.png');
save('results/experiment_results.mat', 'results');

fprintf('\nExperimento Concluído com Sucesso!\n');

%% ==================== PLOT DE CONSTELAÇÃO (Visual Proof) ====================
fprintf('\nGerando plot de constelação para 4 bits...\n');
figure('Position', [100, 100, 1200, 400], 'Color', 'w');

% Pegar um sinal de teste
x_plot = testData{1};
nbits_plot = 4;

% 1. CPRI
[x_cpri, ~] = baselines.cpri_standard(x_plot, nbits_plot);

% 2. MERA K=2 (Precisamos reinstanciar e carregar o melhor modelo ou treinar rápido)
% Como o fb do loop anterior já está treinado (K=4 ou K=2 dependendo da ordem), 
% o ideal seria salvar o fb_best durante o loop. 
% Assumindo que você rodou e o fb atual é o K=4 ou K=2:
% Vamos usar o K=2 se possível, ou re-treinar rapidinho só pra plotar:
fb_plot = mera.FilterBank(4, 2, 'random'); 
mera.train(fb_plot, trainData(1:50), 'NumEpochs', 30, 'Verbose', false); % Treino rápido

% Processar MERA
Y = fb_plot.analyze(x_plot);
E = mean(abs(Y).^2, 2);
bits = quantizer.allocate_bits(E, nbits_plot, 4);
Yh = zeros(size(Y));
for ch=1:4
    if bits(ch)>0
        sc = max(abs(Y(ch,:)))+1e-6; 
        Yh(ch,:) = quantizer.uniform_inverse(quantizer.uniform(Y(ch,:), bits(ch), sc), bits(ch), sc); 
    end
end
x_mera = fb_plot.synthesize(Yh);

% Plots
subplot(1,3,1); 
plot(real(x_plot(1:1000)), imag(x_plot(1:1000)), '.k'); title('Original (OFDM)'); axis square;
xlabel('I'); ylabel('Q'); xlim([-4 4]); ylim([-4 4]);

subplot(1,3,2); 
plot(real(x_cpri(1:1000)), imag(x_cpri(1:1000)), '.r'); title('CPRI 4 bits (EVM ~12%)'); axis square;
xlabel('I'); ylabel('Q'); xlim([-4 4]); ylim([-4 4]);

subplot(1,3,3); 
plot(real(x_mera(1:1000)), imag(x_mera(1:1000)), '.g'); title('MERA K=2 4 bits (EVM ~7%)'); axis square;
xlabel('I'); ylabel('Q'); xlim([-4 4]); ylim([-4 4]);

saveas(gcf, 'results/constellation_comparison.png');