function [iq_time, info] = generate_5g_iq(opts)
% GENERATE_5G_IQ Gera sinal 5G CP-OFDM com carga parcial para testes de esparsidade
% Recuperado do histórico: Versão com 30% de Traffic Load e inserção manual de CP.

arguments
    opts.Nfft (1,1) double = 1024
    opts.SubcarrierSpacing (1,1) double = 15 % kHz
    opts.Modulation string = '16QAM'
    opts.TrafficLoad (1,1) double = 0.3 % <--- 30% de Carga (Esparsidade para Wave6G)
end

    % 1. Determinar número de subportadoras ativas baseada na carga
    % Ocupa a banda central para simular alocação contígua típica
    numActive = floor(opts.Nfft * opts.TrafficLoad);
    
    % Assegura número par para simetria (opcional, mas boa prática)
    if mod(numActive, 2) ~= 0
        numActive = numActive - 1;
    end

    % 2. Geração de Símbolos (Lógica reconstruída para 16QAM)
    if strcmp(opts.Modulation, '16QAM')
        M = 16;
        bps = 4;
    elseif strcmp(opts.Modulation, 'QPSK')
        M = 4;
        bps = 2;
    else
        error('Modulação não suportada neste script recuperado (use 16QAM ou QPSK)');
    end
    
    numBits = numActive * bps;
    dataBits = randi([0 1], numBits, 1);
    symbols = qammod(dataBits, M, 'InputType', 'bit', 'UnitAveragePower', true);

    % 3. Mapeamento no Grid OFDM (Centralizado)
    grid = zeros(opts.Nfft, 1);
    startIdx = floor((opts.Nfft - numActive)/2) + 1;
    endIdx = startIdx + numActive - 1;
    
    grid(startIdx:endIdx) = symbols;

    % 4. Modulação OFDM (IFFT)
    % ifftshift move a DC para o centro antes da IFFT
    ifft_out = ifft(ifftshift(grid)) * sqrt(opts.Nfft);

    % 5. Adição do Prefixo Cíclico (CP)
    % Comprimento padrão LTE/5G normal é aprox 7% (144 para 2048 FFT)
    cp_len = floor(opts.Nfft * 0.07); 
    iq_time = [ifft_out(end-cp_len+1:end); ifft_out];

    % 6. Normalização de Potência
    iq_time = iq_time / std(iq_time);

    % Estrutura de Informação
    info.NumActive = numActive;
    info.Nfft = opts.Nfft;
    info.CPLength = cp_len;
    info.BandwidthOccupancy = numActive / opts.Nfft;
end