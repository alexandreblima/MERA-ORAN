function lossHistory = train(fb, trainData, opts)
% MERA.TRAIN Treina MERA para maximizar COMPACTAÇÃO DE ENERGIA
%
%   Otimiza as matrizes unitárias U_k para concentrar a energia do sinal
%   no primeiro canal (subband 0), explorando a preservação de norma L2
%   garantida pela unitariedade.
%
%   Loss = -E_1 / E_total  onde  E_k = sum(|Y_k|^2)
%
%   Como ||Y||_2 = ||X||_2 (unitário), maximizar E_1 força os demais
%   canais a terem energia residual mínima, produzindo coeficientes
%   esparsos ideais para quantização com alocação dinâmica de bits.
%
%   O algoritmo usa gradiente Riemanniano com retração via decomposição
%   polar para manter as matrizes no grupo unitário U(M). A retração polar
%   é computada via SVD: dado A = U - lr*G, calcula-se A = WΣV' e
%   retorna-se U_new = WV'.
%
%   Referência: Seção IV-B e Algoritmo 1 do paper.

arguments
    fb mera.FilterBank
    trainData cell
    opts.NumEpochs (1,1) double = 100
    opts.LearningRate (1,1) double = 0.05
    opts.BatchSize (1,1) double = 64
    opts.Verbose (1,1) logical = true
    opts.SparsityWeight (1,1) double = 1.0 % Reservado para futuras extensões
    % Argumentos mantidos para compatibilidade de interface:
    opts.TargetBits (1,1) double = 6 
    opts.Temperature (1,1) double = 5
end

% Verificar Deep Learning Toolbox
if ~exist('dlarray', 'file')
    error('Deep Learning Toolbox necessária para diferenciação automática.');
end

% Preparação dos dados
numSamples = length(trainData);
numBatches = floor(numSamples / opts.BatchSize);

% Extrair parâmetros para dlarray
parameters = cell(1, fb.K + 1);
for k = 1:length(fb.Unitaries)
    parameters{k} = dlarray(fb.Unitaries{k});
end

lossHistory = zeros(opts.NumEpochs, 1);
M = fb.M;

% Loop de Treino
for epoch = 1:opts.NumEpochs
    epochLoss = 0;
    perm = randperm(numSamples);
    
    for b = 1:numBatches
        % Carregar e preparar batch
        idx = perm((b-1)*opts.BatchSize + 1 : b*opts.BatchSize);
        batchList = trainData(idx);
        
        % Reshape para formato polifásico (M x N)
        processedBatch = cell(size(batchList));
        for i = 1:length(batchList)
            d = batchList{i};
            if size(d, 1) ~= M
                len = numel(d);
                rem = mod(len, M);
                if rem > 0, d = d(1:end-rem); len = len - rem; end
                d = reshape(d, M, len/M);
            end
            processedBatch{i} = d;
        end
        X_batch = cat(2, processedBatch{:});
        dlX = dlarray(X_batch);
        
        % Forward + Backward via diferenciação automática
        [loss, grads] = dlfeval(@energyCompactionLoss, parameters, dlX, fb.M, fb.K);
        
        epochLoss = epochLoss + extractdata(loss);
        
        % Atualização Riemanniana com retração polar
        for k = 1:length(parameters)
            G = extractdata(grads{k});
            U_curr = extractdata(parameters{k});
            
            % Passo Euclidiano
            A = U_curr - opts.LearningRate * G;
            
            % Retração polar: projeta de volta em U(M)
            [W, ~, V] = svd(A);
            parameters{k} = dlarray(W * V');
        end
    end
    
    lossHistory(epoch) = epochLoss / numBatches;
    
    if opts.Verbose && (mod(epoch, 10) == 0 || epoch == 1)
        fprintf('Época %3d/%d: Energy Ratio Loss = %.4f\n', ...
            epoch, opts.NumEpochs, lossHistory(epoch));
    end
end

% Salvar parâmetros treinados no objeto FilterBank
for k = 1:length(parameters)
    fb.Unitaries{k} = extractdata(parameters{k});
end

end

%% === FUNÇÃO DE PERDA: COMPACTAÇÃO DE ENERGIA ===
function [loss, gradients] = energyCompactionLoss(parameters, X, M, K)
    % Aplica banco de filtros MERA (análise)
    Y = X;
    for k = 1:K
        U = parameters{k};
        Y = U * Y;
        % Delay polifásico: desloca linhas 2:M em uma amostra
        Y_shifted_rows = circshift(Y(2:M, :), 1, 2);
        Y = cat(1, Y(1, :), Y_shifted_rows);
    end
    Y = parameters{K+1} * Y;
    
    % Computa energia por canal
    channel_energies = sum(abs(Y).^2, 2); % Vetor Mx1
    total_energy = sum(channel_energies);
    
    % Maximiza fração de energia no canal 1 (minimiza o negativo)
    ratio_ch1 = channel_energies(1) / total_energy;
    loss = -ratio_ch1;
    
    % Gradientes via diferenciação automática
    gradients = dlgradient(loss, parameters);
end