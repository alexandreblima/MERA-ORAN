function lossHistory = train(fb, trainData, opts)
% MERA.TRAIN Treina MERA para maximizar ESPARSIDADE (L1 Minimization)
%
%   Nesta abordagem (inspirada no seu paper), removemos o termo MSE
%   porque a reconstrução é garantida pela unitariedade.
%   O objetivo é puramente encontrar a rotação que minimiza a norma L1
%   dos coeficientes Y, forçando a energia a se concentrar.
%
%   Loss = sum(abs(Y))  sujeito a  Y = U * X
%
%   Como ||Y||_2 é constante (Unitário), min(L1) => Max(Esparsidade/Kurtosis).

arguments
    fb mera.FilterBank
    trainData cell
    opts.NumEpochs (1,1) double = 100
    opts.LearningRate (1,1) double = 0.05 % Pode ser maior agora (landscape mais simples)
    opts.BatchSize (1,1) double = 64
    opts.Verbose (1,1) logical = true
    opts.SparsityWeight (1,1) double = 1.0 % (Não usado se só tiver L1, mas bom pra futuro)
    % Argumentos não usados mas mantidos para compatibilidade:
    opts.TargetBits (1,1) double = 6 
    opts.Temperature (1,1) double = 5
end

% Verificar Deep Learning Toolbox
if ~exist('dlarray', 'file')
    error('Deep Learning Toolbox necessária para diferenciação automática.');
end

% Preparação dos dados (igual ao anterior)
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
        % Batch loading e Reshape (igual ao anterior)
        idx = perm((b-1)*opts.BatchSize + 1 : b*opts.BatchSize);
        batchList = trainData(idx);
        
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
        
        % === PASSO DIFERENCIAÇÃO AUTOMÁTICA (APENAS L1) ===
        [loss, grads] = dlfeval(@sparsityLoss, parameters, dlX, fb.M, fb.K);
        
        epochLoss = epochLoss + extractdata(loss);
        
        % === ATUALIZAÇÃO RIEMANNIANA ===
        for k = 1:length(parameters)
            G = extractdata(grads{k});
            U_curr = extractdata(parameters{k});
            
            % Gradient Descent na variedade Stiefel/Unitary
            % A = U - lr * G (Euclidiano) -> Retração Polar cuida do resto
            A = U_curr - opts.LearningRate * G;
            
            % Retração: projeta de volta em U(M) via SVD
            [W, ~, V] = svd(A);
            parameters{k} = dlarray(W * V');
        end
    end
    
    lossHistory(epoch) = epochLoss / numBatches;
    
    if opts.Verbose && (mod(epoch, 10) == 0 || epoch == 1)
        fprintf('Época %3d/%d: L1 Loss = %.4f\n', ...
            epoch, opts.NumEpochs, lossHistory(epoch));
    end
end

% Salvar parâmetros treinados
for k = 1:length(parameters)
    fb.Unitaries{k} = extractdata(parameters{k});
end

end

%% === FUNÇÃO DE PERDA: ESPARSIDADE L1 ===
function [loss, gradients] = sparsityLoss(parameters, X, M, K)
    % 1. ANÁLISE (Forward)
    Y = X;
    for k = 1:K
        U = parameters{k};
        Y = U * Y;
        Y_shifted_rows = circshift(Y(2:M, :), 1, 2);
        Y = cat(1, Y(1, :), Y_shifted_rows);
    end
    Y = parameters{K+1} * Y;
    
    % 2. COMPACTAÇÃO DE ENERGIA (A Nova Loss)
    % Calculamos a energia média de cada canal
    channel_energies = sum(abs(Y).^2, 2); % Vetor Mx1
    total_energy = sum(channel_energies);
    
    % Queremos MAXIMIZAR a fração de energia no Canal 1.
    % Logo, minimizamos o negativo dessa fração.
    ratio_ch1 = channel_energies(1) / total_energy;
    
    loss = -ratio_ch1; 
    
    % Exemplo: Se Ch1 tem 25% da energia, Loss = -0.25.
    % Se Ch1 tem 99% da energia, Loss = -0.99.
    % O gradiente vai forçar a rotação para alinhar o Ch1 com a variância máxima.

    gradients = dlgradient(loss, parameters);
end