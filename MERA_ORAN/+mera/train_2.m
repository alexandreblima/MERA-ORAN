function lossHistory = train(fb, trainData, opts)
% MERA.TRAIN Treina o banco de filtros MERA via otimização em U(M)^{K+1}
%
%   train(fb, trainData)
%   train(fb, trainData, 'NumEpochs', 100, 'TargetBits', 6)
%   lossHistory = train(fb, trainData, ...)
%
%   Entradas:
%       fb        - Objeto mera.FilterBank
%       trainData - Cell array de vetores I/Q para treino
%
%   Opções:
%       NumEpochs    - Número de épocas (padrão: 100)
%       LearningRate - Taxa de aprendizado (padrão: 0.01)
%       TargetBits   - Bits por amostra alvo (padrão: 6)
%       BatchSize    - Tamanho do mini-batch (padrão: 32)
%       Verbose      - Exibir progresso (padrão: true)
%
%   Saída:
%       lossHistory - Vetor com loss por época
%
%   O algoritmo usa gradiente Riemanniano com retração via decomposição
%   polar para manter as matrizes no grupo unitário U(M). A retração polar
%   é mais eficiente que o mapa exponencial: polar(A) = U onde A = UP é
%   a decomposição polar, computada via SVD como U = W*V' onde A = WΣV'.

arguments
    fb mera.FilterBank
    trainData cell
    opts.NumEpochs (1,1) double = 100
    opts.LearningRate (1,1) double = 0.01
    opts.TargetBits (1,1) double = 6
    opts.BatchSize (1,1) double = 32
    opts.Verbose logical = true
end

M = fb.M;
K = fb.K;
numUnitaries = K + 1;
numData = length(trainData);

% Histórico de loss
lossHistory = zeros(opts.NumEpochs, 1);

if opts.Verbose
    fprintf('Treinando MERA FilterBank (M=%d, K=%d, %d bits)\n', M, K, opts.TargetBits);
    fprintf('Dados: %d amostras, BatchSize: %d, Épocas: %d\n\n', ...
        numData, opts.BatchSize, opts.NumEpochs);
end

for epoch = 1:opts.NumEpochs
    % Shuffle dados
    idx = randperm(numData);
    epochLoss = 0;
    numBatches = max(1, floor(numData / opts.BatchSize));
    
    for batch = 1:numBatches
        % Índices do batch
        startIdx = (batch - 1) * opts.BatchSize + 1;
        endIdx = min(batch * opts.BatchSize, numData);
        batchIdx = idx(startIdx:endIdx);
        batchData = trainData(batchIdx);
        actualBatchSize = length(batchData);
        
        % Inicializar gradientes
        grads = cell(1, numUnitaries);
        for k = 1:numUnitaries
            grads{k} = zeros(M, M);
        end
        
        batchLoss = 0;
        
        % Calcular loss e gradiente para cada amostra
        for i = 1:actualBatchSize
            x = batchData{i};
            x = x(:);
            N = length(x);
            
            % Forward pass
            Y = fb.analyze(x);
            
            % Quantização SUAVE para treinamento (gradiente não-zero)
            scales = max(abs(Y), [], 2) + eps;  % M×1
            [Yhat, ~] = quantizer.soft_uniform(Y, opts.TargetBits, scales, 5);
            
            % Reconstrução
            xhat = fb.synthesize(Yhat);
            xhat = xhat(1:N);
            
            % Loss (MSE normalizado = EVM^2)
            loss = sum(abs(x - xhat).^2) / sum(abs(x).^2);
            batchLoss = batchLoss + loss;
            
            % Gradiente numérico via diferenças finitas
            delta = 1e-4;  % Aumentado para melhor sensibilidade
            for k = 1:numUnitaries
                U_orig = fb.Unitaries{k};
                
                for p = 1:M
                    for q = 1:M
                        % Base do espaço tangente (anti-Hermitiana)
                        E = zeros(M);
                        E(p, q) = 1;
                        Omega = (E - E') / 2;  % Skew-Hermitian
                        
                        if norm(Omega, 'fro') < 1e-10
                            continue;
                        end
                        
                        % Perturbar: U_new = U * expm(delta * Omega) ≈ U * (I + delta*Omega)
                        % Aproximação de 1ª ordem suficiente para delta pequeno
                        fb.Unitaries{k} = U_orig * (eye(M) + delta * Omega);
                        
                        % Forward com perturbação (quantização suave)
                        Y_pert = fb.analyze(x);
                        scales_pert = max(abs(Y_pert), [], 2) + eps;
                        [Yhat_pert, ~] = quantizer.soft_uniform(Y_pert, opts.TargetBits, scales_pert, 5);
                        xhat_pert = fb.synthesize(Yhat_pert);
                        xhat_pert = xhat_pert(1:N);
                        
                        % Loss perturbado
                        loss_pert = sum(abs(x - xhat_pert).^2) / sum(abs(x).^2);
                        
                        % Gradiente
                        grads{k}(p, q) = grads{k}(p, q) + (loss_pert - loss) / delta;
                        
                        % Restaurar
                        fb.Unitaries{k} = U_orig;
                    end
                end
            end
        end
        
        batchLoss = batchLoss / actualBatchSize;
        epochLoss = epochLoss + batchLoss;
        
        % Atualização Riemanniana (retração via decomposição polar)
        for k = 1:numUnitaries
            G = grads{k} / actualBatchSize;
            
            % Gradiente Riemanniano: projetar no espaço tangente T_U U(M)
            % Tangente em U(M) são matrizes da forma U*Ω onde Ω é skew-Hermitian
            G_tangent = (G - G') / 2;
            
            % Passo no espaço ambiente
            A = fb.Unitaries{k} - opts.LearningRate * (fb.Unitaries{k} * G_tangent);
            
            % Retração polar: A = WΣV' → polar(A) = WV'
            [W, ~, V] = svd(A);
            fb.Unitaries{k} = W * V';
        end
    end
    
    lossHistory(epoch) = epochLoss / numBatches;
    
    if opts.Verbose && (mod(epoch, 10) == 0 || epoch == 1)
        evm_pct = sqrt(lossHistory(epoch)) * 100;
        fprintf('Época %3d/%d: Loss = %.6f (EVM ≈ %.2f%%)\n', ...
            epoch, opts.NumEpochs, lossHistory(epoch), evm_pct);
    end
end

if opts.Verbose
    fprintf('\nTreinamento concluído.\n');
    fprintf('Loss final: %.6f (EVM ≈ %.2f%%)\n', ...
        lossHistory(end), sqrt(lossHistory(end)) * 100);
end

end
