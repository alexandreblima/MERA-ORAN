classdef FilterBank < handle
% MERA.FILTERBANK Banco de filtros paraunitário via cascata MERA
%
%   fb = mera.FilterBank(M, K) cria banco com M canais e K estágios
%
%   Propriedades:
%       M         - Número de canais
%       K         - Número de estágios de delay
%       Unitaries - Cell array com K+1 matrizes unitárias
%       FilterLength - Comprimento dos filtros (M + K*(M-1))
%
%   Métodos:
%       Y = analyze(fb, x)      - Transformada direta
%       x = synthesize(fb, Y)   - Transformada inversa (PR)
%       train(fb, data, opts)   - Treinar via otimização Riemanniana
%
%   Exemplo:
%       fb = mera.FilterBank(4, 2);
%       fb.verifyPR();
%       Y = fb.analyze(x);
%       x_hat = fb.synthesize(Y);
%
%   Referência:
%       Lima, A.B. "Generalized MERA-Wavelet Equivalence: Complete 
%       Characterization via Unitary Lattice Factorizations"

    properties
        M           % Canais
        K           % Estágios de delay
        Unitaries   % {U_0, U_1, ..., U_K} ⊂ U(M)
    end
    
    properties (Dependent)
        FilterLength
        NumParameters
    end
    
    methods
        function obj = FilterBank(M, K, initMethod)
            % Construtor
            %
            %   fb = mera.FilterBank(M, K)
            %   fb = mera.FilterBank(M, K, 'haar')
            %
            %   initMethod: 'random', 'identity', 'dft', 'haar'
            
            arguments
                M (1,1) double {mustBePositive, mustBeInteger}
                K (1,1) double {mustBeNonnegative, mustBeInteger}
                initMethod string = "random"
            end
            
            obj.M = M;
            obj.K = K;
            obj.Unitaries = cell(1, K+1);
            
            % Inicialização
            for k = 1:K+1
                switch initMethod
                    case "random"
                        obj.Unitaries{k} = obj.randomUnitary(M);
                    case "identity"
                        obj.Unitaries{k} = eye(M);
                    case "dft"
                        obj.Unitaries{k} = dftmtx(M) / sqrt(M);
                    case "haar"
                        obj.Unitaries{k} = obj.haarMatrix(M);
                    otherwise
                        error('initMethod deve ser: random, identity, dft, haar');
                end
            end
        end
        
        function L = get.FilterLength(obj)
            % Comprimento dos filtros: L = M + K*(M-1)
            L = obj.M + obj.K * (obj.M - 1);
        end
        
        function n = get.NumParameters(obj)
            % Graus de liberdade reais em U(M)^{K+1}
            n = (obj.K + 1) * obj.M^2;
        end
        
        function Y = analyze(obj, x)
            % ANALYZE Transformada direta (análise)
            %
            %   Y = fb.analyze(x)
            %
            %   Entrada:
            %       x - vetor coluna de amostras I/Q (N x 1)
            %
            %   Saída:
            %       Y - coeficientes de sub-banda (M x N_blocks)
            %
            %   Implementa: E(z) = U_K * Λ(z) * U_{K-1} * ... * U_0
            
            x = x(:);
            N = length(x);
            M = obj.M;
            
            % Padding para múltiplo de M
            padLen = mod(M - mod(N, M), M);
            if padLen > 0 && padLen < M
                x = [x; zeros(padLen, 1)];
            end
            N_padded = length(x);
            N_blocks = N_padded / M;
            
            % Decomposição polifásica
            X = reshape(x, M, N_blocks);
            
            % Aplicar U_0
            Y = obj.Unitaries{1} * X;
            
            % Aplicar sequência Λ(z) * U_k para k = 1, ..., K
            for k = 2:obj.K+1
                Y = obj.applyDelay(Y);
                Y = obj.Unitaries{k} * Y;
            end
        end
        
        function x = synthesize(obj, Y)
            % SYNTHESIZE Transformada inversa (síntese) - PR garantido
            %
            %   x = fb.synthesize(Y)
            %
            %   Entrada:
            %       Y - coeficientes de sub-banda (M x N_blocks)
            %
            %   Saída:
            %       x - vetor reconstruído
            %
            %   Implementa: R(z) = Ẽ(z) = U_0^H * Λ(z^{-1})^H * ... * U_K^H
            
            X = Y;
            
            % Cascata reversa
            for k = obj.K+1:-1:2
                X = obj.Unitaries{k}' * X;
                X = obj.applyDelayInverse(X);
            end
            X = obj.Unitaries{1}' * X;
            
            % Vetor de saída
            x = X(:);
        end
        
        function err = verifyPR(obj, N)
            % VERIFYPR Verificar perfect reconstruction
            %
            %   err = fb.verifyPR()
            %   err = fb.verifyPR(2048)
            %
            %   Testa se synthesize(analyze(x)) ≈ x
            
            arguments
                obj
                N (1,1) double = 1024
            end
            
            % Sinal aleatório complexo
            x = randn(N, 1) + 1j * randn(N, 1);
            
            % Forward e inverse
            Y = obj.analyze(x);
            x_hat = obj.synthesize(Y);
            
            % Erro relativo
            err = norm(x - x_hat(1:N)) / norm(x);
            
            if err < 1e-10
                fprintf('PR verificado: erro = %.2e\n', err);
            else
                warning('PR falhou: erro = %.2e', err);
            end
        end
        
        function displayInfo(obj)
            % DISPLAYINFO Exibe informações do banco de filtros
            
            fprintf('\n=== MERA Filter Bank ===\n');
            fprintf('Canais (M):           %d\n', obj.M);
            fprintf('Estágios (K):         %d\n', obj.K);
            fprintf('Comprimento filtros:  %d taps\n', obj.FilterLength);
            fprintf('Num. unitárias:       %d\n', obj.K + 1);
            fprintf('Parâmetros reais:     %d\n', obj.NumParameters);
            fprintf('========================\n\n');
        end
    end
    
    methods (Access = private)
        function Y = applyDelay(obj, Y)
            % Aplica matriz de delay Λ(z): atrasa linhas 2:M por 1 amostra
            % Usa extensão circular para preservar PR
            M = obj.M;
            Y(2:M, :) = circshift(Y(2:M, :), 1, 2);
        end
        
        function Y = applyDelayInverse(obj, Y)
            % Aplica Λ(z)^{-1}: adianta linhas 2:M por 1 amostra
            % Usa extensão circular para preservar PR
            M = obj.M;
            Y(2:M, :) = circshift(Y(2:M, :), -1, 2);
        end
    end
    
    methods (Static)
        function U = randomUnitary(M)
            % Gera matriz unitária aleatória via decomposição QR
            [U, ~] = qr(randn(M) + 1j * randn(M));
        end
        
        function H = haarMatrix(M)
            % Matriz de Haar normalizada (requer M potência de 2)
            if log2(M) ~= floor(log2(M))
                warning('Haar matrix requer M potência de 2. Usando DFT.');
                H = dftmtx(M) / sqrt(M);
            else
                H = hadamard(M) / sqrt(M);
            end
        end
    end
end
