function [xhat, rate] = daubechies_compress(x, nbits, wavelet, level)
% BASELINES.DAUBECHIES_COMPRESS Compressão com wavelet Daubechies fixa
%
%   [xhat, rate] = baselines.daubechies_compress(x, nbits)
%   [xhat, rate] = baselines.daubechies_compress(x, nbits, wavelet, level)
%
%   Entradas:
%       x       - Vetor de amostras I/Q complexas
%       nbits   - Bits por coeficiente (padrão: 6)
%       wavelet - Nome da wavelet: 'db1' (Haar), 'db2', 'db4', etc. (padrão: 'db4')
%       level   - Nível de decomposição (padrão: 4)
%
%   Saídas:
%       xhat - Sinal reconstruído
%       rate - Taxa em bits por amostra complexa
%
%   Usa DWT para decompor, quantiza coeficientes uniformemente,
%   e reconstrói via IDWT.
%
%   Requer: Wavelet Toolbox

arguments
    x (:,1)
    nbits (1,1) double {mustBePositive, mustBeInteger} = 6
    wavelet string = "db4"
    level (1,1) double {mustBePositive, mustBeInteger} = 4
end

% Separar I e Q
I = real(x(:));
Q = imag(x(:));
N = length(x);

% Ajustar nível máximo baseado no comprimento do sinal
maxLevel = floor(log2(N)) - 2;
level = min(level, maxLevel);

% DWT de I
[cI, lI] = wavedec(I, level, wavelet);

% DWT de Q
[cQ, lQ] = wavedec(Q, level, wavelet);

% Escalas para quantização
scale_I = max(abs(cI)) + eps;
scale_Q = max(abs(cQ)) + eps;

% Número de níveis
levels = 2^nbits;

% Quantizar coeficientes de I
cI_norm = cI / scale_I;
cI_clip = max(min(cI_norm, 1 - eps), -1 + eps);
cI_q = round((cI_clip + 1) / 2 * (levels - 1));

% Quantizar coeficientes de Q
cQ_norm = cQ / scale_Q;
cQ_clip = max(min(cQ_norm, 1 - eps), -1 + eps);
cQ_q = round((cQ_clip + 1) / 2 * (levels - 1));

% Dequantizar
cI_hat = (cI_q / (levels - 1) * 2 - 1) * scale_I;
cQ_hat = (cQ_q / (levels - 1) * 2 - 1) * scale_Q;

% Reconstruir via IDWT
I_hat = waverec(cI_hat, lI, wavelet);
Q_hat = waverec(cQ_hat, lQ, wavelet);

% Ajustar comprimento
I_hat = I_hat(1:N);
Q_hat = Q_hat(1:N);

% Reconstruir sinal complexo
xhat = I_hat + 1j * Q_hat;

% Taxa: bits por amostra complexa
rate = 2 * nbits;

end
