function [xhat, rate] = cpri_standard(x, nbits)
% BASELINES.CPRI_STANDARD Quantização CPRI padrão (escalar uniforme)
%
%   [xhat, rate] = baselines.cpri_standard(x, nbits)
%
%   Entradas:
%       x     - Vetor de amostras I/Q complexas
%       nbits - Bits por componente I ou Q (padrão CPRI: 15)
%
%   Saídas:
%       xhat - Sinal reconstruído após quantização
%       rate - Taxa em bits por amostra complexa (= 2*nbits)
%
%   CPRI usa quantização escalar uniforme separada para I e Q.
%   Cada componente usa nbits, totalizando 2*nbits por amostra.

arguments
    x (:,1) 
    nbits (1,1) double {mustBePositive, mustBeInteger} = 15
end

% Separar I e Q
I = real(x);
Q = imag(x);

% Escala para normalizar ao range [-1, 1]
scale_I = max(abs(I)) + eps;
scale_Q = max(abs(Q)) + eps;

% Número de níveis de quantização
levels = 2^nbits;

% Quantizar I
I_norm = I / scale_I;
I_clip = max(min(I_norm, 1 - eps), -1 + eps);
I_q = round((I_clip + 1) / 2 * (levels - 1));

% Quantizar Q
Q_norm = Q / scale_Q;
Q_clip = max(min(Q_norm, 1 - eps), -1 + eps);
Q_q = round((Q_clip + 1) / 2 * (levels - 1));

% Dequantizar
I_hat = (I_q / (levels - 1) * 2 - 1) * scale_I;
Q_hat = (Q_q / (levels - 1) * 2 - 1) * scale_Q;

% Reconstruir sinal complexo
xhat = I_hat + 1j * Q_hat;

% Taxa: bits por amostra complexa
rate = 2 * nbits;

end
