function Yq = uniform(Y, nbits, scale)
% QUANTIZER.UNIFORM Quantização uniforme de coeficientes complexos
%
%   Yq = quantizer.uniform(Y, nbits, scale)
%
%   Entradas:
%       Y     - Matriz de coeficientes complexos (M x N_blocks)
%       nbits - Bits por componente (I ou Q)
%       scale - Fator de escala: escalar (global) ou vetor Mx1 (por sub-banda)
%
%   Saída:
%       Yq - Coeficientes quantizados (índices inteiros complexos)
%
%   Quantiza separadamente partes real e imaginária.
%   Se scale é vetor, aplica escala diferente para cada sub-banda (linha).

arguments
    Y
    nbits (1,1) double {mustBePositive, mustBeInteger}
    scale double {mustBePositive}
end

levels = 2^nbits;

% Normalizar (suporta escala por sub-banda)
if isscalar(scale)
    Y_norm = Y / scale;
else
    % scale é vetor Mx1: cada linha tem sua escala
    Y_norm = Y ./ scale(:);  % Broadcasting: (M x N) ./ (M x 1)
end

% Separar real e imaginário
Y_real = real(Y_norm);
Y_imag = imag(Y_norm);

% Clipar ao range [-1, 1]
Y_real = max(min(Y_real, 1 - eps), -1 + eps);
Y_imag = max(min(Y_imag, 1 - eps), -1 + eps);

% Quantizar para índices inteiros [0, levels-1]
Yq_real = round((Y_real + 1) / 2 * (levels - 1));
Yq_imag = round((Y_imag + 1) / 2 * (levels - 1));

% Combinar (para manter formato)
Yq = Yq_real + 1j * Yq_imag;

end
