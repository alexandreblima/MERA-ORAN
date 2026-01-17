function Yhat = uniform_inverse(Yq, nbits, scale)
% QUANTIZER.UNIFORM_INVERSE Dequantização uniforme
%
%   Yhat = quantizer.uniform_inverse(Yq, nbits, scale)
%
%   Entradas:
%       Yq    - Coeficientes quantizados (saída de quantizer.uniform)
%       nbits - Bits por componente (I ou Q)
%       scale - Fator de escala: escalar (global) ou vetor Mx1 (por sub-banda)
%
%   Saída:
%       Yhat - Coeficientes reconstruídos (complexos)

arguments
    Yq
    nbits (1,1) double {mustBePositive, mustBeInteger}
    scale double {mustBePositive}
end

levels = 2^nbits;

% Separar componentes
Yq_real = real(Yq);
Yq_imag = imag(Yq);

% Converter de índices para valores normalizados [-1, 1]
Y_real = Yq_real / (levels - 1) * 2 - 1;
Y_imag = Yq_imag / (levels - 1) * 2 - 1;

% Desnormalizar (suporta escala por sub-banda)
if isscalar(scale)
    Yhat = (Y_real + 1j * Y_imag) * scale;
else
    % scale é vetor Mx1
    Yhat = (Y_real + 1j * Y_imag) .* scale(:);
end

end
