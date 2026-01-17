function [Yhat, Yq] = soft_uniform(Y, nbits, scale, temperature)
% QUANTIZER.SOFT_UNIFORM Quantização suave diferenciável para treinamento
%
%   [Yhat, Yq] = quantizer.soft_uniform(Y, nbits, scale, temperature)
%
%   Usa soft-round: aproxima round(x) por x + tanh(β*(x - round(x)))
%   onde β (temperature) controla a suavidade.
%   
%   β → ∞: hard quantization (round)
%   β pequeno: suave, gradiente não-zero
%
%   Entradas:
%       Y           - Matriz de coeficientes complexos (M x N_blocks)
%       nbits       - Bits por componente (I ou Q)
%       scale       - Escalar ou vetor Mx1 (por sub-banda)
%       temperature - Parâmetro de suavidade (padrão: 5)
%
%   Saídas:
%       Yhat - Coeficientes "quantizados" suavemente (para loss)
%       Yq   - Índices quantizados hard (para referência)

arguments
    Y
    nbits (1,1) double {mustBePositive, mustBeInteger}
    scale double {mustBePositive}
    temperature (1,1) double = 5
end

levels = 2^nbits;

% Normalizar
if isscalar(scale)
    Y_norm = Y / scale;
else
    Y_norm = Y ./ scale(:);
end

% Separar real e imaginário
Y_real = real(Y_norm);
Y_imag = imag(Y_norm);

% Clipar ao range [-1, 1]
Y_real = max(min(Y_real, 1 - eps), -1 + eps);
Y_imag = max(min(Y_imag, 1 - eps), -1 + eps);

% Mapear para [0, levels-1]
Y_real_scaled = (Y_real + 1) / 2 * (levels - 1);
Y_imag_scaled = (Y_imag + 1) / 2 * (levels - 1);

% Soft round: x + tanh(β * (x - round(x))) aproxima round(x)
% Gradiente é não-zero em quase todo lugar
Yq_real_soft = Y_real_scaled + tanh(temperature * (Y_real_scaled - round(Y_real_scaled)));
Yq_imag_soft = Y_imag_scaled + tanh(temperature * (Y_imag_scaled - round(Y_imag_scaled)));

% Hard quantization para referência
Yq_real = round(Y_real_scaled);
Yq_imag = round(Y_imag_scaled);
Yq = Yq_real + 1j * Yq_imag;

% Converter soft de volta para valores
Y_real_hat = Yq_real_soft / (levels - 1) * 2 - 1;
Y_imag_hat = Yq_imag_soft / (levels - 1) * 2 - 1;

% Desnormalizar
if isscalar(scale)
    Yhat = (Y_real_hat + 1j * Y_imag_hat) * scale;
else
    Yhat = (Y_real_hat + 1j * Y_imag_hat) .* scale(:);
end

end
