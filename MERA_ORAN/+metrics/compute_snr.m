function snr = compute_snr(x_ref, x_test)
% METRICS.COMPUTE_SNR Calcula SNR de quantização em dB
%
%   snr = metrics.compute_snr(x_ref, x_test)
%
%   Entradas:
%       x_ref  - Sinal de referência (original)
%       x_test - Sinal de teste (reconstruído)
%
%   Saída:
%       snr - Signal-to-Noise Ratio em dB
%
%   Definição:
%       SNR = 10 * log10( sum(|ref|^2) / sum(|error|^2) )
%
%   Relação com EVM:
%       SNR ≈ -20 * log10(EVM/100)

arguments
    x_ref (:,1)
    x_test (:,1)
end

% Garantir vetores coluna
x_ref = x_ref(:);
x_test = x_test(:);

% Alinhar comprimento
N = min(length(x_ref), length(x_test));
x_ref = x_ref(1:N);
x_test = x_test(1:N);

% Ruído de quantização
noise = x_test - x_ref;

% SNR em dB
signal_power = sum(abs(x_ref).^2);
noise_power = sum(abs(noise).^2);

if noise_power < eps
    snr = Inf;
else
    snr = 10 * log10(signal_power / noise_power);
end

end
