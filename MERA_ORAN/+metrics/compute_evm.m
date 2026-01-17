function evm = compute_evm(x_ref, x_test)
% METRICS.COMPUTE_EVM Calcula Error Vector Magnitude (3GPP)
%
%   evm = metrics.compute_evm(x_ref, x_test)
%
%   Entradas:
%       x_ref  - Sinal de referência (original)
%       x_test - Sinal de teste (reconstruído)
%
%   Saída:
%       evm - Error Vector Magnitude em porcentagem (%)
%
%   Definição 3GPP (RMS):
%       EVM = sqrt( mean(|error|^2) / mean(|ref|^2) ) * 100%
%
%   Limites típicos 5G NR:
%       QPSK:   17.5%
%       16QAM:  12.5%
%       64QAM:   8.0%
%       256QAM:  3.5%

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

% Vetor de erro
error = x_test - x_ref;

% EVM RMS (3GPP)
evm = sqrt(mean(abs(error).^2) / mean(abs(x_ref).^2)) * 100;

end
