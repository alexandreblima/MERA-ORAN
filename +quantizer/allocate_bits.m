function bits_per_channel = allocate_bits(energies, total_bits_avg, M)
% QUANTIZER.ALLOCATE_BITS Alocação gulosa (Greedy) de bits baseada em energia
%
%   bits = quantizer.allocate_bits(energies, total_bits_avg, M)
%
%   Entradas:
%       energies       - Vetor (Mx1) com a energia/variância de cada canal
%       total_bits_avg - Média de bits desejada por amostra (ex: 4 bits)
%       M              - Número de canais
%
%   Saída:
%       bits_per_channel - Vetor (Mx1) com inteiros indicando bits para cada canal

    % Alvo total de bits para o bloco M
    target_total_bits = total_bits_avg * M;
    
    % Inicialização: Começar com 0 bits
    bits_per_channel = zeros(M, 1);
    current_total = 0;
    
    % Evitar log de zero e garantir vetor coluna
    energies = energies(:) + eps;
    
    % Algoritmo Greedy (Bit Loading)
    % A cada passo, damos 1 bit para o canal que mais reduziria o erro total.
    % O erro de quantização é proporcional a: energia * 2^(-2*bits)
    
    while current_total < target_total_bits
        % Calcular distorção atual em cada canal
        current_distortions = energies .* (2.^(-2 * bits_per_channel));
        
        % Encontrar quem tem a maior distorção (quem precisa mais do bit)
        [~, idx] = max(current_distortions);
        
        % Atribuir bit
        bits_per_channel(idx) = bits_per_channel(idx) + 1;
        current_total = current_total + 1;
    end
end