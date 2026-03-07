function results = analyze_conditioning(params_base)
% ANALYZE_CONDITIONING - Question 2.2: Condition Number Analysis
%
% Experiment A: Fixed N_sub (fixed H), refined h (incremented Ne_sub)
% kappa(K) ~ O(Ne_sub^2) growth
% kappa(Sp) = constant (S_sub depends only on H, independent of internal h)
%
% Experiment B: Fixed Ne_sub (fixed h/H), increased N_sub
% kappa(K) ~ O(N_sub^2)
% kappa(Sp) ~ O(N_sub^2) non-scalable => requires BDD preconditioner
%
% Input:
% params_base - basic parameter structure (from config_params)

% Output:
% results - structure containing two sets of experimental data, expA and expB

    % Experiment A: Fix N_sub=8 (fix H), increase Ne_sub
    N_sub_A       = 8;
    Ne_sub_list_A = [1, 2, 4, 8, 16, 32];
    nA            = length(Ne_sub_list_A);

    kappa_K_A  = zeros(nA, 1);
    kappa_Sp_A = zeros(nA, 1);
    Ne_A       = zeros(nA, 1);

    fprintf('Exp A  (fixed N_sub=%d):\n', N_sub_A);
    fprintf('  %8s  %6s  %10s  %10s\n', 'Ne_sub', 'Ne', 'kappa(K)', 'kappa(Sp)');
    for i = 1:nA
        p        = params_base;
        p.N_sub  = N_sub_A;
        p.Ne_sub = Ne_sub_list_A(i);
        p.Ne     = p.N_sub * p.Ne_sub;
        p.h      = p.L / p.Ne;
        p.Nn     = p.Ne + 1;
        p.H      = p.Ne_sub * p.h;

        kappa_K_A(i)  = compute_kappa_K(p);
        kappa_Sp_A(i) = compute_kappa_Sp(p);
        Ne_A(i)       = p.Ne;

        fprintf('  %8d  %6d  %10.3e  %10.3e\n', ...
                p.Ne_sub, p.Ne, kappa_K_A(i), kappa_Sp_A(i));
    end


    %  Experiment B: Fix Ne_sub (fix h/H), increase N_sub

    Ne_sub_B     = max(params_base.Ne_sub, 4);
    N_sub_list_B = [4, 8, 16, 32, 64];
    nB           = length(N_sub_list_B);

    kappa_K_B  = zeros(nB, 1);
    kappa_Sp_B = zeros(nB, 1);
    N_sub_B    = zeros(nB, 1);

    fprintf('\nExp B  (fixed Ne_sub=%d, h/H=1/%d):\n', Ne_sub_B, Ne_sub_B);
    fprintf('  %8s  %6s  %10s  %10s\n', 'N_sub', 'Ne', 'kappa(K)', 'kappa(Sp)');
    for i = 1:nB
        p        = params_base;
        p.N_sub  = N_sub_list_B(i);
        p.Ne_sub = Ne_sub_B;
        p.Ne     = p.N_sub * p.Ne_sub;
        p.h      = p.L / p.Ne;
        p.Nn     = p.Ne + 1;
        p.H      = p.Ne_sub * p.h;

        kappa_K_B(i)  = compute_kappa_K(p);
        kappa_Sp_B(i) = compute_kappa_Sp(p);
        N_sub_B(i)    = p.N_sub;

        fprintf('  %8d  %6d  %10.3e  %10.3e\n', ...
                p.N_sub, p.Ne, kappa_K_B(i), kappa_Sp_B(i));
    end


    pA_K  = polyfit(log(Ne_sub_list_A(:)), log(kappa_K_A),  1);
    pA_Sp = polyfit(log(Ne_sub_list_A(:)), log(kappa_Sp_A), 1);
    pB_K  = polyfit(log(N_sub_B),          log(kappa_K_B),  1);
    pB_Sp = polyfit(log(N_sub_B),          log(kappa_Sp_B), 1);

    results.fit.pA_K  = pA_K(1);
    results.fit.pA_Sp = pA_Sp(1);
    results.fit.pB_K  = pB_K(1);
    results.fit.pB_Sp = pB_Sp(1);

  
    %  result
    results.expA.Ne_sub   = Ne_sub_list_A(:);
    results.expA.Ne       = Ne_A;
    results.expA.kappa_K  = kappa_K_A;
    results.expA.kappa_Sp = kappa_Sp_A;
    results.expA.N_sub    = N_sub_A;

    results.expB.N_sub    = N_sub_B;
    results.expB.Ne_sub   = Ne_sub_B;
    results.expB.kappa_K  = kappa_K_B;
    results.expB.kappa_Sp = kappa_Sp_B;


    %  Drawing
    plot_conditioning(results);

end


%  Auxiliary function 1: kappa(K) — Condition number of the global stiffness matrix after applying Dirichlet BC.
function kappa = compute_kappa_K(p)
    ke = (p.E * p.S / p.h) * [1 -1; -1 1];
    K  = zeros(p.Nn, p.Nn);
    for e = 1:p.Ne
        K([e, e+1], [e, e+1]) = K([e, e+1], [e, e+1]) + ke;
    end
    kappa = cond(K(2:end, 2:end));
end


% Auxiliary function 2: kappa(Sp) — Global Primal Schur matrix condition number

function kappa = compute_kappa_Sp(p)

    E = p.E;  S = p.S;  h = p.h;
    N_sub = p.N_sub;  Ne_sub = p.Ne_sub;
    n_nodes = N_sub * Ne_sub + 1;

    ke = (E * S / h) * [1 -1; -1 1];

    % Global number of interface nodes
    if_nodes = ((1:N_sub-1) * Ne_sub + 1).';
    n_if     = length(if_nodes);

    % Global ID -> Sp Index Mapping (0 = Non-interface)
    if_map = zeros(n_nodes, 1);
    for k = 1:n_if
        if_map(if_nodes(k)) = k;
    end

    Sp = zeros(n_if, n_if);

    for s = 1:N_sub
        n_start = (s-1) * Ne_sub + 1;
        n_end   =  s    * Ne_sub + 1;
        nloc    = Ne_sub + 1;

        % Local stiffness matrix
        K_sub = zeros(nloc, nloc);
        for e = 1:Ne_sub
            K_sub([e, e+1], [e, e+1]) = K_sub([e, e+1], [e, e+1]) + ke;
        end

        % Schur (supplement): Cohesive Internal DOF
        idx_b = [1; nloc];
        idx_i = (2:nloc-1).';
        if isempty(idx_i)
            S_sub = K_sub(idx_b, idx_b);
        else
            Kbb = K_sub(idx_b, idx_b);
            Kii = K_sub(idx_i, idx_i);
            Kbi = K_sub(idx_b, idx_i);
            Kib = K_sub(idx_i, idx_b);
            S_sub = Kbb - Kbi * (Kii \ Kib);
        end

        % Assemble into global SP
        global_b = [n_start; n_end];
        for ii = 1:2
            row = if_map(global_b(ii));
            if row == 0; continue; end
            for jj = 1:2
                col = if_map(global_b(jj));
                if col == 0; continue; end
                Sp(row, col) = Sp(row, col) + S_sub(ii, jj);
            end
        end
    end

    kappa = cond(Sp);
end


%  Auxiliary function 3: Plotting
function plot_conditioning(results)

    expA       = results.expA;
    expB       = results.expB;
    Ne_sub_vec = expA.Ne_sub;
    N_sub_vec  = expB.N_sub;

    %% Figure 1
    figure('Name', 'Q2.2 - Exp A: fixed H', 'Position', [80 150 1000 430]);

    % Left : κ(K) vs κ(Sp)
    subplot(1, 2, 1);
    loglog(Ne_sub_vec, expA.kappa_K,  'b-o', 'LineWidth', 2, ...
           'MarkerSize', 7, 'DisplayName', '\kappa(K)');
    hold on;
    loglog(Ne_sub_vec, expA.kappa_Sp, 'r-s', 'LineWidth', 2, ...
           'MarkerSize', 7, 'DisplayName', '\kappa(S_p)');

    C_K2 = expA.kappa_K(1) / Ne_sub_vec(1)^2;
    loglog(Ne_sub_vec, C_K2 * Ne_sub_vec.^2, 'b--', 'LineWidth', 1.2, ...
           'DisplayName', 'O(N_{e,sub}^2)');
    loglog(Ne_sub_vec, mean(expA.kappa_Sp) * ones(size(Ne_sub_vec)), 'r--', ...
           'LineWidth', 1.2, 'HandleVisibility', 'off');

    grid on; box on;
    xlabel('N_{e,sub}  [ h decreases,  H = L/N_{sub} fixed ]', 'FontSize', 11);
    ylabel('\kappa', 'FontSize', 11);
    title(sprintf('Fixed N_{sub} = %d', expA.N_sub), 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 10);

    % Right: κ(K)/κ(Sp)
    subplot(1, 2, 2);
    ratio = expA.kappa_K ./ expA.kappa_Sp;
    loglog(Ne_sub_vec, ratio, 'k-^', 'LineWidth', 2, 'MarkerSize', 7, ...
           'DisplayName', '\kappa(K) / \kappa(S_p)');
    hold on;
    C_r = ratio(1) / Ne_sub_vec(1)^2;
    loglog(Ne_sub_vec, C_r * Ne_sub_vec.^2, 'k--', 'LineWidth', 1.2, ...
           'DisplayName', 'O(N_{e,sub}^2)');

    grid on; box on;
    xlabel('N_{e,sub}', 'FontSize', 11);
    ylabel('\kappa(K) / \kappa(S_p)', 'FontSize', 11);
    title('Schur advantage: ratio grows as N_{e,sub}^2', 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 10);

    sgtitle('Q2.2 — Exp A: Fixed H,  refine internal mesh', ...
            'FontSize', 13, 'FontWeight', 'bold');

    %% Figure 2
    figure('Name', 'Q2.2 - Exp B: fixed h/H', 'Position', [180 200 1000 430]);

    % Left : κ(K) 和 κ(Sp) vs N_sub
    subplot(1, 2, 1);
    loglog(N_sub_vec, expB.kappa_K,  'b-o', 'LineWidth', 2, ...
           'MarkerSize', 7, 'DisplayName', '\kappa(K)');
    hold on;
    loglog(N_sub_vec, expB.kappa_Sp, 'r-s', 'LineWidth', 2, ...
           'MarkerSize', 7, 'DisplayName', '\kappa(S_p)');

    C_K2b = expB.kappa_K(1)  / N_sub_vec(1)^2;
    C_Sp2 = expB.kappa_Sp(1) / N_sub_vec(1)^2;
    loglog(N_sub_vec, C_K2b * N_sub_vec.^2, 'b--', 'LineWidth', 1.2, ...
           'HandleVisibility', 'off');
    loglog(N_sub_vec, C_Sp2 * N_sub_vec.^2, 'r--', 'LineWidth', 1.2, ...
           'DisplayName', 'O(N_{sub}^2)');

    grid on; box on;
    xlabel(sprintf('N_{sub}  [ h/H = 1/%d = const ]', expB.Ne_sub), 'FontSize', 11);
    ylabel('\kappa', 'FontSize', 11);
    title(sprintf('Fixed N_{e,sub} = %d', expB.Ne_sub), 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 10);

    % Right: κ(Sp)
    subplot(1, 2, 2);
    loglog(N_sub_vec, expB.kappa_Sp, 'r-s', 'LineWidth', 2, ...
           'MarkerSize', 7, 'DisplayName', '\kappa(S_p)');
    hold on;
    C_Sp2b = expB.kappa_Sp(1) / N_sub_vec(1)^2;
    loglog(N_sub_vec, C_Sp2b * N_sub_vec.^2, 'r--', 'LineWidth', 1.2, ...
           'DisplayName', 'O(N_{sub}^2)');

    grid on; box on;
    xlabel('N_{sub}  (fixed h/H)', 'FontSize', 11);
    ylabel('\kappa(S_p)', 'FontSize', 11);
    title('\kappa(S_p) \sim N_{sub}^2 ', 'FontSize', 12);
    legend('Location', 'northwest', 'FontSize', 10);

    sgtitle('Q2.2 — Exp B: Fixed h/H,  scalability test', ...
            'FontSize', 13, 'FontWeight', 'bold');
end