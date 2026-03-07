function info = analyze_conditioning_bdd(params)
% ANALYZE_CONDITIONING_BDD - Question 2.6
%
% By explicitly assembling S_p_tilde^{-1} * S_p,
% Verify the condition number of the preconditioning system: kappa(S_p_tilde^{-1} Sp) = O(1)
% And kappa(Sp) = O(N_sub^2)
%
% Reference: Algorithms.pdf, Algorithm 1 (BDD)
% S_p_tilde = sum_s A_s^T S_p^(s)_Neumann A_s (Neumann preconditioner)
% kappa(S_p_tilde^{-1} Sp) = O(1) => Numerical scalability
%
% Input:
% params - Parameter structure
%
% Output:
% info - Structure containing condition number results and graphical handle

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('Q2.6: Conditioning of preconditioned system\n');
    fprintf('========================================\n\n');

    N_sub_list   = [2, 4, 8, 16, 32, 64];
    Ne_sub_fixed = max(params.Ne_sub, 4);

    kappa_Sp        = zeros(length(N_sub_list), 1);
    kappa_SpTildeInv_Sp = zeros(length(N_sub_list), 1);

    fprintf('  %6s  %14s  %22s\n', 'N_sub', 'kappa(Sp)', 'kappa(Sptilde^{-1} Sp)');
    fprintf('  %s\n', repmat('-', 1, 48));

    for idx = 1:length(N_sub_list)
        Ns = N_sub_list(idx);

        p = params;
        p.N_sub  = Ns;
        p.Ne_sub = Ne_sub_fixed;
        p.Ne     = Ns * Ne_sub_fixed;
        p.h      = p.L / p.Ne;
        p.Nn     = p.Ne + 1;
        p.H      = Ne_sub_fixed * p.h;

        [Sp, Sp_tilde] = assemble_Sp_and_Sptilde(p);

        %% kappa(Sp): eigenvalues of Sp (SPD, restricted to range)
        eigs_Sp  = sort(real(eig(Sp)));
        tol_eig  = max(abs(eigs_Sp)) * 1e-10;
        pos_Sp   = eigs_Sp(eigs_Sp > tol_eig);
        kappa_Sp(idx) = pos_Sp(end) / pos_Sp(1);

        %% kappa(Sp_tilde^{-1} Sp)
        eigs_prec = sort(real(eig(Sp, Sp_tilde)));
        pos_prec  = eigs_prec(eigs_prec > max(abs(eigs_prec)) * 1e-10);
        kappa_SpTildeInv_Sp(idx) = pos_prec(end) / pos_prec(1);

        fprintf('  %6d  %14.4e  %22.4e\n', Ns, kappa_Sp(idx), kappa_SpTildeInv_Sp(idx));
    end

    %% log(kappa) ~ p * log(N_sub)
    log_N   = log(N_sub_list(:));
    log_kSp = log(kappa_Sp);
    p_fit_Sp = polyfit(log_N, log_kSp, 1);

    fprintf('\n  kappa(Sp)              ~ N_sub^%.2f  (theory: N_sub^2)\n', p_fit_Sp(1));
    fprintf('  kappa(Sptilde^{-1} Sp) ~ O(1)  (mean=%.3f, max=%.3f)\n', ...
            mean(kappa_SpTildeInv_Sp), max(kappa_SpTildeInv_Sp));

    %% ---- Figure: Double logarithmic coordinates ----
    if params.plot_results
        figure('Name', 'Q2.6 - Condition numbers', 'Position', [300 200 850 500]);

        loglog(N_sub_list, kappa_Sp, 'r-s', 'LineWidth', 2, 'MarkerSize', 9, ...
               'DisplayName', sprintf('\\kappa(S_p)  ~  N_{sub}^{%.1f}', p_fit_Sp(1)));
        hold on;
        loglog(N_sub_list, kappa_SpTildeInv_Sp, 'b-o', 'LineWidth', 2.5, 'MarkerSize', 10, ...
               'DisplayName', '\kappa(\tilde{S}_p^{-1} S_p)  =  O(1)  [BDD]');

        % Reference line N_sub^2
        ref_quad = N_sub_list.^2 * (kappa_Sp(1) / N_sub_list(1)^2);
        loglog(N_sub_list, ref_quad, 'r--', 'LineWidth', 1.4, ...
               'DisplayName', 'O(N_{sub}^2)  reference');

        % Ideal O(1)
        yline(1, 'g--', 'LineWidth', 1.8, 'DisplayName', 'Ideal: \kappa = 1');

        grid on; box on;
        xlabel('N_{sub}  (fixed h/H = 1/N_{e,sub})', 'FontSize', 12);
        ylabel('Condition number \kappa', 'FontSize', 12);
        title(sprintf(['Q2.6: Condition number of preconditioned system  (N_{e,sub}=%d)\n' ...
                       '\\kappa(\\tilde{S}_p^{-1} S_p) = O(1)  \\Rightarrow  BDD is numerically scalable'], ...
                      Ne_sub_fixed), 'FontSize', 12);
        legend('Location', 'northwest', 'FontSize', 11);
        xticks(N_sub_list);
        xticklabels(arrayfun(@num2str, N_sub_list, 'UniformOutput', false));

        if params.save_results
            output_dir = '../results/section2_1/figures/';
            if ~exist(output_dir, 'dir'); mkdir(output_dir); end
            saveas(gcf, fullfile(output_dir, 'Q2_6_precond_conditioning.png'));
            fprintf('\n  Image saved : Q2_6_precond_conditioning.png\n');
        end
    end

    %% Output
    info.N_sub_list             = N_sub_list;
    info.kappa_Sp               = kappa_Sp;
    info.kappa_SpTildeInv_Sp    = kappa_SpTildeInv_Sp;
    info.Ne_sub_fixed           = Ne_sub_fixed;
    info.power_fit_Sp           = p_fit_Sp(1);

end



%  Local functions: Assembling Sp and Sp_tilde
function [Sp, Sp_tilde] = assemble_Sp_and_Sptilde(params)

    N_sub  = params.N_sub;
    Ne_sub = params.Ne_sub;
    Nn     = params.Nn;
    E = params.E; S = params.S; h = params.h;
    ke = (E*S/h) * [1,-1;-1,1];

    if_nodes = ((1:N_sub-1)*Ne_sub+1)';
    n_if     = length(if_nodes);
    if_map   = zeros(Nn, 1);
    for k = 1:n_if; if_map(if_nodes(k)) = k; end

    Sp       = zeros(n_if, n_if);
    Sp_tilde = zeros(n_if, n_if);

    for s = 1:N_sub
        ns = (s-1)*Ne_sub+1; ne = s*Ne_sub+1; nloc = Ne_sub+1;

        % Assembly Substructure Stiffness Matrix
        K_sub = zeros(nloc);
        for e = 1:Ne_sub
            K_sub([e,e+1],[e,e+1]) = K_sub([e,e+1],[e,e+1]) + ke;
        end

        % Interface Nodes
        b_loc = []; b_glo = [];
        if if_map(ns) > 0; b_loc(end+1) = 1;    b_glo(end+1) = ns; end
        if if_map(ne) > 0; b_loc(end+1) = nloc;  b_glo(end+1) = ne; end
        b_loc = b_loc(:); b_glo = b_glo(:);

        if isempty(b_loc); continue; end

        %% --- Sp: Dirichlet BC (The first substructure is fixed on the left end) ---
        d_loc  = []; if s==1; d_loc = [1]; end
        all_b  = unique([b_loc; d_loc(:)]);
        i_loc  = setdiff((1:nloc)', all_b);

        if ~isempty(i_loc)
            Kii = K_sub(i_loc, i_loc);
            Kib = K_sub(i_loc, b_loc);
            Kbi = K_sub(b_loc, i_loc);
            Kbb = K_sub(b_loc, b_loc);
            Ss  = Kbb - Kbi * (Kii \ Kib);
        else
            Ss  = K_sub(b_loc, b_loc);
        end

        for ii = 1:length(b_glo)
            ki = if_map(b_glo(ii)); if ki==0; continue; end
            for jj = 1:length(b_glo)
                kj = if_map(b_glo(jj)); if kj==0; continue; end
                Sp(ki,kj) = Sp(ki,kj) + Ss(ii,jj);
            end
        end

        %% Sp_tilde: Neumann BC
        i_loc_N = setdiff((1:nloc)', b_loc);

        if ~isempty(i_loc_N)
            Kii_N = K_sub(i_loc_N, i_loc_N);
            Kib_N = K_sub(i_loc_N, b_loc);
            Kbi_N = K_sub(b_loc, i_loc_N);
            Kbb_N = K_sub(b_loc, b_loc);
            Ss_N  = Kbb_N - Kbi_N * pinv(Kii_N) * Kib_N;
        else
            Ss_N  = K_sub(b_loc, b_loc);
        end

        for ii = 1:length(b_glo)
            ki = if_map(b_glo(ii)); if ki==0; continue; end
            for jj = 1:length(b_glo)
                kj = if_map(b_glo(jj)); if kj==0; continue; end
                Sp_tilde(ki,kj) = Sp_tilde(ki,kj) + Ss_N(ii,jj);
            end
        end
    end
end