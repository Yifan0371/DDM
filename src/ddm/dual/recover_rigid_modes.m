function [u_global, rigid_info] = recover_rigid_modes(lambda_b, subs, F, params, feti_info)
% RECOVER_RIGID_MODES - Question 2.9: Recover the rigid body modes from Lagrange multipliers and plot them.

    N_sub = params.N_sub;
    Nn    = params.Nn;
    E = params.E; S = params.S; h = params.h;
    k0 = E*S/h;

    fprintf('====================================\n');
    fprintf('Rigid body modal recovery and displacement reconstruction (Q2.9)\n');
    fprintf('====================================\n');

    %% Identify the source of information
    if isfield(feti_info, 'subs_updated')
        subs = feti_info.subs_updated;
    end

    A_dual = feti_info.A_dual;
    nb_offsets = feti_info.nb_offsets;
    true_intf_nodes = feti_info.true_intf_nodes;

    %% Recovering rigid body modal amplitude
    lambda_b_diag = A_dual' * lambda_b;

    floating_subs = [];
    n_rigid_per_sub = zeros(N_sub, 1);
    for s = 1:N_sub
        nr_s = size(subs(s).Rb_dual, 2);
        n_rigid_per_sub(s) = nr_s;
        if nr_s > 0
            floating_subs = [floating_subs; s];
        end
    end
    n_floating = length(floating_subs);
    n_rigid_total = sum(n_rigid_per_sub);

    fprintf('Floating substructure: [%s]\n', num2str(floating_subs'));
    fprintf('Total rigid body modal number: %d\n\n', n_rigid_total);

    if isfield(feti_info, 'alpha') && ~isempty(feti_info.alpha)
        alpha_global = feti_info.alpha;
        fprintf('Use the alpha value from feti_info\n');
    else
        fprintf('alpha is calculated using displacement continuity...\n');
        n_eq = length(true_intf_nodes);
        C_mat = zeros(n_eq, n_rigid_total);
        d_vec = zeros(n_eq, 1);

        for j = 1:n_eq
            intf_node = true_intf_nodes(j);
            s_left = j; s_right = j + 1;
            nodes_L = subs(s_left).node_ids;
            [~, loc_L] = ismember(intf_node, nodes_L(subs(s_left).loc_b_idx_dual));
            nodes_R = subs(s_right).node_ids;
            [~, loc_R] = ismember(intf_node, nodes_R(subs(s_right).loc_b_idx_dual));
            idx_L = nb_offsets(s_left)+1 : nb_offsets(s_left+1);
            lam_L = lambda_b_diag(idx_L);
            ub_L_noR = subs(s_left).Sd_dual * (subs(s_left).bp_dual + lam_L);
            idx_R = nb_offsets(s_right)+1 : nb_offsets(s_right+1);
            lam_R = lambda_b_diag(idx_R);
            ub_R_noR = subs(s_right).Sd_dual * (subs(s_right).bp_dual + lam_R);
            u_L = 0; u_R = 0;
            if loc_L > 0, u_L = ub_L_noR(loc_L); end
            if loc_R > 0, u_R = ub_R_noR(loc_R); end
            d_vec(j) = u_L - u_R;
            alpha_off_L = sum(n_rigid_per_sub(1:s_left-1));
            nr_L = n_rigid_per_sub(s_left);
            if nr_L > 0 && loc_L > 0
                C_mat(j, alpha_off_L+1:alpha_off_L+nr_L) = subs(s_left).Rb_dual(loc_L, :);
            end
            alpha_off_R = sum(n_rigid_per_sub(1:s_right-1));
            nr_R = n_rigid_per_sub(s_right);
            if nr_R > 0 && loc_R > 0
                C_mat(j, alpha_off_R+1:alpha_off_R+nr_R) = ...
                    C_mat(j, alpha_off_R+1:alpha_off_R+nr_R) - subs(s_right).Rb_dual(loc_R, :);
            end
        end
        if n_rigid_total > 0
            if rank(C_mat) >= n_rigid_total
                alpha_global = C_mat \ d_vec;
            else
                alpha_global = pinv(C_mat) * d_vec;
            end
        else
            alpha_global = [];
        end
    end

    if ~isempty(alpha_global)
        fprintf('\nrigid body modal amplitude:\n');
        alpha_off = 0;
        for s = 1:N_sub
            nr_s = n_rigid_per_sub(s);
            if nr_s > 0
                alp_s = alpha_global(alpha_off+1:alpha_off+nr_s);
                fprintf('  Substructure %d: alpha = [%s]\n', s, num2str(alp_s'));
                fprintf('    Physical meaning: Overall translation = %.6e m\n', alp_s(1));
                alpha_off = alpha_off + nr_s;
            end
        end
    end

    %  Complete displacement field reconstruction
    
    u_global     = zeros(Nn, 1);
    u_particular = zeros(Nn, 1);   

    alpha_off = 0;
    for s = 1:N_sub
        nb_s = subs(s).nb_dual;
        nr_s = n_rigid_per_sub(s);
        nodes_s = subs(s).node_ids;

        idx = nb_offsets(s)+1 : nb_offsets(s+1);
        lam_s = lambda_b_diag(idx);

        if nr_s > 0
            alp_s = alpha_global(alpha_off+1:alpha_off+nr_s);
            alpha_off = alpha_off + nr_s;
        else
            alp_s = [];
        end

        % Interface displacement special solution
        ub_s_part = subs(s).Sd_dual * (subs(s).bp_dual + lam_s);

        % Complete interface displacement (with rigid body modes)
        ub_s = ub_s_part;
        if nr_s > 0
            ub_s = ub_s + subs(s).Rb_dual * alp_s;
        end

        % Internal displacement
        if ~isempty(subs(s).loc_i_idx_dual)
            ui_s      = subs(s).Kii_dual \ (subs(s).fi_dual - subs(s).Kib_dual * ub_s);
            ui_s_part = subs(s).Kii_dual \ (subs(s).fi_dual - subs(s).Kib_dual * ub_s_part);
        else
            ui_s      = [];
            ui_s_part = [];
        end

        % Global Assembly — Complete Solution
        loc_b_global = nodes_s(subs(s).loc_b_idx_dual);
        for j = 1:nb_s
            u_global(loc_b_global(j))     = ub_s(j);
            u_particular(loc_b_global(j)) = ub_s_part(j);
        end
        loc_i_global = nodes_s(subs(s).loc_i_idx_dual);
        for j = 1:length(ui_s)
            u_global(loc_i_global(j))     = ui_s(j);
            u_particular(loc_i_global(j)) = ui_s_part(j);
        end
    end
    u_global(1)     = 0;
    u_particular(1) = 0;

    %% verify
    mesh_x = linspace(0, params.L, Nn)';
    u_exact = (params.Fd / (params.E * params.S)) * mesh_x;
    error_L2   = norm(u_global - u_exact) / norm(u_exact);
    error_Linf = norm(u_global - u_exact, inf) / norm(u_exact, inf);

    %  Plotting
    fprintf('\nPlotting displacement field...\n');
    colors = lines(N_sub);

    figure('Name', 'Q2.9: Rigid Body Mode Recovery', 'Position', [100 100 850 480]);
    hold on;

    for s = 1:N_sub
        nodes_s = subs(s).node_ids;
        xs = mesh_x(nodes_s);
        nr_s = n_rigid_per_sub(s);

        % Particular solution (dashed, possibly discontinuous at interface)
        plot(xs*1e3, u_particular(nodes_s)*1e6, '--', ...
             'Color', colors(s,:), 'LineWidth', 1.5, ...
             'HandleVisibility', 'off');

        % Full solution (solid + circle, continuous)
        plot(xs*1e3, u_global(nodes_s)*1e6, '-o', ...
             'Color', colors(s,:), 'LineWidth', 2, 'MarkerSize', 5, ...
             'DisplayName', sprintf('Sub %d – full solution', s));

        % Annotation: sub 1 has no rigid body mode
        if nr_s == 0
            x_mid = mean(xs) * 1e3;
            y_mid = mean(u_global(nodes_s)) * 1e6;
            text(x_mid, y_mid * 0.6, 'particular = full', ...
                 'Color', colors(s,:), 'FontSize', 9, ...
                 'HorizontalAlignment', 'center', 'FontAngle', 'italic');
        end
    end

    % Legend entries for line styles
    plot(nan, nan, 'k--', 'LineWidth', 1.5, ...
         'DisplayName', 'Particular solution \tilde{u}^{(s)}  (no rigid mode)');
    plot(nan, nan, 'k-o', 'LineWidth', 2, ...
         'DisplayName', 'Full solution  \tilde{u}^{(s)} + R_b^{(s)}\alpha^{(s)}');

    % Interface markers
    for j = 1:length(true_intf_nodes)
        xline(mesh_x(true_intf_nodes(j))*1e3, ':k', 'LineWidth', 1, ...
              'HandleVisibility', 'off');
    end

    grid on;
    xlabel('Position x [mm]', 'FontSize', 13);
    ylabel('Displacement u(x) [\mum]', 'FontSize', 13);
    title('Q2.9: Rigid Body Mode Recovery – Particular vs Full Solution', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 10);
    set(gca, 'FontSize', 11);

    % ---- Figure 2: global displacement (numerical vs analytical) ----
    figure('Name', 'Q2.9: FETI Displacement Field', 'Position', [150 150 900 550]);

    subplot(2,1,1);
    plot(mesh_x*1e3, u_global*1e6, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 5, ...
         'DisplayName', 'FETI numerical');
    hold on;
    plot(mesh_x*1e3, u_exact*1e6, 'r--', 'LineWidth', 2, 'DisplayName', 'Analytical');
    for j = 1:length(true_intf_nodes)
        xline(mesh_x(true_intf_nodes(j))*1e3, ':k', 'LineWidth', 1, ...
              'HandleVisibility', 'off');
    end
    grid on;
    xlabel('Position x [mm]', 'FontSize', 12);
    ylabel('u(x) [\mum]', 'FontSize', 12);
    title('FETI – Displacement Field (with rigid body mode recovery)', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 11);
    set(gca, 'FontSize', 11);

    subplot(2,1,2);
    hold on;
    for s = 1:N_sub
        nodes_s = subs(s).node_ids;
        plot(mesh_x(nodes_s)*1e3, u_global(nodes_s)*1e6, '-o', ...
             'Color', colors(s,:), 'LineWidth', 1.5, 'MarkerSize', 5, ...
             'DisplayName', sprintf('Sub %d', s));
    end
    for j = 1:length(true_intf_nodes)
        xline(mesh_x(true_intf_nodes(j))*1e3, ':k', 'LineWidth', 1, ...
              'HandleVisibility', 'off');
    end
    grid on;
    xlabel('Position x [mm]', 'FontSize', 12);
    ylabel('u(x) [\mum]', 'FontSize', 12);
    title('Displacement per substructure', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 10);
    set(gca, 'FontSize', 11);

    % ---- Figure 3: pointwise error ----
    figure('Name', 'Q2.9: FETI Error', 'Position', [200 200 800 400]);
    semilogy(mesh_x*1e3, abs(u_global - u_exact), 'mo-', ...
             'LineWidth', 1.5, 'MarkerSize', 5);
    grid on;
    xlabel('Position x [mm]', 'FontSize', 12);
    ylabel('|u_{FETI} - u_{exact}| [m]', 'FontSize', 12);
    title('Q2.9: FETI – Pointwise Displacement Error', 'FontSize', 13);
    set(gca, 'FontSize', 11);

    %% Output rigid_info
    rigid_info.alpha           = alpha_global;
    rigid_info.n_floating      = n_floating;
    rigid_info.floating_subs   = floating_subs;
    rigid_info.error_L2        = error_L2;
    rigid_info.error_Linf      = error_Linf;
    rigid_info.u_exact         = u_exact;
    rigid_info.u_particular    = u_particular;
    rigid_info.n_rigid_per_sub = n_rigid_per_sub;

    fprintf('\nPlotting done.\n\n');

end