function [u_interface, u_global, info] = solve_schur_pcg(subs, F, params)
% SOLVE_SCHUR_PCG - Question 2.3: Solving the Primal Interface Problem using the Distributed Conjugate Gradient Method

%
% Problem Format:
% Sp * u_b = b_p
% Sp = A^diamond * Sp^diamond * (A^diamond)^T
% b_p = A^diamond * b_p^diamond

%
% Input:
% subs - Array of substructures (from generate_substructures)
% F - Global load vector [Nn x 1]
% params - Parameter structure (from config_params)

%
% Output:
% u_interface - Interface node displacements [n_if x 1]
% u_global - Global node displacements [Nn x 1]
% info - Solution information structure:
% .n_iter - Actual number of iterations
% .res_history - Residual history [n_iter+1 x 1]
% .error_L2 - L2 relative error with analytical solution
% .u_exact - Solution Analysis
% .n_interface - Interface DOF Count

    N_sub    = params.N_sub;
    Ne_sub   = params.Ne_sub;
    Nn       = params.Nn;
    tol      = params.tol;
    max_iter = params.max_iter;

    fprintf('------------------------------------\n');
    fprintf('Q2.3: Distributed CG  (N_sub=%d, Ne_sub=%d)\n', N_sub, Ne_sub);
    fprintf('------------------------------------\n');

    %  Create DOF information interface
    if_nodes = ((1:N_sub-1) * Ne_sub + 1).';   % [n_if x 1], 1-based
    n_if     = length(if_nodes);
    if_map = zeros(Nn, 1);
    for k = 1:n_if
        if_map(if_nodes(k)) = k;
    end


    %  Preprocess substructure local information
    sub_data = preprocess_subs(subs, F, N_sub, Ne_sub, Nn, if_map);

    %   Initialization (Algorithm 2, Initialization)
    u_b = zeros(n_if, 1);
    r_b = compute_residual(u_b, sub_data, if_map, if_nodes, n_if, Nn);
    d_b = r_b;

    rTr      = r_b' * r_b;
    r0_norm  = sqrt(rTr);
    res_hist = zeros(max_iter + 1, 1);
    res_hist(1) = r0_norm;

    fprintf('  iter     rel_res\n');
    fprintf('  %-6d   %.4e  (init)\n', 0, 1.0);

    %  CG iteration (Algorithm 2, main loop)

    n_iter = 0;

    for k = 1:max_iter
        Sp_d = compute_Sp_times_d(d_b, sub_data, if_map, if_nodes, n_if, Nn);

        dT_Sp_d = d_b' * Sp_d;
        alpha   = rTr / dT_Sp_d;

        u_b = u_b + alpha * d_b;

        r_b = r_b - alpha * Sp_d;

        rTr_new = r_b' * r_b;
        beta    = rTr_new / rTr;

        d_b = r_b + beta * d_b;
        rTr = rTr_new;

        n_iter  = k;
        rel_res = sqrt(rTr) / r0_norm;
        res_hist(k + 1) = sqrt(rTr);

        if mod(k, max(1, floor(max_iter/10))) == 0 || rel_res < tol
            fprintf('  %-6d   %.4e\n', k, rel_res);
        end

        if rel_res < tol
            break;
        end
    end

    res_hist = res_hist(1:n_iter+1);

 
    %  Restore global displacement field
 
    u_global = recover_global_displacement(u_b, sub_data, if_map, if_nodes, n_if, Nn);

    %  Calculate error

    x_nodes = linspace(0, params.L, Nn).';
    u_exact = (params.Fd / (params.E * params.S)) * x_nodes;
    err_L2  = norm(u_global - u_exact) / norm(u_exact);

    u_interface = u_b;

    info.n_iter      = n_iter;
    info.res_history = res_hist;
    info.error_L2    = err_L2;
    info.u_exact     = u_exact;
    info.n_interface = n_if;

    fprintf('  Converged: %d iters | L2 error = %.4e\n', n_iter, err_L2);
    fprintf('------------------------------------\n\n');

end


% Local Function 1: Preprocessing Substructure — Extracting Block Matrix and Local Information
% % Extraction for each subdomain:
% b_loc - Local index of interface nodes
% b_glo - Global index of interface nodes
% i_loc - Local index of internal nodes (excluding interface nodes and Dirichlet nodes)
% d_loc - Local index of Dirichlet fixed nodes
% active - [i_loc; b_loc], effective degrees of freedom participating in Schur complement calculation
% Kii, Kib, Kbi, Kbb - Blocks of subdomain stiffness matrix on active degrees of freedom

function sub_data = preprocess_subs(subs, F, N_sub, Ne_sub, Nn, if_map)

    sub_data = struct();

    for s = 1:N_sub
        n_start = (s-1) * Ne_sub + 1;
        n_end   =  s    * Ne_sub + 1;
        nloc    = Ne_sub + 1;

        K_sub = subs(s).K_sub;
        f_loc = F(n_start : n_end);

        b_glo = [];
        b_loc = [];
        if if_map(n_start) > 0
            b_glo(end+1) = n_start;
            b_loc(end+1) = 1;
        end
        if if_map(n_end) > 0
            b_glo(end+1) = n_end;
            b_loc(end+1) = nloc;
        end
        b_glo = b_glo(:);
        b_loc = b_loc(:);

        d_loc = [];
        if s == 1
            d_loc = [1];
        end

        all_b = unique([b_loc(:); d_loc(:)]);
        i_loc = setdiff((1:nloc)', all_b);

        active = [i_loc; b_loc];

        if ~isempty(i_loc) && ~isempty(b_loc)
            Kii = K_sub(i_loc, i_loc);
            Kib = K_sub(i_loc, b_loc);
            Kbi = K_sub(b_loc, i_loc);
            Kbb = K_sub(b_loc, b_loc);
        elseif isempty(i_loc)
            Kii = []; Kib = []; Kbi = []; Kbb = K_sub(b_loc, b_loc);
        else
            Kii = K_sub(i_loc, i_loc);
            Kib = zeros(length(i_loc), 0);
            Kbi = zeros(0, length(i_loc));
            Kbb = zeros(0, 0);
        end

        sub_data(s).n_start = n_start;
        sub_data(s).n_end   = n_end;
        sub_data(s).nloc    = nloc;
        sub_data(s).K_sub   = K_sub;
        sub_data(s).f_loc   = f_loc;
        sub_data(s).b_loc   = b_loc;
        sub_data(s).b_glo   = b_glo;
        sub_data(s).i_loc   = i_loc;
        sub_data(s).d_loc   = d_loc;
        sub_data(s).active  = active;   % 新增: active DOF 索引
        sub_data(s).Kii     = Kii;
        sub_data(s).Kib     = Kib;
        sub_data(s).Kbi     = Kbi;
        sub_data(s).Kbb     = Kbb;
    end
end



% Local function 2: Calculate the residual r_b = b_p - Sp * u_b
% % Strictly corresponds to Algorithm 2 Initialization steps 1-2:
% Step 1: u_i^(s) = Kii^{-1} (f_i^(s) - Kib u_b^(s))
% Step 2: r_b^(s) = -(K^(s)[active, active] * [u_i; u_b] - f[active])
% Interface line (corresponding to b_loc)
% % Key modification: Use K_sub(active, active) instead of the full K_sub,

function r_b = compute_residual(u_b, sub_data, if_map, if_nodes, n_if, Nn)

    N_sub   = length(sub_data);
    r_local = zeros(Nn, 1);

    for s = 1:N_sub
        sd     = sub_data(s);
        b_loc  = sd.b_loc;
        b_glo  = sd.b_glo;
        i_loc  = sd.i_loc;
        active = sd.active;   % = [i_loc; b_loc]

        if isempty(b_loc); continue; end

        u_b_loc = zeros(length(b_loc), 1);
        for j = 1:length(b_glo)
            k_if = if_map(b_glo(j));
            u_b_loc(j) = u_b(k_if);
        end

        %   u_i^(s) = Kii^{-1} (f_i^(s) - Kib u_b^(s))
        if ~isempty(i_loc)
            u_i_loc = sd.Kii \ (sd.f_loc(i_loc) - sd.Kib * u_b_loc);
        else
            u_i_loc = [];
        end

        %   [K_ii K_ib] [u_i]   [f_i]       [       0        ]
        %   [K_bi K_bb] [u_b] - [f_b]  =    [-r_b^(s)]
        u_active = [u_i_loc; u_b_loc];   
        f_active = sd.f_loc(active);

        n_i = length(i_loc);
        n_b = length(b_loc);
        K_b_active = sd.K_sub(b_loc, active);   % [n_b x (n_i+n_b)]

        residual_b = K_b_active * u_active - f_active(n_i+1:end);

        for j = 1:length(b_glo)
            r_local(b_glo(j)) = r_local(b_glo(j)) - residual_b(j);
        end
    end

    r_b = r_local(if_nodes);
end



% Local Function 3: Distributed Computation Sp * d_b
%
% Strictly corresponds to Algorithm 2 iteration step:
% — d_b^(s) = (A^(s))^T d_b
% — d_i^(s) = -Kii^{-1} Kib d_b^(s) (Local Dirichlet problem)
% — S_p^(s) d_b^(s) = K^(s)[b_loc, active] * [d_i; d_b]
% (Interface row of local matrix-vector product)
% — Sp d_b = A^diamond (Sp^diamond d_b^diamond) (Assembly)

function Sp_d = compute_Sp_times_d(d_b, sub_data, if_map, if_nodes, n_if, Nn)

    N_sub   = length(sub_data);
    Sp_d_gl = zeros(Nn, 1);

    for s = 1:N_sub
        sd     = sub_data(s);
        b_loc  = sd.b_loc;
        b_glo  = sd.b_glo;
        i_loc  = sd.i_loc;
        active = sd.active;   % = [i_loc; b_loc]

        if isempty(b_loc); continue; end

        d_b_loc = zeros(length(b_loc), 1);
        for j = 1:length(b_glo)
            k_if = if_map(b_glo(j));
            d_b_loc(j) = d_b(k_if);
        end

        if ~isempty(i_loc)
            d_i_loc = -sd.Kii \ (sd.Kib * d_b_loc);
        else
            d_i_loc = [];
        end

        d_active    = [d_i_loc; d_b_loc];
        K_b_active  = sd.K_sub(b_loc, active);   % [n_b x (n_i+n_b)]
        Sp_d_loc    = K_b_active * d_active;      % = S_p^(s) d_b^(s)

        for j = 1:length(b_glo)
            Sp_d_gl(b_glo(j)) = Sp_d_gl(b_glo(j)) + Sp_d_loc(j);
        end
    end

    Sp_d = Sp_d_gl(if_nodes);
end


% Local function 4: Recover the global displacement field from the interface displacement u_b
% % Solve the final Dirichlet problem for each subdomain:
% u_i^(s) = Kii^{-1} (f_i^(s) - Kib u_b^(s))

function u_global = recover_global_displacement(u_b, sub_data, if_map, if_nodes, n_if, Nn)

    N_sub    = length(sub_data);
    u_global = zeros(Nn, 1);

    for k = 1:n_if
        u_global(if_nodes(k)) = u_b(k);
    end

    for s = 1:N_sub
        sd    = sub_data(s);
        b_loc = sd.b_loc;
        b_glo = sd.b_glo;
        i_loc = sd.i_loc;

        if isempty(i_loc); continue; end

        u_b_loc = zeros(length(b_loc), 1);
        for j = 1:length(b_glo)
            k_if = if_map(b_glo(j));
            u_b_loc(j) = u_b(k_if);
        end

        u_i = sd.Kii \ (sd.f_loc(i_loc) - sd.Kib * u_b_loc);

        node_ids = (sd.n_start : sd.n_end).';
        u_global(node_ids(i_loc)) = u_i;
    end
end