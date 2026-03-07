function [u_interface, u_global, info] = solve_bdd_pcg(subs, F, params)
% SOLVE_BDD_PCG - Question 2.5: BDD preconditional conjugate gradient method (with verification)
%
% Changes vs your version:
%   1) Fix bug: GtSpG_inv undefined when n_rigid = 0
%   2) Cache Sp_tilde_pinv = pinv(Sp_tilde) once
%   3) Add init residual check -> explain "0 iteration" phenomena
%   4) Add kappa_eff estimate on Range(P) for (P^T S_p P)
%

    N_sub    = params.N_sub;
    Ne_sub   = params.Ne_sub;
    Nn       = params.Nn;
    tol      = params.tol;
    max_iter = params.max_iter;

    fprintf('------------------------------------\n');
    fprintf('Q2.5: BDD PCG  (N_sub=%d, Ne_sub=%d)\n', N_sub, Ne_sub);
    fprintf('------------------------------------\n');


    % Interface DOFs

    if_nodes = ((1:N_sub-1) * Ne_sub + 1).';
    n_if     = length(if_nodes);

    if_map = zeros(Nn, 1);
    for k = 1:n_if
        if_map(if_nodes(k)) = k;
    end


    % Preprocess substructures

    sub_data = preprocess_subs(subs, F, N_sub, Ne_sub, Nn, if_map);


    % Step 1: build b_p and G

    b_p = compute_b_p(sub_data, if_map, if_nodes, n_if, Nn);
    G   = build_G_matrix(sub_data, if_map, n_if, N_sub);

    n_rigid = size(G, 2);
    fprintf('  n_interface = %d,  n_rigid_modes = %d\n', n_if, n_rigid);


    % Step 2: build Neumann preconditioner Sp_tilde (assembled)

    Sp_tilde = build_Sp_tilde(sub_data, if_map, if_nodes, n_if, Nn);
    Sp_tilde_pinv = pinv(Sp_tilde);  % cache once (verification-friendly)


    % Step 3: projector P and coarse correction u0

    SpG = zeros(n_if, n_rigid);
    for j = 1:n_rigid
        SpG(:, j) = compute_Sp_times_d(G(:,j), sub_data, if_map, if_nodes, n_if, Nn);
    end

    GtSpG = G.' * SpG;    % [n_rigid x n_rigid]

    if n_rigid > 0
        % Safer than inv: solve linear system
        % GtSpG_inv = inv(GtSpG);
        GtSpG_inv = (GtSpG \ eye(size(GtSpG)));
        u0 = G * (GtSpG_inv * (G.' * b_p));
    else
        u0 = zeros(n_if, 1);
        GtSpG_inv = [];  % FIX: avoid undefined variable later
    end


    % Step 4: BDD PCG iterations

    % r(0) = P^T (b_p - Sp u0)
    Sp_u0 = compute_Sp_times_d(u0, sub_data, if_map, if_nodes, n_if, Nn);
    r     = apply_PT(b_p - Sp_u0, G, GtSpG_inv, SpG, n_if);

    % Print init residual norm (raw)
    r_norm0_raw = norm(r);
    fprintf('  init ||r|| (projected) = %.4e\n', r_norm0_raw);

    % z(0) = P * Sp_tilde^+ * r(0)
    z = apply_precond_projected(r, Sp_tilde_pinv, G, GtSpG_inv, SpG, n_if);
    rz = r.' * z;

    if rz <= 0
        warning('Init r^T z is non-positive (%.3e). Preconditioner/projection may be inconsistent.', rz);
    end

    r0_norm = sqrt(max(rz, 0));
    info.init_rel_res = r0_norm; % mainly for debug

    % If already converged at init (this explains "0-iter" blue curve)
    if r0_norm < tol
        fprintf('  NOTE: converged at initialization (0 PCG iterations).\n');
        u_interface = u0;     % u* = 0, so u_b = u0
        u_global = recover_global(u_interface, sub_data, if_map, if_nodes, n_if, Nn);

        % error
        x_nodes = linspace(0, params.L, Nn).';
        u_exact = (params.Fd / (params.E * params.S)) * x_nodes;
        err_L2  = norm(u_global - u_exact) / norm(u_exact);

        % kappa_eff estimate (optional, still useful)
        info.kappa_eff = estimate_kappa_eff( ...
            @(v) apply_P(v, G, GtSpG_inv, SpG, n_if), ...
            @(v) apply_PT(v, G, GtSpG_inv, SpG, n_if), ...
            @(v) compute_Sp_times_d(v, sub_data, if_map, if_nodes, n_if, Nn), ...
            n_if);

        info.n_iter        = 0;
        info.n_iter_main   = 0;
        info.res_history   = 1.0;
        info.error_L2      = err_L2;
        info.u_exact       = u_exact;
        info.n_interface   = n_if;
        info.n_rigid       = n_rigid;
        info.u0            = u0;

        fprintf('  BDD converged: 0 iters | L2 err = %.4e | kappa_eff ~= %.3e\n', err_L2, info.kappa_eff);
        fprintf('------------------------------------\n\n');
        return;
    end

    % Initialize PCG
    d = z;
    res_hist = zeros(max_iter + 1, 1);
    res_hist(1) = 1.0;

    p_hist = {};
    d_hist = {};

    u_star = zeros(n_if, 1);
    n_iter = 0;

    fprintf('  iter    rel_res\n');
    fprintf('  %-5d   %.4e\n', 0, 1.0);

    for i = 1:max_iter

        % p_i = P^T Sp d
        Sp_d = compute_Sp_times_d(d, sub_data, if_map, if_nodes, n_if, Nn);
        p_i  = apply_PT(Sp_d, G, GtSpG_inv, SpG, n_if);

        % alpha
        dTp = d.' * p_i;
        if abs(dTp) < 1e-35
            fprintf('  WARNING: d^T p is too small -> break.\n');
            break;
        end
        alpha = (r.' * d) / dTp;

        % update u*
        u_star = u_star + alpha * d;

        % update residual
        r = r - alpha * p_i;

        % precondition + project
        z = apply_precond_projected(r, Sp_tilde_pinv, G, GtSpG_inv, SpG, n_if);

        % convergence check in energy norm (r^T z)
        rz_new = r.' * z;
        rel_res = sqrt(max(rz_new, 0)) / r0_norm;

        n_iter = i;
        res_hist(i + 1) = rel_res;

        if mod(i, max(1, floor(max_iter/8))) == 0 || rel_res < tol
            fprintf('  %-5d   %.4e\n', i, rel_res);
        end

        if rel_res < tol
            break;
        end

        % full reorthogonalization
        p_hist{end+1} = p_i;
        d_hist{end+1} = d;

        d_new = z;
        for j = 1:length(p_hist)
            djTpj = d_hist{j}.' * p_hist{j};
            if abs(djTpj) < 1e-35
                continue;
            end
            beta_j = -(z.' * p_hist{j}) / djTpj;
            d_new  = d_new + beta_j * d_hist{j};
        end

        d = d_new;
    end

    res_hist = res_hist(1:n_iter+1);


    % Step 5: recover final interface solution u_b = u0 + P u*

    u_interface = u0 + apply_P(u_star, G, GtSpG_inv, SpG, n_if);

    % Step 6: recover global displacement

    u_global = recover_global(u_interface, sub_data, if_map, if_nodes, n_if, Nn);

    % Step 7: error + diagnostics

    x_nodes = linspace(0, params.L, Nn).';
    u_exact = (params.Fd / (params.E * params.S)) * x_nodes;
    err_L2  = norm(u_global - u_exact) / norm(u_exact);

    % Effective condition number on Range(P)
    kappa_eff = estimate_kappa_eff( ...
        @(v) apply_P(v, G, GtSpG_inv, SpG, n_if), ...
        @(v) apply_PT(v, G, GtSpG_inv, SpG, n_if), ...
        @(v) compute_Sp_times_d(v, sub_data, if_map, if_nodes, n_if, Nn), ...
        n_if);

    info.n_iter        = n_iter;
    info.n_iter_main   = n_iter;
    info.res_history   = res_hist;
    info.error_L2      = err_L2;
    info.u_exact       = u_exact;
    info.n_interface   = n_if;
    info.n_rigid       = n_rigid;
    info.u0            = u0;
    info.kappa_eff     = kappa_eff;

    fprintf('  BDD converged: %d iters | L2 err = %.4e | kappa_eff ~= %.3e\n', n_iter, err_L2, kappa_eff);
    fprintf('------------------------------------\n\n');
end


%  Local function 1: preprocess substructures

function sub_data = preprocess_subs(subs, F, N_sub, Ne_sub, Nn, if_map)
    sub_data = struct();
    for s = 1:N_sub
        n_start = (s-1)*Ne_sub + 1;
        n_end   =  s   *Ne_sub + 1;
        nloc    = Ne_sub + 1;

        K_sub = subs(s).K_sub;
        f_loc = F(n_start:n_end);

        % interface nodes
        b_glo = []; b_loc = [];
        if if_map(n_start) > 0; b_glo(end+1)=n_start; b_loc(end+1)=1;    end
        if if_map(n_end)   > 0; b_glo(end+1)=n_end;   b_loc(end+1)=nloc; end
        b_glo=b_glo(:); b_loc=b_loc(:);

        % Dirichlet nodes (left end fixed)
        d_loc = [];
        if s==1; d_loc=[1]; end

        % interior nodes (Dirichlet elimination)
        all_b  = unique([b_loc(:); d_loc(:)]);
        i_loc  = setdiff((1:nloc)', all_b);

        % blocks (Dirichlet)
        if ~isempty(i_loc) && ~isempty(b_loc)
            Kii=K_sub(i_loc,i_loc); Kib=K_sub(i_loc,b_loc);
            Kbi=K_sub(b_loc,i_loc); Kbb=K_sub(b_loc,b_loc);
        elseif isempty(i_loc)
            Kii=[]; Kib=[]; Kbi=[]; Kbb=K_sub(b_loc,b_loc);
        else
            Kii=K_sub(i_loc,i_loc); Kib=zeros(length(i_loc),0);
            Kbi=zeros(0,length(i_loc)); Kbb=zeros(0,0);
        end

        % interior nodes for Neumann preconditioner (do not eliminate Dirichlet)
        i_loc_N = setdiff((1:nloc)', b_loc);
        if ~isempty(i_loc_N) && ~isempty(b_loc)
            Kii_N=K_sub(i_loc_N,i_loc_N); Kib_N=K_sub(i_loc_N,b_loc);
            Kbi_N=K_sub(b_loc,i_loc_N);   Kbb_N=K_sub(b_loc,b_loc);
        else
            Kii_N=[]; Kib_N=[]; Kbi_N=[]; Kbb_N=K_sub(b_loc,b_loc);
        end

        is_floating = isempty(d_loc);

        sub_data(s).n_start     = n_start;
        sub_data(s).n_end       = n_end;
        sub_data(s).nloc        = nloc;
        sub_data(s).K_sub       = K_sub;
        sub_data(s).f_loc       = f_loc;
        sub_data(s).b_loc       = b_loc;
        sub_data(s).b_glo       = b_glo;
        sub_data(s).i_loc       = i_loc;
        sub_data(s).d_loc       = d_loc;
        sub_data(s).Kii         = Kii;
        sub_data(s).Kib         = Kib;
        sub_data(s).Kbi         = Kbi;
        sub_data(s).Kbb         = Kbb;
        sub_data(s).i_loc_N     = i_loc_N;
        sub_data(s).Kii_N       = Kii_N;
        sub_data(s).Kib_N       = Kib_N;
        sub_data(s).Kbi_N       = Kbi_N;
        sub_data(s).Kbb_N       = Kbb_N;
        sub_data(s).is_floating = is_floating;
    end
end


%  Local function 2: compute b_p (interface residual at u_b = 0)

function b_p = compute_b_p(sub_data, if_map, if_nodes, n_if, Nn)
    r_local = zeros(Nn, 1);
    for s = 1:length(sub_data)
        sd = sub_data(s);
        if isempty(sd.b_loc); continue; end
        u_full = zeros(sd.nloc, 1);
        if ~isempty(sd.i_loc)
            u_full(sd.i_loc) = sd.Kii \ (sd.f_loc(sd.i_loc) - sd.Kib * u_full(sd.b_loc));
        end
        Ku_f = sd.K_sub * u_full - sd.f_loc;
        for j = 1:length(sd.b_glo)
            r_local(sd.b_glo(j)) = r_local(sd.b_glo(j)) - Ku_f(sd.b_loc(j));
        end
    end
    b_p = r_local(if_nodes);
end


%  Local function 3: build G matrix (rigid body modes of floating subdomains)

function G = build_G_matrix(sub_data, if_map, n_if, N_sub)
    G_cols = {};
    for s = 1:N_sub
        sd = sub_data(s);
        if ~sd.is_floating; continue; end
        g = zeros(n_if, 1);
        for j = 1:length(sd.b_glo)
            k_if = if_map(sd.b_glo(j));
            if k_if > 0; g(k_if) = 1.0; end
        end
        G_cols{end+1} = g;
    end
    if isempty(G_cols)
        G = zeros(n_if, 0);
    else
        G = cell2mat(G_cols);
    end
end

%  Local function 4: build Neumann preconditioner Sp_tilde (assembled)

function Sp_tilde = build_Sp_tilde(sub_data, if_map, if_nodes, n_if, Nn)
    Sp_tilde = zeros(n_if, n_if);
    for s = 1:length(sub_data)
        sd = sub_data(s);
        if isempty(sd.b_loc); continue; end

        % S_N = Kbb - Kbi * pinv(Kii) * Kib  (Neumann Schur, pseudo-inverse)
        if ~isempty(sd.i_loc_N)
            Kii_N_pinv = pinv(sd.Kii_N);
            Ss_N = sd.Kbb_N - sd.Kbi_N * Kii_N_pinv * sd.Kib_N;
        else
            Ss_N = sd.Kbb_N;
        end

        for ii = 1:length(sd.b_glo)
            row = if_map(sd.b_glo(ii));
            if row == 0; continue; end
            for jj = 1:length(sd.b_glo)
                col = if_map(sd.b_glo(jj));
                if col == 0; continue; end
                Sp_tilde(row, col) = Sp_tilde(row, col) + Ss_N(ii, jj);
            end
        end
    end
end


%  Local function 5: distributed compute Sp * d (no assembly of Sp)

function Sp_d = compute_Sp_times_d(d, sub_data, if_map, if_nodes, n_if, Nn)
    Sp_d_gl = zeros(Nn, 1);
    for s = 1:length(sub_data)
        sd = sub_data(s);
        if isempty(sd.b_loc); continue; end
        d_full = zeros(sd.nloc, 1);

        for j = 1:length(sd.b_glo)
            k_if = if_map(sd.b_glo(j));
            if k_if > 0
                d_full(sd.b_loc(j)) = d(k_if);
            end
        end

        if ~isempty(sd.i_loc)
            d_full(sd.i_loc) = -sd.Kii \ (sd.Kib * d_full(sd.b_loc));
        end

        Kd = sd.K_sub * d_full;

        for j = 1:length(sd.b_glo)
            Sp_d_gl(sd.b_glo(j)) = Sp_d_gl(sd.b_glo(j)) + Kd(sd.b_loc(j));
        end
    end
    Sp_d = Sp_d_gl(if_nodes);
end


%  Local function 6-8: projectors
%  P  v  = v - G (G^T Sp G)^{-1} G^T Sp v
%  P^T v = v - SpG (G^T Sp G)^{-1} G^T v

function Pv = apply_P(v, G, GtSpG_inv, SpG, n_if)
    if size(G, 2) == 0
        Pv = v; 
        return;
    end
    Pv = v - G * (GtSpG_inv * (SpG.' * v));
end

function PTv = apply_PT(v, G, GtSpG_inv, SpG, n_if)
    if size(G, 2) == 0
        PTv = v;
        return;
    end
    PTv = v - SpG * (GtSpG_inv * (G.' * v));
end

function z = apply_precond_projected(r, Sp_tilde_pinv, G, GtSpG_inv, SpG, n_if)
    % z = P * (Sp_tilde^{+} r)
    Spt_pinv_r = Sp_tilde_pinv * r;
    z = apply_P(Spt_pinv_r, G, GtSpG_inv, SpG, n_if);
end


%  Local function 9: recover global displacement

function u_global = recover_global(u_b, sub_data, if_map, if_nodes, n_if, Nn)
    u_global = zeros(Nn, 1);

    % interface
    for k = 1:n_if
        u_global(if_nodes(k)) = u_b(k);
    end

    for s = 1:length(sub_data)
        sd = sub_data(s);
        u_b_loc = zeros(sd.nloc, 1);

        for j = 1:length(sd.b_glo)
            k_if = if_map(sd.b_glo(j));
            if k_if > 0
                u_b_loc(sd.b_loc(j)) = u_b(k_if);
            end
        end

        if ~isempty(sd.i_loc)
            u_i = sd.Kii \ (sd.f_loc(sd.i_loc) - sd.Kib * u_b_loc(sd.b_loc));
            node_ids = (sd.n_start:sd.n_end).';
            u_global(node_ids(sd.i_loc)) = u_i;
        end
    end
end

%  Local function 10: estimate effective cond number on Range(P)

function kappa_eff = estimate_kappa_eff(applyP, applyPT, applySp, n_if)
% Estimate effective cond number of A = P^T S_p P restricted to Range(P)

    % Build basis for Range(P) using projected canonical vectors
    V = zeros(n_if, n_if);
    cnt = 0;
    for j = 1:n_if
        e = zeros(n_if,1); e(j) = 1;
        v = applyP(e);
        nv = norm(v);
        if nv > 1e-12
            cnt = cnt + 1;
            V(:,cnt) = v / nv;
        end
    end
    V = V(:,1:cnt);
    if cnt == 0
        kappa_eff = 1;
        return;
    end

    [Q,~] = qr(V,0);
    m = size(Q,2);

    % Build small matrix B = Q^T (P^T S_p P) Q
    B = zeros(m,m);
    for j = 1:m
        qj  = Q(:,j);
        Aqj = applyPT(applySp(applyP(qj)));
        B(:,j) = Q.' * Aqj;
    end

    B = 0.5*(B + B.'); % symmetrize
    lam = sort(real(eig(B)));

    if isempty(lam)
        kappa_eff = 1;
        return;
    end

    % keep positive eigenvalues only
    lam_max = max(lam);
    lam_pos = lam(lam > 1e-10*max(1,lam_max));

    if isempty(lam_pos)
        kappa_eff = 1;
    else
        kappa_eff = max(lam_pos) / min(lam_pos);
    end
end