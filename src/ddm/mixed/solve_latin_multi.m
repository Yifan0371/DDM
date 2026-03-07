function [u_sub, lam_sub, info] = solve_latin_multi(subs, F, params)
% SOLVE_LATIN_MULTI  Q2.13 - Multiscale LaTIn mixed DDM
% Inputs/Outputs: same as solve_latin_mono (minus k_plus/k_minus args)
%

    N_sub    = params.N_sub;
    Ne_sub   = params.Ne_sub;
    Nn       = params.Nn;
    tol      = params.tol;
    max_iter = params.max_iter;
    n_if     = N_sub - 1;
    mu       = 0.8;   % relaxation parameter (Algorithm 5, step 2)

    % Use theoretical optimal: k = E·S/H
    k_plus  = params.E * params.S / params.H;
    k_minus = k_plus;
    fprintf('  LaTIn multi: N_sub=%d, k=%.3e (=E*S/H)\n', N_sub, k_plus);

    sd = preprocess_sd(subs, F, N_sub, Ne_sub);

    %  PRECOMPUTE macro operator [L^F]_{e^M} and b_ext_macro
    %
    %  [L^F_E] = Schur complement of (K_E + k⁻_E) at interface DOF of E
    %  [L^F]_{e^M} = assembled sum of [L^F_E] contributions
    %
    %  b_ext_macro = assembled condensed external load on interfaces
    %    (standard Schur RHS from f_ext, using (K+k⁻) system)
    %
    [L_macro, b_ext_macro] = build_macro_operator(sd, subs, k_minus, n_if);
    fprintf('  Macro system: %d×%d, cond=%.2e\n', n_if, n_if, cond(L_macro));

    % INITIALIZATION  ŝ^{-1/2} = 0
    W_hat    = zeros(n_if, 1);
    F_hat    = zeros(n_if, 1);
    % Preliminary local stage (with s_{-1} = 0, gives ŝ^{-1/2} = 0 too)
    % => no-op, already initialized to 0

    % Previous iteration state for relaxation
    W_prev = zeros(n_if, 1);
    F_prev = zeros(n_if, 1);

    eta_hist = zeros(max_iter, 1);
    n_iter   = 0;
    W_macro  = zeros(n_if, 1);  % will be filled in loop

    for iter = 1:max_iter

     
        %  STEP 1a: MICRO SOLVE – data problem
        %

        W_breve = zeros(n_if, 2);
        F_breve = zeros(n_if, 2);

        for s = 1:N_sub
            nloc  = sd(s).nloc;
            K_loc = sd(s).K_sub;
            f_loc = sd(s).f_orig;
            for j = 1:numel(sd(s).if_idx)
                i_if = sd(s).if_idx(j);  sgn = sd(s).if_sgn(j);  bloc = sd(s).if_loc(j);
                K_loc(bloc,bloc) = K_loc(bloc,bloc) + k_minus;
                f_loc(bloc) = f_loc(bloc) + sgn*F_hat(i_if) + k_minus*W_hat(i_if);
            end
            u_breve = solve_sub_dirichlet(K_loc, f_loc, sd(s));
            for j = 1:numel(sd(s).if_idx)
                i_if = sd(s).if_idx(j);  sgn = sd(s).if_sgn(j);  bloc = sd(s).if_loc(j);
                w = u_breve(bloc);
                f_s = sgn*F_hat(i_if) + k_minus*(W_hat(i_if) - w);  % E⁻
                if sgn == 1
                    W_breve(i_if,1) = w;  F_breve(i_if,1) = f_s;
                else
                    W_breve(i_if,2) = w;  F_breve(i_if,2) = -f_s;
                end
            end
        end

        %  STEP 1b: MACRO SOLVE
   
        W_macro = L_macro \ b_ext_macro;   % Solve Sp * W_macro = b_ext (constant!)

        %  STEP 1c: MICRO SOLVE – macro-loading problem
        %
        %  Solve (K_E + k⁻) ũ_E = k⁻·(W̃^M_E - ǔ_E|_Γ)

        W_tilde = zeros(n_if, 2);
        F_tilde = zeros(n_if, 2);

        for s = 1:N_sub
            nloc  = sd(s).nloc;
            K_loc = sd(s).K_sub;
            f_macro = zeros(nloc, 1);
            for j = 1:numel(sd(s).if_idx)
                i_if = sd(s).if_idx(j);  sgn = sd(s).if_sgn(j);  bloc = sd(s).if_loc(j);
                K_loc(bloc,bloc) = K_loc(bloc,bloc) + k_minus;
                % RHS = k⁻ * (W̃^M - ǔ_E|_Γ)  [the displacement gap to correct]
                if sgn == 1
                    gap = W_macro(i_if) - W_breve(i_if,1);   % LEFT sub gap
                else
                    gap = W_macro(i_if) - W_breve(i_if,2);   % RIGHT sub gap
                end
                f_macro(bloc) = f_macro(bloc) + k_minus * gap;
            end
            u_tilde = solve_sub_dirichlet(K_loc, f_macro, sd(s));
            for j = 1:numel(sd(s).if_idx)
                i_if = sd(s).if_idx(j);  sgn = sd(s).if_sgn(j);  bloc = sd(s).if_loc(j);
                w_t = u_tilde(bloc);
                % F̃_E = k⁻ * (gap - ũ_E|_Γ) = remaining correction after micro solve
                if sgn == 1
                    gap = W_macro(i_if) - W_breve(i_if,1);
                    W_tilde(i_if,1) = w_t;
                    F_tilde(i_if,1) = k_minus * (gap - w_t);
                else
                    gap = W_macro(i_if) - W_breve(i_if,2);
                    W_tilde(i_if,2) = w_t;
                    F_tilde(i_if,2) = k_minus * (gap - w_t);
                end
            end
        end

        %  STEP 1d: Sum  u_E = ǔ_E + ũ_E,   F_E = Ǧ_E + F̃_E

        W_curr = W_breve + W_tilde;
        F_curr = F_breve + F_tilde;

        %  STEP 2: RELAXATION  s_n ← µ s_n + (1-µ) s_{n-1}

        if iter > 1
            W_curr = mu*W_curr + (1-mu)*W_prev;
            F_curr = mu*F_curr + (1-mu)*F_prev;
        end
        W_prev = W_curr;
        F_prev = F_curr;


        %  STEP 3: LOCAL STAGE (E⁺ projection onto Gamma)

        W_hat_new = zeros(n_if,1);
        F_hat_new = zeros(n_if,1);
        for i = 1:n_if
            WL=W_curr(i,1); WR=W_curr(i,2);
            FL=F_curr(i,1); FR=F_curr(i,2);
            W_hat_new(i) = (WL + WR) / 2;
            F_hat_new(i) = (FL + FR)/2 - k_plus/2*(WL - WR);  % MINUS sign
        end

        %  STEP 4: CONVERGENCE CRITERION

        dW = W_hat_new - W_hat;
        dF = F_hat_new - F_hat;
        scale_W = max(norm(W_hat_new), 1e-30);
        scale_F = max(norm(F_hat_new)/k_plus, 1e-30);
        eta = sqrt( norm(dW)^2/scale_W^2 + (norm(dF)/k_plus)^2/scale_F^2 ) / sqrt(2);

        eta_hist(iter) = eta;
        n_iter = iter;
        W_hat = W_hat_new;
        F_hat = F_hat_new;
        if eta < tol; break; end
    end
    eta_hist = eta_hist(1:n_iter);

    % Recover solution 
    [u_sub, lam_sub, u_global] = recover_solution(sd, subs, W_hat, F, N_sub, Ne_sub, Nn);

    x_nodes = linspace(0, params.L, Nn).';
    u_exact = (params.Fd/(params.E*params.S)) * x_nodes;
    err_L2  = norm(u_global - u_exact) / norm(u_exact);

    fprintf('  -> %d iters, L2 err=%.2e\n', n_iter, err_L2);
    fprintf('  (1D degenerate: macro=full interface space => ~1 iter)\n');

    info.n_iter       = n_iter;
    info.eta_history  = eta_hist;
    info.error_L2     = err_L2;
    info.u_global     = u_global;
    info.u_exact      = u_exact;
    info.W_hat        = W_hat;
    info.F_hat        = F_hat;
    info.W_macro      = W_macro;
    info.L_macro      = L_macro;
    info.b_ext_macro  = b_ext_macro;
    info.k_plus       = k_plus;
    info.k_minus      = k_minus;
end



function [L_macro, b_ext] = build_macro_operator(sd, subs, k_minus, n_if)
% Assemble [L^F]_{e^M} = STANDARD primal Schur Sp (K_E, NO k⁻ modification)
% and b_ext = assembled condensed external loads (always + sign).


    N_sub  = numel(sd);
    L_macro = zeros(n_if, n_if);
    b_ext   = zeros(n_if, 1);

    for s = 1:N_sub
        nloc   = sd(s).nloc;
        K_orig = subs(s).K_sub;   % ORIGINAL K, no k⁻!
        f_orig = sd(s).f_orig;
        b_loc  = sd(s).if_loc(:);
        b_if   = sd(s).if_idx(:);
        if isempty(b_loc); continue; end

        % Internal DOF = all except (interface + Dirichlet)
        if sd(s).has_dirichlet
            all_b = unique([b_loc; sd(s).dir_loc]);
        else
            all_b = b_loc;
        end
        i_loc = setdiff((1:nloc)', all_b);

        % Standard Schur complement of K_orig at interface DOF
        if ~isempty(i_loc)
            Kii    = K_orig(i_loc, i_loc);
            Kbi    = K_orig(b_loc, i_loc);
            Kib    = K_orig(i_loc, b_loc);
            Kbb    = K_orig(b_loc, b_loc);
            S_E    = Kbb - Kbi * (Kii \ Kib);
            f_cond = f_orig(b_loc) - Kbi * (Kii \ f_orig(i_loc));
        else
            S_E    = K_orig(b_loc, b_loc);
            f_cond = f_orig(b_loc);
        end

        % Assemble: always + sign for both L_macro and b_ext
        for ii = 1:length(b_if)
            ri = b_if(ii);
            b_ext(ri) = b_ext(ri) + f_cond(ii);
            for jj = 1:length(b_if)
                cj = b_if(jj);
                L_macro(ri,cj) = L_macro(ri,cj) + S_E(ii,jj);
            end
        end
    end
end


function u_loc = solve_sub_dirichlet(K_loc, f_loc, sd_s)
    nloc  = sd_s.nloc;
    u_loc = zeros(nloc, 1);
    if sd_s.has_dirichlet
        free = setdiff((1:nloc)', sd_s.dir_loc);
        u_loc(free) = K_loc(free,free) \ f_loc(free);
    else
        u_loc = K_loc \ f_loc;
    end
end


function [u_sub, lam_sub, u_global] = recover_solution(sd, subs, W_hat, F, N_sub, Ne_sub, Nn)
    u_global = zeros(Nn,1);
    u_sub    = cell(N_sub,1);
    lam_sub  = cell(N_sub,1);
    for s = 1:N_sub
        nloc  = sd(s).nloc;
        K_loc = subs(s).K_sub;
        f_loc = F(sd(s).n_start:sd(s).n_end);
        bc_loc=[]; bc_val=[];
        if sd(s).has_dirichlet
            bc_loc(end+1)=sd(s).dir_loc; bc_val(end+1)=0.0;
        end
        for j=1:numel(sd(s).if_idx)
            bc_loc(end+1)=sd(s).if_loc(j); bc_val(end+1)=W_hat(sd(s).if_idx(j));
        end
        bc_loc=bc_loc(:); bc_val=bc_val(:);
        free=setdiff((1:nloc)',bc_loc);
        u_loc=zeros(nloc,1); u_loc(bc_loc)=bc_val;
        if ~isempty(free)
            u_loc(free)=K_loc(free,free)\(f_loc(free)-K_loc(free,bc_loc)*bc_val);
        end
        u_sub{s}=u_loc; lam_sub{s}=K_loc*u_loc-f_loc;
        u_global(sd(s).n_start:sd(s).n_end)=u_loc;
    end
end


function sd = preprocess_sd(subs, F, N_sub, Ne_sub)
    for s = 1:N_sub
        n_start=(s-1)*Ne_sub+1; n_end=s*Ne_sub+1; nloc=Ne_sub+1;
        if_idx=[]; if_sgn=[]; if_loc=[];
        if s>1,     if_idx(end+1)=s-1; if_sgn(end+1)=-1; if_loc(end+1)=1;    end
        if s<N_sub, if_idx(end+1)=s;   if_sgn(end+1)=+1; if_loc(end+1)=nloc; end
        sd(s).n_start=n_start; sd(s).n_end=n_end; sd(s).nloc=nloc;
        sd(s).K_sub=subs(s).K_sub; sd(s).f_orig=F(n_start:n_end);
        sd(s).if_idx=if_idx(:); sd(s).if_sgn=if_sgn(:); sd(s).if_loc=if_loc(:);
        sd(s).has_dirichlet=(s==1); sd(s).dir_loc=1;
    end
end