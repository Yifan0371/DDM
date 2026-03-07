function [u_sub, lam_sub, info] = solve_latin_mono(subs, F, params, k_plus, k_minus)
% SOLVE_LATIN_MONO  Q2.10 - Monoscale LaTIn mixed DDM
%

%
%   Initialize: ŝ^{-1/2} = 0  (W_hat=0, F_hat=0 on all interfaces)
%
%   For n = 0, 1, 2, ...
%     ─────────────────────────────────────────────────
%     LINEAR STAGE  (global, find s_n in Ad from ŝ^{n-1/2})
%     ─────────────────────────────────────────────────
%     For each subdomain s, solve modified system with E⁻ Robin BC:
%
%       (K_s + k⁻·I_Γ) u_s = f_s + b_Γ
%
%     where at each interface boundary of s:
%       b_Γ += F̂_on_s + k⁻·Ŵ
%       F̂_on_s = +F_hat(i) if s is LEFT sub at interface i
%              = -F_hat(i) if s is RIGHT sub at interface i
%
%     Then recover interface force via E⁻ search direction:
%       F_s^n = F̂_on_s + k⁻·(Ŵ - u_s|_Γ)
%
%     ─────────────────────────────────────────────────
%     LOCAL STAGE  (local, find ŝ^{n+1/2} on Γ from s_n via E⁺)
%     ─────────────────────────────────────────────────
%     For each interface i, with s_n = (W_L, W_R, F_L, F_R):
%
%       Ŵ^{n+1/2} = (W_L + W_R)/2  +  (F_L - F_R)/(2k⁺)
%       F̂^{n+1/2} = (F_L + F_R)/2  +  k⁺/2·(W_L - W_R)
%
%     where:
%       W_L, F_L = displacement/force of LEFT  sub at interface i
%       W_R, F_R = displacement/force of RIGHT sub at interface i
%       F_L, F_R both have the same sign convention: positive = tension
%       (F_L is force ON left sub pointing outward; F_R is force ON right
%        sub pointing outward; at convergence F_L = F_R = tension)
%
%     Derivation (E⁺ for LEFT: F_L - F̂ = k⁺(W_L - Ŵ)
%                 E⁺ for RIGHT: F_R - F̂ = k⁺(W_R - Ŵ), same sign because
%                 RIGHT sub sees F̂ from its outward direction = +F̂ in our conv.)
%     Adding:    F_L+F_R - 2F̂ = k⁺(W_L+W_R-2Ŵ) ...(A)
%     This gives TWO unknowns. Need second equation from compatibility Ŵ_L=Ŵ_R=Ŵ
%     and equilibrium which says F̂_L_sub + F̂_R_sub = 0 at convergence.
%     With our convention (both forces = tension), (A) alone with
%     symmetry gives the two formulas above. ✓
%
%     CONVERGENCE:
%       η = ||ŝ^{n+1/2} - s_n|| / ||ŝ^{n+1/2}||
%
%
% Inputs:
%   subs    - struct array from generate_substructures (fields: K_sub)
%   F       - global force vector [Nn × 1]
%   params  - struct: N_sub, Ne_sub, Nn, tol, max_iter, L, E, S, Fd
%   k_plus  - E⁺ search direction [N/m]
%   k_minus - E⁻ search direction [N/m]
%
% Outputs:
%   u_sub   - {N_sub×1} cell of local displacement vectors
%   lam_sub - {N_sub×1} cell of local nodal reaction vectors
%   info    - struct: n_iter, eta_history, error_L2, u_global, u_exact, W_hat, F_hat


    N_sub    = params.N_sub;
    Ne_sub   = params.Ne_sub;
    Nn       = params.Nn;
    tol      = params.tol;
    max_iter = params.max_iter;
    n_if     = N_sub - 1;

    sd = preprocess_sd(subs, F, N_sub, Ne_sub);

    %  Initialize ŝ^{-1/2} = 0 
    W_hat    = zeros(n_if, 1);
    F_hat    = zeros(n_if, 1);
    eta_hist = zeros(max_iter, 1);
    n_iter   = 0;

    for iter = 1:max_iter

        %  LINEAR STAGE  (find s_n in Ad, E⁻ direction)
        W_curr = zeros(n_if, 2);   % W at each interface, left/right
        F_curr = zeros(n_if, 2);   % F at each interface (tension convention)

        for s = 1:N_sub
            nloc  = sd(s).nloc;
            K_loc = sd(s).K_sub;   % fresh copy each iter (modified below)
            f_loc = sd(s).f_orig;

            % Add E⁻ Robin contributions at each interface boundary
            for j = 1:numel(sd(s).if_idx)
                i_if = sd(s).if_idx(j);
                sgn  = sd(s).if_sgn(j);   % +1=LEFT, -1=RIGHT
                bloc = sd(s).if_loc(j);
                K_loc(bloc,bloc) = K_loc(bloc,bloc) + k_minus;
                % F̂ seen by sub s: +F_hat if LEFT, -F_hat if RIGHT
                f_loc(bloc) = f_loc(bloc) + sgn*F_hat(i_if) + k_minus*W_hat(i_if);
            end

            % Solve (Dirichlet u=0 at left wall for sub 1)
            u_loc = solve_sub_dirichlet(K_loc, f_loc, sd(s));

            % Recover interface quantities via E⁻:
            %   F_on_s = F̂_on_s + k⁻·(Ŵ - u_s|_Γ)
            for j = 1:numel(sd(s).if_idx)
                i_if = sd(s).if_idx(j);
                sgn  = sd(s).if_sgn(j);
                bloc = sd(s).if_loc(j);
                w_s  = u_loc(bloc);
                f_s  = sgn*F_hat(i_if) + k_minus*(W_hat(i_if) - w_s);
                % f_s = force ON sub s at this interface (tension > 0 for s)
                % Store with uniform tension convention:
                %   col 1: LEFT sub  (sgn=+1), store as-is
                %   col 2: RIGHT sub (sgn=-1), flip because outward dir differs
                if sgn == 1
                    W_curr(i_if,1) = w_s;
                    F_curr(i_if,1) = f_s;      % tension > 0  ✓
                else
                    W_curr(i_if,2) = w_s;
                    F_curr(i_if,2) = -f_s;     % flip: -f_s > 0 when tension
                end
            end
        end

        %  LOCAL STAGE  (find ŝ^{n+1/2} on Γ, E⁺ direction)
        %  LOCAL STAGE FORMULAS (derived from E+ condition):
        %
   
        W_hat_new = zeros(n_if,1);
        F_hat_new = zeros(n_if,1);
        for i = 1:n_if
            WL = W_curr(i,1);  WR = W_curr(i,2);
            FL = F_curr(i,1);  FR = F_curr(i,2);
            W_hat_new(i) = (WL + WR) / 2;                      % average displacement
            F_hat_new(i) = (FL + FR)/2 - k_plus/2*(WL - WR);  % MINUS sign (E+ projection)
        end

        %  CONVERGENCE  η = ||ŝ^{n+1/2} - s_n|| / ||ŝ^{n+1/2}||
        dW = W_hat_new - W_hat;
        dF = F_hat_new - F_hat;
        % Scale: combine displacement and force in energy norm
        scale_W = max(norm(W_hat_new), 1e-30);
        scale_F = max(norm(F_hat_new) / k_plus, 1e-30);  % F/k has units of displacement
        eta = sqrt( norm(dW)^2/scale_W^2 + (norm(dF)/k_plus)^2/scale_F^2 ) / sqrt(2);

        eta_hist(iter) = eta;
        n_iter = iter;
        W_hat = W_hat_new;
        F_hat = F_hat_new;
        if eta < tol; break; end
    end
    eta_hist = eta_hist(1:n_iter);

    %  Recover full displacement: Dirichlet solve with u|_Γ = W_hat 
    [u_sub, lam_sub, u_global] = recover_solution(sd, subs, W_hat, F, N_sub, Ne_sub, Nn);

    x_nodes = linspace(0, params.L, Nn).';
    u_exact = (params.Fd/(params.E*params.S)) * x_nodes;
    err_L2  = norm(u_global - u_exact) / norm(u_exact);

    info.n_iter      = n_iter;
    info.eta_history = eta_hist;
    info.error_L2    = err_L2;
    info.u_global    = u_global;
    info.u_exact     = u_exact;
    info.W_hat       = W_hat;
    info.F_hat       = F_hat;
    info.W_curr      = W_curr;
    info.F_curr      = F_curr;
end



function u_loc = solve_sub_dirichlet(K_loc, f_loc, sd_s)
% Solve K_loc * u = f_loc with u(dir_loc)=0 for sub 1
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
% Final solve: for each sub, fix interface DOF = W_hat and left wall = 0
    u_global = zeros(Nn,1);
    u_sub    = cell(N_sub,1);
    lam_sub  = cell(N_sub,1);
    for s = 1:N_sub
        nloc  = sd(s).nloc;
        K_loc = subs(s).K_sub;
        f_loc = F(sd(s).n_start:sd(s).n_end);
        bc_loc = [];  bc_val = [];
        if sd(s).has_dirichlet
            bc_loc(end+1) = sd(s).dir_loc;  bc_val(end+1) = 0.0;
        end
        for j = 1:numel(sd(s).if_idx)
            bc_loc(end+1) = sd(s).if_loc(j);
            bc_val(end+1) = W_hat(sd(s).if_idx(j));
        end
        bc_loc = bc_loc(:);  bc_val = bc_val(:);
        free  = setdiff((1:nloc)', bc_loc);
        u_loc = zeros(nloc,1);
        u_loc(bc_loc) = bc_val;
        if ~isempty(free)
            u_loc(free) = K_loc(free,free) \ (f_loc(free) - K_loc(free,bc_loc)*bc_val);
        end
        u_sub{s}   = u_loc;
        lam_sub{s} = K_loc*u_loc - f_loc;
        u_global(sd(s).n_start:sd(s).n_end) = u_loc;
    end
end


function sd = preprocess_sd(subs, F, N_sub, Ne_sub)
% Extract per-subdomain interface/boundary info
    for s = 1:N_sub
        n_start = (s-1)*Ne_sub + 1;
        n_end   =  s   *Ne_sub + 1;
        nloc    = Ne_sub + 1;
        if_idx = [];  if_sgn = [];  if_loc = [];
        if s > 1
            if_idx(end+1)=s-1; if_sgn(end+1)=-1; if_loc(end+1)=1;
        end
        if s < N_sub
            if_idx(end+1)=s;   if_sgn(end+1)=+1; if_loc(end+1)=nloc;
        end
        sd(s).n_start       = n_start;
        sd(s).n_end         = n_end;
        sd(s).nloc          = nloc;
        sd(s).K_sub         = subs(s).K_sub;
        sd(s).f_orig        = F(n_start:n_end);
        sd(s).if_idx        = if_idx(:);
        sd(s).if_sgn        = if_sgn(:);
        sd(s).if_loc        = if_loc(:);
        sd(s).has_dirichlet = (s == 1);
        sd(s).dir_loc       = 1;
    end
end