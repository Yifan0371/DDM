function [lambda_b, u_global, feti_info] = solve_dual_feti(subs, F, params)
% SOLVE_DUAL_FETI - Question 2.8: FETI Method (Projected Conjugate Gradient + Dirichlet Preconditions)
% % Principle:
% The FETI method uses projected conjugate gradients (CG) to iteratively solve the dual interface problem:
% Sd * lambda = -bd
% Additional compatibility constraints:
% G^T * lambda = -e (force equilibrium condition)
% % The projection operator P = I - G*(G^T*G)^{-1}*G^T ensures that the iteration is within the constrained subspace.
% Dirichlet preconditioner: ~Sd^{-1} ≈ Sp (assembling the original Schur complement).

% Algorithm (Constrained/Projected PCG):
% 1. Initialization: lambda_0 satisfies G^T*lambda_0 = -e
% 2. r_0 = Sd*lambda_0 + bd
% 3. Check if ||P*r_0|| has converged
% 4. z_0 = M * r_0 (preconditioner)
% 5. w_0 = P * z_0 (projection)
% 6. Iterate CG

%
% All matrix-vector products Sd*v are computed in a distributed manner:
% Sd*v = A_ * Sd_diag * (A_^T * v)
%
% Input:
% subs - array of substructures
% F - global load vector [Nn x 1]
% params - parameter structure
%
% Output:
% lambda_b - GUI Lagrange multiplier [n_lambda x 1]
% u_global - Global shift vector [Nn x 1]
% feti_info - FETI solution information
%


N_sub = params.N_sub;
Nn    = params.Nn;
E = params.E; S = params.S; h = params.h;
k0 = E*S/h;
tol = params.tol;
max_iter = params.max_iter;

fprintf('====================================\n');
fprintf('FETI Method - Projection PCG(Q2.8)\n');
fprintf('====================================\n');

%  Compute dual operators
true_intf_nodes = [];
for s = 1:N_sub-1
    true_intf_nodes = [true_intf_nodes; s * params.Ne_sub + 1];
end
n_lambda = length(true_intf_nodes);

for s = 1:N_sub
    nloc = subs(s).nloc;
    K_sub = subs(s).K_sub;
    nodes_s = subs(s).node_ids;
    
    [~, loc_b_in] = ismember(true_intf_nodes, nodes_s);
    loc_b_idx = loc_b_in(loc_b_in > 0);
    nb_s = length(loc_b_idx);
    loc_i_idx = setdiff((1:nloc)', loc_b_idx);
    
    if s == 1
        loc_i_idx = setdiff(loc_i_idx, 1);
    end
    
    Kbb = K_sub(loc_b_idx, loc_b_idx);
    Kbi = K_sub(loc_b_idx, loc_i_idx);
    Kii = K_sub(loc_i_idx, loc_i_idx);
    Kib = K_sub(loc_i_idx, loc_b_idx);
    
    f_loc = F(nodes_s);
    fb = f_loc(loc_b_idx);
    fi = f_loc(loc_i_idx);
    
    if isempty(loc_i_idx)
        Sp_s = Kbb; bp_s = fb;
    else
        Sp_s = Kbb - Kbi * (Kii \ Kib);
        bp_s = fb - Kbi * (Kii \ fi);
    end
    
    tol_sv = 1e-10 * k0;
    if nb_s == 1
        if abs(Sp_s) < tol_sv
            Sd_s = 0; Rb_s = 1; bd_s = 0;
        else
            Sd_s = 1/Sp_s; Rb_s = zeros(1,0); bd_s = Sd_s * bp_s;
        end
    elseif nb_s > 1
        [~, Sig, V] = svd(Sp_s);
        sv = diag(Sig); n_zero = sum(sv < tol_sv);
        if n_zero == 0
            Sd_s = inv(Sp_s); Rb_s = zeros(nb_s, 0);
        else
            Sd_s = pinv(Sp_s, tol_sv);
            Rb_s = V(:, end-n_zero+1:end);
        end
        bd_s = Sd_s * bp_s;
    else
        Sd_s = []; Rb_s = []; bp_s = []; bd_s = [];
    end
    
    subs(s).loc_b_idx_dual = loc_b_idx;
    subs(s).loc_i_idx_dual = loc_i_idx;
    subs(s).nb_dual = nb_s;
    subs(s).Sp_dual = Sp_s;
    subs(s).Sd_dual = Sd_s;
    subs(s).Rb_dual = Rb_s;
    subs(s).bp_dual = bp_s;
    subs(s).bd_dual = bd_s;
    subs(s).Kii_dual = Kii;
    subs(s).Kib_dual = Kib;
    subs(s).Kbi_dual = Kbi;
    subs(s).fi_dual = fi;
end

% Dual assembly operator
nb_total = 0;
nb_offsets = zeros(N_sub+1, 1);
for s = 1:N_sub
    nb_offsets(s) = nb_total;
    nb_total = nb_total + subs(s).nb_dual;
end
nb_offsets(N_sub+1) = nb_total;

A_dual = zeros(n_lambda, nb_total);
for s = 1:N_sub
    nb_s = subs(s).nb_dual;
    nodes_s = subs(s).node_ids;
    loc_b_global = nodes_s(subs(s).loc_b_idx_dual);
    for j = 1:nb_s
        gnode = loc_b_global(j);
        intf_idx = find(true_intf_nodes == gnode);
        if ~isempty(intf_idx)
            col = nb_offsets(s) + j;
            if s <= intf_idx
                A_dual(intf_idx, col) = 1;
            else
                A_dual(intf_idx, col) = -1;
            end
        end
    end
end

% Diagonal of the splicing blocks
Sd_diag = zeros(nb_total);
Sp_diag = zeros(nb_total);
bp_vec = zeros(nb_total, 1);
bd_vec = zeros(nb_total, 1);
n_rigid_total = 0;
Rb_cells = cell(N_sub, 1);

for s = 1:N_sub
    idx = nb_offsets(s)+1 : nb_offsets(s+1);
    Sd_diag(idx, idx) = subs(s).Sd_dual;
    Sp_diag(idx, idx) = subs(s).Sp_dual;
    bp_vec(idx) = subs(s).bp_dual;
    bd_vec(idx) = subs(s).bd_dual;
    Rb_cells{s} = subs(s).Rb_dual;
    n_rigid_total = n_rigid_total + size(subs(s).Rb_dual, 2);
end
Rb_diag = blkdiag(Rb_cells{:});

% Assembly
Sd = A_dual * Sd_diag * A_dual';
bd = A_dual * bd_vec;
G  = A_dual * Rb_diag;
e_vec = Rb_diag' * bp_vec;

fprintf('n_lambda = %d, n_rigid = %d\n', n_lambda, n_rigid_total);

%  Dirichlet Preconditioner
%  ~Sd^{-1} = A_ * Sp_diag * A_^T
M_Dirichlet = A_dual * Sp_diag * A_dual';

%  Projection operator P = I - G*(G^T*G)^{-1}*G^T
if n_rigid_total > 0
    GtG_inv = inv(G' * G);
    project = @(v) v - G * (GtG_inv * (G' * v));
else
    project = @(v) v;
end

%  Distributed matrix-vector product
%  Sd*v = A_ * Sd_diag * (A_^T * v)
Sd_matvec = @(v) A_dual * (Sd_diag * (A_dual' * v));

%  Initialize lambda_0 such that G^T*lambda_0 = -e
%  lambda_0 = -G * (G^T*G)^{-1} * e
if n_rigid_total > 0
    lambda_0 = -G * (GtG_inv * e_vec);
else
    lambda_0 = zeros(n_lambda, 1);
end

%  Projected preconditional conjugate gradient (Projected PCG)
lambda = lambda_0;
r = Sd_matvec(lambda) + bd;

% --- Check if the initial residuals have converged. ---
r_proj = project(r);
res0_norm = norm(r_proj);

res_history = [];
converged = false;
n_iter = 0;

if res0_norm < tol * max(1, norm(bd))
    
    % Special case: Initialization is the exact solution
    converged = true;
    res_history = [res0_norm];
    fprintf('The initial lambda is already an exact solution (||P*r_0|| = %.2e), so no iteration is needed.\n', res0_norm);
else
    % Normal PCG iteration
    z = M_Dirichlet * r;
    w = project(z);
    p = w;
    
    for iter = 1:max_iter
        % Distributed matrix-vector multiplication
        q = Sd_matvec(p);
        
        rw = r' * w;
        pq = p' * q;
        if abs(pq) < 1e-30 * (norm(p) * norm(q) + 1e-30)
            fprintf('  The iteration %d: p^T*q ≈ 0, the CG direction is exhausted, and the process terminates.\n', iter);
            break;
        end
        alpha_cg = rw / pq;
        
        lambda = lambda + alpha_cg * p;
        r = r - alpha_cg * q;
        
        % Convergence check
        r_proj = project(r);
        res_norm = norm(r_proj);
        rel_res = res_norm / res0_norm;
        res_history = [res_history; rel_res];
        n_iter = iter;
        
        fprintf('  iteration %3d: ||Pr||/||Pr_0|| = %.6e\n', iter, rel_res);
        
        if rel_res < tol
            converged = true;
            fprintf('Convergence! Iterated %d times, relative residual = %.6e\n', iter, rel_res);
            break;
        end
        
        % Preconditions + Projection
        z_new = M_Dirichlet * r;
        w_new = project(z_new);
        
        beta_cg = (r' * w_new) / rw;
        p = w_new + beta_cg * p;
        w = w_new;
    end
    
    if ~converged
        fprintf('WARNING: Maximum number of iterations reached %d, not converged. (rel_res = %.2e)\n', ...
            max_iter, res_history(end));
    end
end

%  Solve for the modal amplitude alpha of the rigid body.
%
if n_rigid_total > 0
    residual_for_alpha = -bd - Sd_matvec(lambda);
    alpha_global = GtG_inv * (G' * residual_for_alpha);
else
    alpha_global = [];
end

fprintf('lambda = [%s]\n', num2str(lambda'));
if ~isempty(alpha_global)
    fprintf('alpha  = [%s]\n', num2str(alpha_global'));
end

%  Restore displacement
lambda_b = lambda;
lambda_b_diag = A_dual' * lambda_b;
u_global = zeros(Nn, 1);

alpha_off = 0;
for s = 1:N_sub
    nb_s = subs(s).nb_dual;
    nr_s = size(subs(s).Rb_dual, 2);
    nodes_s = subs(s).node_ids;
    
    idx = nb_offsets(s)+1 : nb_offsets(s+1);
    lam_s = lambda_b_diag(idx);
    
    if nr_s > 0
        alp_s = alpha_global(alpha_off+1 : alpha_off+nr_s);
        alpha_off = alpha_off + nr_s;
    else
        alp_s = [];
    end
    
    ub_s = subs(s).Sd_dual * (subs(s).bp_dual + lam_s);
    if nr_s > 0
        ub_s = ub_s + subs(s).Rb_dual * alp_s;
    end
    
    if ~isempty(subs(s).loc_i_idx_dual)
        ui_s = subs(s).Kii_dual \ (subs(s).fi_dual - subs(s).Kib_dual * ub_s);
    else
        ui_s = [];
    end
    
    loc_b_global = nodes_s(subs(s).loc_b_idx_dual);
    for j = 1:nb_s
        u_global(loc_b_global(j)) = ub_s(j);
    end
    loc_i_global = nodes_s(subs(s).loc_i_idx_dual);
    for j = 1:length(ui_s)
        u_global(loc_i_global(j)) = ui_s(j);
    end
end

u_global(1) = 0;

%  verify
mesh_x = linspace(0, params.L, Nn)';
u_exact = (params.Fd / (params.E * params.S)) * mesh_x;
error_L2 = norm(u_global - u_exact) / norm(u_exact);

fprintf('\n====================================\n');
fprintf('Verification results (Q2.8 FETI)\n');
fprintf('====================================\n');
fprintf('u(L) Numerical solution  = %.12e m\n', u_global(end));
fprintf('u(L) Analytical solution  = %.12e m\n', u_exact(end));
fprintf('L2 relative error  = %.6e\n', error_L2);
fprintf('PCG Number of iterations = %d\n', n_iter);
if converged && n_iter == 0
    fprintf('For uniform stretching problems, an exact solution can be obtained from initialization, without the need for iteration.\n');
end
fprintf('====================================\n\n');

%% Output info


if n_iter == 0
    % PCG iteration was not actually performed -> residual sequence was not recorded.
    res_history = [];
else
    % Normally, res_history is simply the length = n_iter
    res_history = res_history(1:n_iter);
end

feti_info.n_iter      = n_iter;
feti_info.residuals   = res_history;
feti_info.alpha       = alpha_global;
feti_info.error_L2    = error_L2;
feti_info.u_exact     = u_exact;
feti_info.Sd          = Sd;
feti_info.G           = G;
feti_info.M_Dirichlet = M_Dirichlet;
feti_info.converged   = converged;
feti_info.subs_updated = subs;
feti_info.nb_offsets  = nb_offsets;
feti_info.A_dual      = A_dual;
feti_info.true_intf_nodes = true_intf_nodes;

end
