function u = apply_boundary_conditions(K, F, mesh, params, method)
% APPLY_BOUNDARY_CONDITIONS - Apply boundary conditions and solve
%
% Input:
%   K      - Overall stiffness matrix [Nn x Nn]
%   F      - Overall load vector [Nn x 1]
%   mesh   - Mesh structure
%   params - Parameter structure
%   method - Boundary condition handling methods: 'elimination' | 'penalty' | 'lagrange'
%
% Output:
%   u - Nodal displacement vector [Nn x 1]
%


    Nn = mesh.Nn;
    
    if nargin < 5
        method = 'elimination';  
    end
    
    %% Define boundary conditions: fix u(1) = 0 at x = 0
    fixed_dof = 1;
    u_prescribed = 0;
    
    fprintf('Apply boundary conditions (method: %s)...\n', method);
    
    switch lower(method)
        case 'elimination'
            %% Method 1: Direct elimination method
            free_dofs = setdiff(1:Nn, fixed_dof);
            
            u = zeros(Nn, 1);
            uc = u_prescribed;
            
            % Block-based solution
            Kff = K(free_dofs, free_dofs);
            Kfc = K(free_dofs, fixed_dof);
            Ff  = F(free_dofs);
            
            uf = Kff \ (Ff - Kfc*uc);
            u(free_dofs) = uf;
            u(fixed_dof) = uc;
            
        case 'penalty'
            %% Method 2: Penalty Function Method
            k_max = max(max(abs(K)));
            alpha = k_max * 1e8;
            
            K_mod = K;
            F_mod = F;
            
            K_mod(fixed_dof, fixed_dof) = K_mod(fixed_dof, fixed_dof) + alpha;
            F_mod(fixed_dof) = F_mod(fixed_dof) + alpha * u_prescribed;
            
            u = K_mod \ F_mod;
            
        case 'lagrange'
            %% Method 3: Lagrange multiplier method
            C = zeros(1, Nn);
            C(fixed_dof) = 1;
            
            K_aug = [K,   C'; 
                     C,   0 ];
            F_aug = [F; u_prescribed];
            
            sol = K_aug \ F_aug;
            u = sol(1:Nn);
            
        otherwise
            error('Unknown boundary condition handling methods: %s', method);
    end
    
    fprintf('Solution complete (method: %s)\n', method);
    fprintf('  Maximum displacement: %.6e m\n\n', max(abs(u)));
    
end
