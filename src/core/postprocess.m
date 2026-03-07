function results = postprocess(u, mesh, params, K, F)
% POSTPROCESS - Calculate stress and strain and compare with analytical solutions.
%
% Input:
% u - Nodal displacements [Nn x 1]
% mesh - Mesh structure
% params - Parametric structure
% K - Global stiffness matrix (used to calculate reactions)
% F - Global load vector (used to calculate reactions)

% Output:
% results - Results structure, containing:
% .u - Nodal displacements
% .strain - Element strain
% .stress - Element stress
% .xmid - Element midpoint coordinates
% .u_exact - Analytical displacements
% .sigma_exact - Analytical stress
% .reaction - Support reactions
% .error - Errors
%

    %% Extract parameters
    E    = params.E;
    Fd   = params.Fd;
    h    = mesh.h;
    Ne   = mesh.Ne;
    x    = mesh.x;
    conn = mesh.conn;

    %% Calculate element strain and stress
    results.strain = zeros(Ne, 1);
    results.stress = zeros(Ne, 1);
    results.xmid   = zeros(Ne, 1);

    for e = 1:Ne
        nodes = conn(e,:);
        ue = u(nodes);
        results.strain(e) = (ue(2) - ue(1)) / h;
        results.stress(e) = E * results.strain(e);
        results.xmid(e)   = 0.5 * (x(nodes(1)) + x(nodes(2)));
    end

    %% Analytical solution (reusing analytical_solution)
    [results.u_exact, results.sigma_exact] = analytical_solution(x, params);

    %% Calculate support reactions
    R = K*u - F;
    results.reaction = R(1);

    %% Calculation error
    results.error.u_L1   = norm(u - results.u_exact, 1)   / norm(results.u_exact, 1);
    results.error.u_L2   = norm(u - results.u_exact, 2)   / norm(results.u_exact, 2);
    results.error.u_Linf = norm(u - results.u_exact, inf) / norm(results.u_exact, inf);

    %% Preserve displacement
    results.u = u;

end