function [u_exact, sigma_exact] = analytical_solution(x, params)
% ANALYTICAL_SOLUTION - Calculate the analytical solution for a 1D tension bar.
%
% Input:
%   x      - Position coordinates [Nn x 1]
%   params - Parameter structure
%
% Output:
%   u_exact     - Analytical displacement [Nn x 1]
%   sigma_exact - Analytical stress (constant)
%
% Theory:
%   For a 1D uniform rod under tension:
%   u(x) = (F / (E*S)) * x
%   σ(x) = F / S  (constant)
%


    E  = params.E;
    S  = params.S;
    Fd = params.Fd;
    
    %% Analytical displacement
    u_exact = (Fd / (E*S)) * x;
    
    %% Analytical stress (constant)
    sigma_exact = Fd / S;
    
end
