# Define custom types
Parameters = Union{Vector{Any}, Nothing};
CovMatrix = Union{Matrix, UniformScaling{Bool}, Diagonal{Bool, Vector{Bool}}};
State = Union{Vector, SVector}

# Define RateSystem
"""
    RateSystem(f, pf, dim, σ, g, pg, Σ, process)
Defines a stochastic dynamical system with a rate dependent shift in `CriticalTransitions.jl`. See [documentation](https://reykboerner.github.io/CriticalTransitions.jl/dev/man/stochsystem/).
"""
struct RateSystem
    f::Function
    pf::Parameters
    L::Function     # Vector of functions describing the change of each parameter
    T_trans::Float64
    T_shift::Float64
    dim::Int64
    σ::Float64
    g::Function
    pg::Parameters
    Σ::CovMatrix
    process::Any
end;

# Methods of RateSystem
RateSystem(f, pf, L, T_trans, T_shift, dim) = RateSystem(f, pf, L, T_trans, T_shift, dim, 1.0, idfunc, nothing, I(dim), "WhiteGauss")
RateSystem(f, pf, L, T_trans, T_shift, dim, σ) = RateSystem(f, pf, L, T_trans, T_shift, dim, σ, idfunc, nothing, I(dim), "WhiteGauss")
RateSystem(f, pf, L, T_trans, T_shift, dim, σ, Σ) = RateSystem(f, pf, L, T_trans, T_shift, dim, σ, idfunc, nothing, Σ, "WhiteGauss")

# functions for RateSystem

"""
    simulate(sys::RateSystem, init::State; kwargs...)
Simulates the RateSystem `sys` forward in time, starting at initial condition `init`.
## Keyword arguments
* `dt=0.01`: time step of integration
* `tmax=1e3`: total time of simulation
* `solver=EM()`: numerical solver. Defaults to Euler-Mayurama
* `callback=nothing`: callback condition
* `progress=true`: shows a progress bar during simulation
* `kwargs...`: keyword arguments for `solve(SDEProblem)`
For more info, see [`SDEProblem`](https://diffeq.sciml.ai/stable/types/sde_types/#SciMLBase.SDEProblem).
> Warning: This function has only been tested for the `EM()` solver and out-of-place `SDEFunction`s.
"""
function simulate(sys::RateSystem, init::State;
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    callback=nothing,
    progress=true,
    kwargs...)
    func(u,p,t) = fL(u,p,t,sys)

    prob = SDEProblem(func, σg(sys), init, (0, tmax), p(sys), noise=stochprocess(sys))
    solve(prob, solver; dt=dt, callback=callback, progress=progress, kwargs...)
end;

function fL(u,p,t,sys::RateSystem)
#    stationary = 0..t_trans ∪ t_trans+t_shift..Inf
#    du = t in stationary ? hcat(sys.f(u,p,t),zeros(length(sys.pf)) : hcat(sys.f(u,p,t),sys.L(u,p,t))

    nonstationary = t_trans..t_trans+t_shift
    du = t in nonstationary ? hcat(sys.f(u,p,t),sys.L(u,p,t)) : hcat(sys.f(u,p,t),zeros(length(sys.pf))
    return SVector{du}
end;


stochtorate(sys::StochSystem,L::Function,T_trans::Float64,T_shift::Float64)=RateSystem(sys.f,sys.pf,L,T_trans,T_shift,sys.dim,sys.σ,sys.g,sys.pg,sys.Σ,sys.process)