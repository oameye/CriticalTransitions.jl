using DifferentialEquations

include("StochSystem.jl")
include("noise.jl")

function simulate(sys::StochSystem, init::Vector;
    dt=0.01,
    tmax=1e3,
    solver=EM(),
    callback=nothing)

    if sys.process == "WhiteGauss"
        prob = SDEProblem(sys.f, σg(sys), init, (0, tmax), p(sys), noise=gauss(sys))
        sol = solve(prob, solver; dt=dt, callback=callback)
        sol
    else
        ArgumentError("ERROR: Noise process not yet implemented.")
    end
end;

function relax(sys::StochSystem, init::Vector;
    dt=0.01,
    tmax=1e3,
    solver=Euler(),
    callback=nothing)
    
    prob = ODEProblem(sys.f, init, (0, tmax), p(sys))
    sol = solve(prob, solver; dt=dt, callback=callback)

    sol
end;

function transition_(sys::StochSystem, x_i::Vector, x_f::Vector;
    rad_i=1.0,
    rad_f=1.0,
    dt=0.01,
    tmax=1e2,
    solver=EM(),
    cut_start=true)

    condition(u,t,integrator) = norm(u - x_f) < rad_f
    affect!(integrator) = terminate!(integrator)
    cb_ball = DiscreteCallback(condition, affect!)

    sim = simulate(sys, x_i, dt=dt, tmax=tmax, solver=solver, callback=cb_ball)

    if sim.t[end] == tmax
        error("WARNING: Simulation stopped before a transition occurred.")        
    else
        simt = sim.t
        if cut_start
            idx = size(sim)[2]
            dist = norm(sim[:,idx] - x_i)
            while dist > rad_i
                idx -= 1
                dist = norm(sim[:,idx] - x_i)
            end
            sim = sim[:,idx:end]
            simt = simt[idx:end]
        end
        sim, simt
    end
end;