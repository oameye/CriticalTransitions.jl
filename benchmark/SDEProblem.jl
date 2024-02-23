using CriticalTransitions, StaticArrays, ModelingToolkit, BenchmarkTools
using ModelingToolkit, OrdinaryDiffEq, StochasticDiffEq

σ = 0.03;
γ, α, ω₀, ω, λ = 0.01, 1.0, 1.0, 1.0, 0.6
@variables t u(t) v(t)
D = Differential(t)

eqs = [D(u) ~ -γ*u/2 - (3*α*(u^2 + v^2)/8 + (ω₀^2-ω^2)/2 + λ/4)*v/ω,
       D(v) ~ -γ*v/2 + (3*α*(u^2 + v^2)/8 + (ω₀^2-ω^2)/2 - λ/4)*u/ω]

noiseeqs = [0.1*u,
            0.1*v]

@named sys = ODESystem(eqs)
sys = structural_simplify(sys)
@named sdesys = SDESystem(sys, noiseeqs)

u0 = [u => 0.2,
      v => 0.3]
init = getindex.(u0, 2);

function KPO!(dx, x, p, t) # in-place
    u, v = x
    dx[1] = -γ*u/2 - (3*α*(u^2 + v^2)/8 + (ω₀^2-ω^2)/2 + λ/4)*v/ω
    dx[2] = -γ*v/2 + (3*α*(u^2 + v^2)/8 + (ω₀^2-ω^2)/2 - λ/4)*u/ω
end
function KPO(x, p, t) # out-of-place
    u, v = x
    du = -γ*u/2 - (3*α*(u^2 + v^2)/8 + (ω₀^2-ω^2)/2 + λ/4)*v/ω
    dv = -γ*v/2 + (3*α*(u^2 + v^2)/8 + (ω₀^2-ω^2)/2 - λ/4)*u/ω
    SA[du, dv]
end
init = getindex.(u0,2);
stochsys = StochSystem(KPO, init, zeros(2), σ, idfunc, nothing, I(2), "WhiteGauss")
stochsys! = StochSystem(KPO!, init, zeros(2), σ, idfunc, nothing, I(2), "WhiteGauss")
coupledODE = CoupledODEs(stochsys)
coupledODE! = CoupledODEs(stochsys!)

tmax = 1000.0; tspan = (0.0, tmax); dt = 5
prob_od = ODEProblem{false}(stochsys.f, SVector{2}(init), (0, tmax), CriticalTransitions.p(stochsys))
prob_od! = ODEProblem{true}(stochsys!.f, init, (0, tmax), CriticalTransitions.p(stochsys!))
prob_mt_jac = ODEProblem{false}(sys, u0, tspan, []; jac=true, u0_constructor=x -> SVector(x...))
prob_mt_jac! = ODEProblem{true}(sys, u0, tspan, []; jac=true)
prob_mt = ODEProblem{false}(sys, u0, tspan, []; jac=false, u0_constructor=x -> SVector(x...))
prob_mt! = ODEProblem{true}(sys, u0, tspan, []; jac=false)

@btime trajectory($coupledODE, $tmax, Δt = $dt); # 84.700 μs (2915 allocations: 61.86 KiB)
@btime trajectory($coupledODE!, $tmax, Δt = $dt); # 46.100 μs (2102 allocations: 49.14 KiB)
@btime solve($prob_od, Tsit5(), saveat=5); # 3.076 ms (108533 allocations: 2.14 MiB)
@btime solve($prob_od!, Tsit5(), saveat=5); # 1.359 ms (70165 allocations: 1.10 MiB)
@btime solve($prob_mt_jac, Tsit5(), saveat=5); # 102.900 μs (25 allocations: 16.91 KiB)
@btime solve($prob_mt_jac!, Tsit5(), saveat=5); # 121.600 μs (243 allocations: 32.34 KiB)
@btime solve($prob_mt, Tsit5(), saveat=5); # 103.500 μs (25 allocations: 16.91 KiB)
@btime solve($prob_mt!, Tsit5(), saveat=5); # 139.300 μs (253 allocations: 33.31 KiB)
