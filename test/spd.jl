using BenchmarkTools
using LinearAlgebra

function a(ys, xs, zs, ds)
    @. ys = zs + ds * xs
    return ys
end

function b(ys, xs, zs, ds)
    mul!(ys, zs, I, ds, xs)
    return ys
end

xs = randn(1000)
ys = randn(1000)
ds = rand()

@btime a($xs, $ys, $ds)


xs = randn(1000)
ys = randn(1000)
ds = rand()

@btime b($xs, $ys, $ds)
nothing