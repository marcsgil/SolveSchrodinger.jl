module SolveSchrodinger

export solveSchrodinger

using SparseArrays,Arpack,Kronecker,LinearAlgebra

D(N) = SymTridiagonal(-2ones(N),ones(N-1))

"""
    solveSchrodinger(xs,V;mass=1,par=nothing)

Return `(Es,ψs)`, which are the eigenvalues and eigenvectors of the discretized Schrödinger's equation 

``-ψ''(x)/2mass + V(x,par)ψ(x) = Eψ`` 

over the grid `xs`.
"""
function solveSchrodinger(xs,V;mass=1,par=nothing)
    H = -D(length(xs))/(2*mass*(xs[2]-xs[1])^2) + diagm( map(x->V(x,par), xs) )
    eigen(H)
end

"""
    solveSchrodinger(xs,ys,V;xmass=1,ymass=1,nev=50,par=nothing)

Return `(Es,ψs)`, which are the eigenvalues and eigenvectors of the discretized Schrödinger's equation 

``-∂₁² ψ(x)/2xmass -∂₂² ψ(x)/2ymass + V(x,par)ψ(x) = Eψ`` 

over the grid defined as the tensor product of `xs` and `ys`.
"""
function solveSchrodinger(xs,ys,V;xmass=1,ymass=1,nev=50,par=nothing)
    H = sparse( -( D(length(xs))/(2*xmass*(xs[2]-xs[1])^2) ⊕ D(length(ys))/(2*ymass*(ys[2]-ys[1])^2) )
    + diagm(vec(map(r->V([r[1],r[2]],par), Iterators.product(xs,ys)))) )
    eigs(H,nev=nev,which=:SR)
end

end 