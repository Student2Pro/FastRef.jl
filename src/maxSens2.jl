#MaxSens2 multi tasks

@with_kw struct MaxSens2
    resolution::Float64 = 1.0
    tight::Bool         = false
end

# This is the main function
function solve(solver::MaxSens2, problem::Problem) #multi tasks
    result = true
    delta = solver.resolution
    lower, upper = low(problem.input), high(problem.input)
    n_hypers_per_dim = BigInt.(max.(ceil.(Int, (upper-lower) / delta), 1))

    # preallocate work arrays
    nt = nthreads()
    println(nt)
    local_lower = zeros(Float64, nt, lastindex(lower))
    local_upper = similar(local_lower)
    CI = zeros(Int64, nt, lastindex(lower))

    @threads for i in 1:prod(n_hypers_per_dim)
        n = i
        id = threadid()
        for j in firstindex(lower):lastindex(lower)
            n, CI[id, j] = fldmod1(n, n_hypers_per_dim[j])
        end
        @. local_lower[id,:] = lower + delta * (CI[id,:] - 1)
        @. local_upper[id,:] = min(local_lower[id,:] + delta, upper)
        hyper = Hyperrectangle(low = local_lower[id,:], high = local_upper[id,:])
        reach = forward_network(solver, problem.network, hyper)
        if !issubset(reach, problem.output)
            result = false
        end
    end
    if result
        return BasicResult(:holds)
    end
    return BasicResult(:violated)
end

# This function is called by forward_network
function forward_layer(solver::MaxSens2, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_node(solver::MaxSens2, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    β    = node.act(output)  # TODO expert suggestion for variable name. beta? β? O? x?
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    if solver.tight
        return ((βmax + βmin)/2, (βmax - βmin)/2)
    else
        return (β, max(abs(βmax - β), abs(βmin - β)))
    end
end
