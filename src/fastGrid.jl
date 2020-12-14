@with_kw struct FastGrid
    resolution::Float64 = 1.0
    #tight::Bool         = false
end

# This is the main function
function solve(solver::FastGrid, problem::Problem) #original
    result = true
    delta = solver.resolution

    center = problem.input.center
    radius = problem.input.radius[1]

    (W, b) = (problem.network.layers[1].weights, problem.network.layers[1].bias)

    input = forward_affine_map(solver, W, b, problem.input)

    lower, upper = low(input), high(input)
    n_hypers_per_dim = BigInt.(max.(ceil.(Int, (upper-lower) / delta), 1))

    # preallocate work arrays
    local_lower, local_upper, CI = similar(lower), similar(lower), similar(lower)
    for i in 1:prod(n_hypers_per_dim)
        n = i
        for j in firstindex(CI):lastindex(CI)
            n, CI[j] = fldmod1(n, n_hypers_per_dim[j])
        end
        @. local_lower = lower + delta * (CI - 1)
        @. local_upper = min(local_lower + delta, upper)
        hyper = Hyperrectangle(low = local_lower, high = local_upper)

        k_1 = size(W, 1) #length(lower)
        I = zeros(k_1, k_1) #Identity matrix
        IN = zeros(k_1, k_1) #negative Identity matrix
        for k = 1:k_1
            I[k, k] = 1.0
            IN[k, k] = -1.0
        end
        #Hi
        C_1 = vcat(I, IN)
        d_1 = vcat(local_upper, -local_lower)
        #Pi
        C = C_1 * W #size(2k_1, k_0)
        d = d_1 - C_1 * b #size(2k_1)

        inter = true #if the intersection of Pi and the input set is empty
        for l = 1:length(d)
            sum = 0.0
            for m = 1:size(C, 2)
                if C[l,m] > 0
                    sum += low(problem.input, m)
                elseif C[l,m] < 0
                    sum += high(problem.input, m)
                end
            end
            if sum > d[l]
                inter = false
                break
            end
        end

        if inter
            hull = false #if Hi is a hull of Z, or if Pi and center of the input set are close
            for l = 1:length(d)
                if distance(center, C[l,:], d[l]) > r
                    hull = true
                    break
                end
            end

            if hull
                reach = forward_network(solver, problem.network, hyper)
                if !issubset(reach, problem.output)
                    result = false
                end
            end
        end
    end
    if result
        return BasicResult(:holds)
    end
    return BasicResult(:violated)
end

function forward_network(solver::FastGrid, nnet::Network, input::Hyperrectangle)
    layers = nnet.layers
    act = layers[1].activation
    reach = Hyperrectangle(low = act.(low(input)), high = act.(high(input)))

    for i in 2:length(layers)
        reach = forward_layer(solver, layer[i], reach)
    end
    return reach
end

# This function is called by forward_network
function forward_layer(solver::FastGrid, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_affine_map(solver::FastGrid, W::Matrix, b::Vector, input::Hyperrectangle)
    (W, b) = (L.weights, L.bias)
    center = W * input.center + b
    radius = abs.(W) * input.radius
    return Hyperrectangle(center, radius)
end

function forward_node(solver::FastGrid, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    β    = node.act(output)  # TODO expert suggestion for variable name. beta? β? O? x?
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    return ((βmax + βmin)/2, (βmax - βmin)/2)
    #if solver.tight
        #return ((βmax + βmin)/2, (βmax - βmin)/2)
    #else
        #return (β, max(abs(βmax - β), abs(βmin - β)))
    #end
end

function distance(point::Vector, c::Vector, d::real)
    if length(point) == length(c)
        return abs(point .* c + d)/sqrt(sum(abs2.(c)))
    else
        error("Dimesion dismathch for point and constraint")
    end
end
