#SpeGuid2 count

@with_kw struct SpeGuid2
    tolerance::Float64 = 1.0
end

# This is the main function
function solve(solver::SpeGuid2, problem::Problem)
    input = problem.input
    stack = Vector{Hyperrectangle}(undef, 0)
    push!(stack, input)
    count = 0
    while !isempty(stack)
        interval = popfirst!(stack)
        reach = forward_network(solver, problem.network, interval)
        count += 1
        if issubset(reach, problem.output)
            continue
        else
            if get_largest_width(interval) > solver.tolerance
                sections = bisect(interval)
                for i in 1:2
                    push!(stack, sections[i])
                end
            else
                println(count)
                return BasicResult(:violated)
            end
        end
    end
    println(count)
    return BasicResult(:holds)
end

function forward_layer(solver::SpeGuid2, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_node(solver::SpeGuid2, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    return ((βmax + βmin)/2, (βmax - βmin)/2)
end
