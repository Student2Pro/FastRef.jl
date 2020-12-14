#SpeGuid1 multi tasks

@with_kw struct SpeGuid1
    tolerance::Float64 = 1.0
end

# This is the main function
function solve(solver::SpeGuid1, problem::Problem) #multi tasks
    result = true
    input = problem.input
    channel = Channel{Hyperrectangle}(512) #64
    put!(channel, input)
    remain = 1 #remain Hyperrectangle in channel
    count = 0
    while remain != 0
        #println("while loop begin")
        if isopen(channel)
            interval = take!(channel)
        else
            println(isopen(channel))
            break
        end
        @async begin
            #println("Task start")
            if isopen(channel) #if channel is closed, stop Task
                reach = forward_network(solver, problem.network, interval)
                count += 1
            end
            if isopen(channel)
                if issubset(reach, problem.output)
                    remain -= 1
                else
                    if get_largest_width(interval) > solver.tolerance
                        sections = bisect(interval)
                        for i in 1:2
                            put!(channel, sections[i])
                        end
                        remain += 1
                    else
                        println(interval)
                        close(channel)
                        result = false
                    end
                end
            end
            #println("Task end")
        end
        #println("while loop end")
    end
    println("count: " * string(count) * " remain: " * string(remain))
    if result
        return BasicResult(:holds)
    end
    return BasicResult(:violated)
end

function forward_layer(solver::SpeGuid1, L::Layer, input::Hyperrectangle)
    (W, b, act) = (L.weights, L.bias, L.activation)
    center = zeros(size(W, 1))
    gamma  = zeros(size(W, 1))
    for j in 1:size(W, 1)
        node = Node(W[j,:], b[j], act)
        center[j], gamma[j] = forward_node(solver, node, input)
    end
    return Hyperrectangle(center, gamma)
end

function forward_node(solver::SpeGuid1, node::Node, input::Hyperrectangle)
    output    = node.w' * input.center + node.b
    deviation = sum(abs.(node.w) .* input.radius)
    βmax = node.act(output + deviation)
    βmin = node.act(output - deviation)
    return ((βmax + βmin)/2, (βmax - βmin)/2)
end
