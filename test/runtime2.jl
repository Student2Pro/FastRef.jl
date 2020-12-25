using FastRef
using LazySets
import FastRef: forward_network, forward_affine_map, ishull

nnet = read_nnet("nnet/86442.nnet")

delta = 0.001

solver5 = SpeGuid(delta)
solver6 = HullTree(delta)
solver7 = DimTree(delta)
solver8 = FastTree(delta)

in_hyper = Hyperrectangle(fill(1.0, 8), fill(1.0, 8))
out_hyper = Hyperrectangle(fill(0.0, 2), fill(10.0, 2))
problem = Problem(nnet, in_hyper, out_hyper)

file = open("results/group2.txt", "a")
print(file, "Test Result of Group 2:\n\n")

#solver5

time5 = 0

solve(solver5, problem)
for i = 1:10
    timed_result =@timed solve(solver5, problem)
    print(file, "SpeGuid - test " * string(i) * " - Time: " * string(timed_result.time) * " s")
    print(file, " - Output: " * string(timed_result.value) * "\n")
    global time5 += timed_result.time
end

print(file, "Average time: " * string(time5/10) * " s\n\n")


#solver6

time6 = 0

solve(solver6, problem)
for i = 1:10
    timed_result =@timed solve(solver6, problem)
    print(file, "HullTree - test " * string(i) * " - Time: " * string(timed_result.time) * " s")
    print(file, " - Output: " * string(timed_result.value) * "\n")
    global time6 += timed_result.time
end

print(file, "Average time: " * string(time6/10) * " s\n\n")


#solver7

time7 = 0

solve(solver7, problem)
for i = 1:10
    timed_result =@timed solve(solver7, problem)
    print(file, "DimTree - test " * string(i) * " - Time: " * string(timed_result.time) * " s")
    print(file, " - Output: " * string(timed_result.value) * "\n")
    global time7 += timed_result.time
end

print(file, "Average time: " * string(time7/10) * " s\n\n")


#solver8

time8 = 0

solve(solver8, problem)
for i = 1:10
    timed_result =@timed solve(solver8, problem)
    print(file, "FastTree - test " * string(i) * " - Time: " * string(timed_result.time) * " s")
    print(file, " - Output: " * string(timed_result.value) * "\n")
    global time8 += timed_result.time
end

print(file, "Average time: " * string(time8/10) * " s\n\n")

close(file)
