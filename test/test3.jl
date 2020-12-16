using FastRef
using LazySets
import FastRef: forward_network, forward_affine_map, ishull

nnet = read_nnet("nnet/86442.nnet")
solver = FastGrid(0.3)

in_hyper = Hyperrectangle(fill(1.0, 8), fill(1.0, 8))
out_hyper = Hyperrectangle(fill(0.0, 2), fill(1.0, 2))
problem = Problem(nnet, in_hyper, out_hyper)
timed_result =@timed solve(solver, problem)
print("FastGrid - test")
print(" - Time: " * string(timed_result[2]) * " s")
print(" - Output: ")
print(timed_result[1].status)
#print("\n")
