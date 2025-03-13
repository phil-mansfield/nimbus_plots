import symlib
import numpy as np
import lib

sim_dir = symlib.get_host_directory(lib.base_dir, "SymphonyLMC", 3)
part = symlib.Particles(sim_dir)

rs, hist = symlib.read_rockstar(sim_dir)
"""
print(rs["m"][165,190:210])
print(np.sqrt(np.sum((rs["x"][165,190:210] - rs["x"][0,190:210])**2, axis=1))/rs["rvir"][0,190:210])
print(rs["m"][166,190:210])
print(np.sqrt(np.sum((rs["x"][166,190:210] - rs["x"][0,190:210])**2, axis=1))/rs["rvir"][0,190:210])
"""
print(rs["x"][165,193])

"""
sf, hist = symlib.read_rockstar(sim_dir)
print(sf["m"][165,190:210])
print(hist["merger_snap"][165])
print(hist["first_infall_snap"][165])
print(hist["mpeak"][165])
print(hist["mpeak_pre"][165])

#for i in [164, 165, 166]:
for i in [165]:
    p = part.read(193, mode="all", halo=i)
    print(len(p))
    print(p["smooth"])
    print(p["ok"])
    print(p["snap"])
    print(p["x"][:,0])
"""   

print(part.part_info.tags.id[165])
print(part.part_info.tags.snap[165])
print(part.part_info.tags.flag[165])

p = part.read(192, mode="all", halo=166)
p = part.read(192, mode="smooth", halo=166)
print()
print()
p = part.read(193, mode="all", halo=165)
print(p["x"])
p = part.read(192, mode="all", halo=165)
print()
print()
p = part.read(193, mode="smooth", halo=165)
p = part.read(192, mode="smooth", halo=165)

