using Pkg
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using Profile
using Dates
# 引入你的包
using PolymermcCPCMEM 

# ==============================================================================
# 1. 构造测试环境 (与之前一致)
# ==============================================================================
N = 100
counts = (50, 50)
conns = (2, 6) # 混合连接，测试你的 batch kernel
params = SimulationParameters(
    N = N,
    tf_counts = counts,
    tf_connectivities = conns,
    Steps_FINAL = 1_00_000, # 足够长的步数以收集样本
    num_samples = 10,
    box_size = 20.0,
    fbead_contact = fill(-1, sum(counts), 6),
    alpha = zeros(N, N),
    alpha_0 = zeros(N, N),
    Pij_mediated_matrix = zeros(N, N)
)
temp_dir = mktempdir()

# ==============================================================================
# 2. 执行 Profiling
# ==============================================================================

println("1. Compiling (Warmup)...")
# 预热：让 Julia 编译所有函数，避免把编译时间算进性能分析
PolymermcCPCMEM.run_simulation_core(params, temp_dir)

println("2. Profiling...")
Profile.clear() # 清空旧数据
@profile PolymermcCPCMEM.run_simulation_core(params, temp_dir)

# ==============================================================================
# 3. 输出分析报告
# ==============================================================================

println("\n" ^ 3)
println("=" ^ 60)
println("TOP BOTTLENECKS (FLAT VIEW)")
println("=" ^ 60)

# format=:flat : 不显示调用层级，只看“哪个函数占用了最多的CPU时间”
# sortedby=:count : 按耗时排序
# mincount=100 : 过滤掉那些耗时极短的噪音
Profile.print(format=:flat, sortedby=:count, mincount=500, combine=true)

println("\n" ^ 3)
println("=" ^ 60)
println("DETAILED PHYSICS KERNEL ANALYSIS")
println("=" ^ 60)

# 1. mincount=10: 只要耗时超过 1% (10/766) 就显示，展示细节
# 2. maxdepth=30: 增加显示深度
# 3. noisefloor=0.0: 不隐藏低占比分支
Profile.print(format=:tree, mincount=10, maxdepth=30, noisefloor=0.0)