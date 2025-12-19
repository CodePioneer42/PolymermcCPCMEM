#!/usr/bin/env julia

#SBATCH --job-name Stage2_Opt
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --exclude cpu1-1
#SBATCH --partition i64m512u
#SBATCH --output %j-out.txt
#SBATCH --error %j-err.txt
using Pkg

# --- 关键步骤：激活当前包环境 ---
# 这会让脚本识别到 src/ 下的 PolymerMC 模块
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()

using Distributed
using ProgressMeter
using LinearAlgebra
using Statistics
using Parameters
using DelimitedFiles
using CSV
using DataFrames
using Printf          # <--- 添加这行，或者将 using Printf 移到这里


using PolymermcCPCMEM


println("--- Starting Stage 2: Specific Interaction Optimization (alpha & gamma) ---")

# --- Parallel Setup ---
@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PolymermcCPCMEM
end

@everywhere using Printf
@everywhere using Statistics

# --- Load Code on All Workers ---
@everywhere begin

    using DelimitedFiles  # 推荐也将这个放在这里

    using Parameters
    using LinearAlgebra
    using Random
    using Statistics
    using StaticArrays

    function run_simulation_core(params::SimulationParameters, snapshot_base_dir)
        
        worker_id = myid()
        pdb_filename = joinpath(snapshot_base_dir, "worker_$(worker_id).pdb")
        energy_filename = joinpath(snapshot_base_dir, "energy_worker_$(worker_id).txt")

        # --- [新增] 初始化一个缓冲区来存储输出行 ---
        output_buffer = String[]
        # 可选：为了效率，预分配大小
        sizehint!(output_buffer, params.num_samples)

        # --- [修改] 使用新的参数来计算总TF数 ---
        total_N_B = sum(params.tf_counts)

        # 初始化
        output_chains = Vector{Vector{Particle3D}}()
        tem_beads = Vector{Vector{Particle3D}}()
        chain = initialize_random_chain(params.N, params.r0_bond, params.min_distance, 1000)

        free_beads = if total_N_B > 0
            initialize_free_beads(total_N_B, chain, params.min_distance, 20.0)
        else
            Particle3D[] 
        end
        
        params.Pij_mediated_matrix = fill(0.0, params.N, params.N)



        if total_N_B > 0
            # --- [修改] 根据最大连接数动态设置 fbead_contact 矩阵大小 ---
            max_conn = maximum(params.tf_connectivities)
            params.fbead_contact = fill(-1, total_N_B, max_conn)
        end

        current_energy = compute_total_energy!(chain, free_beads, params)
        move_functions = (mcdiff!, mcdiff_free_bead!, mcpivot!, mcdoublepivot!, freebead_snake!, mcloop_extrusion!)

        # --- [新增] 定义移动方式的权重 ---
        move_weights = [100, 100, 20, 20, 5, 1]
        
        # --- [新增] 预计算累积权重以进行高效的带权抽样 ---
        # cumsum 计算累积和: [100, 200, 220, 240, 245, 246]
        cumulative_weights = cumsum(move_weights)
        total_weight = cumulative_weights[end] # 总权重为 246

        # 主循环
        T = params.MTf
        equilibration_steps = round(Int, params.Steps_FINAL * 0.5)
        sampling_steps = params.Steps_FINAL - equilibration_steps
        sampling_interval = round(Int, max(1, sampling_steps / params.num_samples))

        for step in 1:params.Steps_FINAL
            # --- [修改] 根据权重选择移动方式, 为了提高效率不使用均等的概率 ---
            rand_val = rand(1:total_weight)
            move_idx = searchsortedfirst(cumulative_weights, rand_val)

            _, current_energy = move_functions[move_idx](chain, free_beads, params, current_energy, T)
            
            if step > equilibration_steps && (step - equilibration_steps) % sampling_interval == 0
                push!(output_chains, deepcopy(chain))
                push!(tem_beads, deepcopy(free_beads))
                
                # --- [修改 2/3] 计算能量和平均配位数，并写入文件 ---
                energy_out = [compute_total_energy!(chain, free_beads, params),
                                compute_bond_energy(chain, params), 
                                compute_bond_angle_energy(chain, params), 
                                compute_chain_nonbond_energy(chain, params), 
                                compute_wall_interaction(chain,params), 
                                compute_free_bead_energy(free_beads, params), 
                                compute_specific_interaction_energy(chain, free_beads, params), # 特异性相互作用能 (alpha 项)
                                ideal_chromosome_Pij(chain, params)
                                ]
                
                # 计算平均配位数
                mean_coord_num = calculate_mean_coordination_number(chain, params.coord_num_threshold)
                
                # 准备要写入文件的行
                output_step = step - equilibration_steps
                energy_str = join([@sprintf("%.4f", e) for e in energy_out], " ")
                line_to_write = @sprintf("%d %d %s %.4f\n", worker_id, output_step, energy_str, mean_coord_num)
                push!(output_buffer, line_to_write)
            end
        end
        
        open(energy_filename, "w") do f
            for line in output_buffer
                println(f, line)
            end
        end

        if isempty(output_chains)
            error("No valid chains recorded during simulation on worker $(worker_id)")
        end
        for i in 1:length(output_chains)
            write_pdb_multiframe(output_chains[i], tem_beads[i], pdb_filename, i, params)
        end
        ensemble_contacts, ensemble_distances = 
            calculate_modulated_contact_probability(output_chains, tem_beads, params)
        
        # 3. 返回两个结果
        return ensemble_contacts, ensemble_distances

    end


    """
        _calculate_snapshot_maps(chain, free_beads, params)

    计算单个构象的接触概率图和距离图。
    这是一个为特定场景（固定顺序的CN2-CN3混合）进行超优化的内部函数。
    """
    function _calculate_snapshot_maps(
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D},
        params::SimulationParameters
    )
        N = length(chain)
        contact_map = zeros(Float64, N, N)
        distance_map = zeros(Float64, N, N)

        # 1. 首先计算该构象的距离图 (不变)
        for i in 1:N
            for j in i+1:N
                a = chain[i]; b = chain[j]
                dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
                dist = sqrt(dx*dx + dy*dy + dz*dz)
                distance_map[i, j] = distance_map[j, i] = dist
            end
        end

        # 2. 根据模型计算接触概率图
        total_N_B = sum(params.tf_counts)

        # 如果没有TF粒子，则直接返回
        if total_N_B <= 0
            # (可选：保留对旧的NB=-1模型的兼容性)
            if isdefined(params, :N_B) && params.N_B == -1
                k_c = params.calculate_contacts_kc
                r0 = params.calculate_contacts_r0
                for i in 1:N, j in i+1:N
                    r = distance_map[i, j]
                    Pij = 0.5 * (1.0 - tanh(k_c * (r - r0)))
                    contact_map[i, j] = contact_map[j, i] = Pij
                end
            end
            return contact_map, distance_map
        end
        
        # 模式二: 媒介接触模型 (CN2-CN3 超优化版)
        # --- [核心优化开始] ---
        log_term_matrix = zeros(Float64, N, N)

        k_c = params.k_c
        r0 = params.r0
        Pcutoff_ik_sq = params.Pcutoff_ik^2
        count1, count2 = params.tf_counts

        # --- [第一部分] 处理所有 CN2 粒子 (硬编码优化) ---
        initial_neighbors_N2 = ntuple(_ -> (Inf, -1), Val(2))
        for k_bead in 1:count1
            bead = free_beads[k_bead]
            closest_neighbors = initial_neighbors_N2
            for i_mono in 1:N
                mono = chain[i_mono]
                dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                rik_sq = dx*dx + dy*dy + dz*dz
                if rik_sq < Pcutoff_ik_sq && rik_sq < closest_neighbors[2][1]^2
                    rik = sqrt(rik_sq)
                    if rik < closest_neighbors[2][1]
                        closest_neighbors = insert_sorted_tuple_N2(closest_neighbors, rik, i_mono)
                    end
                end
            end
            closest_neighbors[2][2] == -1 && continue
            
            dist1, idx1 = closest_neighbors[1]
            dist2, idx2 = closest_neighbors[2]
            i_pair, j_pair = minmax(idx1, idx2)
            
            P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
            P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
            term = log(max(eps(Float64), 1.0 - P_ik * P_jk))
            log_term_matrix[i_pair, j_pair] += term
        end

        # --- [第二部分] 处理所有 CN3 粒子 (硬编码优化) ---
        start_index = count1 + 1
        end_index = count1 + count2
        initial_neighbors_N3 = ntuple(_ -> (Inf, -1), Val(3))
        for k_bead in start_index:end_index
            bead = free_beads[k_bead]
            closest_neighbors = initial_neighbors_N3
            for i_mono in 1:N
                mono = chain[i_mono]
                dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                rik_sq = dx*dx + dy*dy + dz*dz
                if rik_sq < Pcutoff_ik_sq && rik_sq < closest_neighbors[3][1]^2
                    rik = sqrt(rik_sq)
                    if rik < closest_neighbors[3][1]
                        closest_neighbors = insert_sorted_tuple_N3(closest_neighbors, rik, i_mono)
                    end
                end
            end
            closest_neighbors[2][2] == -1 && continue

            # 硬编码循环范围: C(3,2) = 3 对
            for i in 1:2
                dist1, idx1 = closest_neighbors[i]; idx1 == -1 && break
                for j in (i + 1):3
                    dist2, idx2 = closest_neighbors[j]; idx2 == -1 && break
                    i_pair, j_pair = minmax(idx1, idx2)

                    P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                    P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                    term = log(max(eps(Float64), 1.0 - P_ik * P_jk))
                    log_term_matrix[i_pair, j_pair] += term
                end
            end
        end
        # --- [核心优化结束] ---

        # 将 log-sum 矩阵转换为最终的接触概率矩阵 (不变)
        for i in 1:N
            for j in i+1:N
                Pij_effective = 1.0 - exp(log_term_matrix[i, j])
                contact_map[i, j] = contact_map[j, i] = Pij_effective
            end
        end

        return contact_map, distance_map
    end

    """
        calculate_modulated_contact_probability(output_chains, tem_beads, params)

    根据蒙特卡洛模拟的轨迹，计算平均接触概率图和平均距离图。

    该函数会根据 `params.N_B` 和 `params.N_max_contacts` 自动选择合适的
    接触概率定义（直接接触或自由珠子媒介的接触）。

    # 参数
    - `output_chains`: 一个向量，每个元素是模拟中一个时间点的聚合物链构象。
    类型: `Vector{Vector{Particle3D}}`
    - `tem_beads`: 一个向量，每个元素是对应 `output_chains` 时间点的自由珠子构象。
    类型: `Vector{Vector{Particle3D}}`
    - `params`: 包含所有模拟参数的结构体。

    # 返回
    - `avg_contact_map`: `N x N` 的矩阵，表示单体 i 和 j 之间的平均接触概率。
    - `avg_distance_map`: `N x N` 的矩阵，表示单体 i 和 j 之间的平均空间距离。
    """
    function calculate_modulated_contact_probability(
        output_chains::Vector{Vector{Particle3D}}, 
        tem_beads::Vector{Vector{Particle3D}}, 
        params::SimulationParameters
    )
        num_snapshots = length(output_chains)
        if num_snapshots == 0
            @warn "Input 'output_chains' is empty. Returning empty matrices."
            return Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0)
        end

        # 从第一个构象获取链长 N
        N = length(output_chains[1])
        
        # 初始化用于累加的矩阵
        total_contact_map = zeros(Float64, N, N)
        total_distance_map = zeros(Float64, N, N)

        println("Starting calculation over $num_snapshots snapshots...")

        # 遍历所有快照
        for s in 1:num_snapshots
            # 确保 tem_beads 和 output_chains 的长度一致
            # （在实际使用中可能需要更强的错误检查）
            current_chain = output_chains[s]
            current_beads = isempty(tem_beads) ? Vector{Particle3D}() : tem_beads[s]

            # 计算当前快照的接触图和距离图
            snapshot_contact, snapshot_distance = _calculate_snapshot_maps(
                current_chain, current_beads, params
            )

            # 累加到总矩阵中
            total_contact_map .+= snapshot_contact
            total_distance_map .+= snapshot_distance
            
            # (可选) 打印进度
            if s % 100 == 0 || s == num_snapshots
                println("Processed snapshot $s / $num_snapshots")
            end
        end

        # 计算平均值
        avg_contact_map = total_contact_map ./ num_snapshots
        avg_distance_map = total_distance_map ./ num_snapshots
        
        println("Calculation finished.")

        return avg_contact_map, avg_distance_map
    end

    """
    从构象系综中计算配位数。
    返回一个包含所有bead所有构象的配位数的长向量。
    """

    function calculate_coordination_numbers(
        ensemble_chains::Vector{Vector{Particle3D}}, 
        r_c::Float64, 
        mu::Float64
    )
        # The result will be floating-point numbers
        all_coord_numbers = Float64[]

        for chain in ensemble_chains
            n_beads = length(chain)
            # Pre-calculate a distance matrix for the current chain for efficiency
            dist_mat = zeros(Float64, n_beads, n_beads)
            for i in 1:n_beads
                for j in (i+1):n_beads
                    dist = distance(chain[i], chain[j])
                    dist_mat[i, j] = dist
                    dist_mat[j, i] = dist
                end
            end

            # Calculate the contact strength matrix C_ij
            # C_ij = 0.5 * (1 + tanh(μ * (r_c - r_ij)))
            arg_matrix = mu .* (r_c .- dist_mat)
            contact_strength_matrix = 0.5 .* (1.0 .+ tanh.(arg_matrix))
            
            # Set diagonal to zero to remove self-coordination
            # The diagonal of contact_strength_matrix would otherwise be 1.0
            for i in 1:n_beads
                contact_strength_matrix[i, i] = 0.0
            end

            # Calculate coordination numbers by summing the strengths for each bead
            # sum(matrix, dims=2) sums over columns for each row
            coord_numbers_for_this_chain = vec(sum(contact_strength_matrix, dims=2))
            
            # Append the results for this chain to the main list
            append!(all_coord_numbers, coord_numbers_for_this_chain)
        end
        
        return all_coord_numbers
    end

    function calculate_mean_coordination_number(chain::Vector{Particle3D}, contact_threshold::Float64)
        # 调用上面的函数，但只传入一个包含当前链的列表
        all_numbers = calculate_coordination_numbers([chain], 1.6, 4.0)
        # 计算并返回平均值。如果列表为空则返回0.0。
        return isempty(all_numbers) ? 0.0 : mean(all_numbers)
    end

end
println("Finished loading code on workers.")


# ==============================================================================
#                   B. 并行模拟与优化主循环
# ==============================================================================

# 并行模拟框架，现在接受固定的背景物理参数
function run_parallel_simulations(alpha_matrix, alpha_0_matrix; parallel_runs=10,iter_num=0)

    # 为当前迭代创建一个基础目录
    snapshot_base_dir = "sim_out/iter_$(iter_num)"
    mkpath(snapshot_base_dir)
    
    N = size(alpha_matrix, 1)

    # --- [新增] 定义混合TF种群 ---
    tf_counts_config = (40, 5)         # 例如: 10个A类TF, 10个B类TF
    tf_connectivities_config = (2, 3)   # 例如: A类连接数为2, B类连接数为3， 这个代码只进行2 和3的混合

    # --- [新增] 根据配置计算派生参数 ---
    total_N_B = sum(tf_counts_config)
    max_connectivity = maximum(tf_connectivities_config)
    
    # 创建每个TF的连接数映射
    connectivity_map = vcat(
        fill(tf_connectivities_config[1], tf_counts_config[1]),
        fill(tf_connectivities_config[2], tf_counts_config[2])
    )

    results_per_worker = @distributed (append!) for _ in 1:parallel_runs
        params = SimulationParameters(
            # --- [修改] 使用新的参数 ---
            alpha=alpha_matrix,
            alpha_0=alpha_0_matrix,
            
            tf_counts=tf_counts_config,
            tf_connectivities=tf_connectivities_config,
            tf_connectivity_map=connectivity_map,


            # --- [关键] 使用第一阶段找到的最佳背景参数 ---
            lj_epsilon=2.1,
            k_angle=0.2,
            lj_range=65, # 使用全范围的LJ

            # --- 其他所有相关的模拟参数 ---
            N=65, 
            MTf= 3.0, 
            Steps_FINAL=2e7,  #7是合理的,即使在Nconnect=6的情况下

            num_samples=500, # 每个模拟采集的样本数

            r0_bond=1.4, k_bond=10, De_bond=20.0,
            lj_sigma=0.5, lj_cutoff=2.5,
            R_wall=10.0,

            # free bead 相关参数
            r0= 1.2, k_c=14, Pcutoff_ik=1.5, 
            
            free_bead_LJ_ε=0.0,

            loop_cutoff=2.0, 
            factor_effective_cutoff_Pcutoff_ik=1.0,
            calculate_contacts_r0=1.6, # 这里的意义已经变了, 只是NB=-1时的r0
            calculate_contacts_kc=4.0

        )
        
        # run_simulation_core 返回接触图和距离图的系综
        [run_simulation_core(params,snapshot_base_dir)]
    end

    # 1. 提取所有 worker 的平均接触图并计算总平均值
    all_worker_means = [res[1] for res in results_per_worker]
    mean_contact = mean(all_worker_means) # 对平均值再取平均，得到总平均值

    # 2. 提取所有 worker 的距离图
    all_worker_distances = [res[2] for res in results_per_worker]
    mean_distance = mean(all_worker_distances) # 这里您可能需要不同的聚合方式

    # 您之前的代码试图聚合3D数组，但现在我们得到的是平均距离图，所以逻辑要简化
    sim_distance_ensemble_mean = mean_distance # 或者其他处理方式

    return mean_contact, sim_distance_ensemble_mean # 返回两个平均图
end

function optimize_alpha_main(initial_alpha, initial_alpha_0, target_contact; 
                             parallel_runs=10, maxiter=20, Step_start=1, correction_exponent=0.8, 
                             lambda_alpha_base=2.0, lambda_alpha0_base=0.0)

    alpha = copy(initial_alpha)
    alpha_0 = copy(initial_alpha_0)
    N = size(alpha, 1)
    k_max = N - 1 # 最大对角线索引
    # --- 设置日志目录 ---
    mkpath("sim_out")
    mkpath("alpha_log")
    mkpath("contacts")


    # --- 预计算距离校正矩阵 ---
    CorrectionMatrix = ones(Float64, N, N) # 初始化为1（无校正）
    if correction_exponent > 0
        for k in 2:k_max # 从次次对角线开始校正 (k=2)
            correction_factor = Float64(k)^correction_exponent
            # 设置上三角和下三角的校正因子
            for i in 1:(N-k)
                j = i + k
                CorrectionMatrix[i, j] = correction_factor
                CorrectionMatrix[j, i] = correction_factor # 如果需要对称更新
            end
        end
        println("Correction matrix calculated with exponent: $correction_exponent")
    else
        println("No distance correction applied (exponent is zero).")
    end

    # --- 主优化循环 ---
    for t in Step_start:maxiter
        println("\n--- Stage 2, Iteration: $t ---")

        # 1. 运行模拟
        save_contact_map(alpha, "alpha_log/$t.txt")

        simulated_contact, sim_distance_ensemble_3D = run_parallel_simulations(
            alpha, alpha_0, parallel_runs=parallel_runs, iter_num=t
        )
        println("Simulations complete.")

        # 2. 计算损失和相关性 (上三角，跳过 k=0, k=1)
        numerator = 0.0
        denominator = 0.0
        simulated_values = Float64[]
        target_values = Float64[]

        for i in 1:N
            for j in (i+2):N
                sim_val = simulated_contact[i, j]
                target_val = target_contact[i, j]

                numerator += abs(sim_val - target_val)
                denominator += target_val
                push!(simulated_values, sim_val)
                push!(target_values, target_val)
            end
        end

        loss = denominator ≈ 0.0 ? 0.0 : numerator / denominator
        correlation = cor(simulated_values, target_values) # 计算 Pearson 相关系数


        println("Hi-C Loss: ", loss, ", Pearson Correlation: ", correlation)
        open("loss.dat", "a") do io; println(io, t, " ", loss, " ", correlation); end
        save_contact_map(simulated_contact, "contacts/$t.txt")

        # 4. 更新 alpha (应用距离校正)
        delta = simulated_contact - target_contact # 完整矩阵差值
        λ_alpha = lambda_alpha_base / sqrt(t)     # alpha 的自适应学习率
        λ_alpha0 = lambda_alpha0_base / sqrt(t)    # alpha_0 的自适应学习率

        # 5. 更新 alpha_0 / γ_d (基于对角线均值差异)
        update_term = λ_alpha * (delta .* CorrectionMatrix)
        for i in 1:N
            for j in (i+1):N # 考虑更新 k=1 对角线，但校正从k=2开始
                 alpha[i,j] += update_term[i,j]
                 # 可选：强制对称性
                 # alpha[j,i] = alpha[i,j]
            end
        end
        
        # 将 alpha 中所有大于 0 的元素置为 0
        # alpha[alpha .> 0] .= 0
        
        # 6. 记录 alpha_0
        sim_means = zeros(Float64, k_max)
        target_means = zeros(Float64, k_max)
        valid_k = Int[] # 记录实际计算了均值的k

        for k in 1:k_max
             diag_sim = diag(simulated_contact, k)
             diag_target = diag(target_contact, k)
             if !isempty(diag_sim) # 确保对角线不为空
                 sim_means[k] = mean(diag_sim)
                 target_means[k] = mean(diag_target)
                 push!(valid_k, k)
            else
                println("Warning: Diagonal k=$k is empty, skipping alpha_0 update for this diagonal.")
            end
        end

        # 更新 alpha_0 的对角线
        for k in valid_k
            delta_k = sim_means[k] - target_means[k]
            # 获取当前 alpha_0 在该对角线上的值 (假设是 Toeplitz)
            # 这里假设 alpha_0 是上三角带状矩阵，对角线上值相同
            current_alpha0_k = alpha_0[1, 1+k]
            # 使用 '+' 因为我们希望模拟接触概率 sim_means[k] 接近 target_means[k]
            if k > 50
                λ_alpha0 = λ_alpha0 * 1.2 # 针对尾部进行特殊处理，现在的尾巴以及不会起来了
            end
            new_alpha0_k = current_alpha0_k + λ_alpha0 * delta_k
            alpha_0[diagind(alpha_0, k)] .= new_alpha0_k

        end

        # 7. 记录 alpha_0
        open("alpha_0_log.txt", "a") do io
            # 记录第一行的上三角部分（代表每个对角线的值）
            println(io, t, " ", join(round.(alpha_0[1, 2:end], digits=6), " "))
        end
        
        println("----------------------------------------")
    end

    println("Stage 2 Optimization finished.")
    return (best_alpha=alpha, best_alpha_0=alpha_0)
end


# ==============================================================================
#                            C. 主执行流程
# ==============================================================================


# --- 2. 准备优化所需的初始数据和参数 ---
println("\nPreparing for Stage 2 Optimization...")
N = 65
target_contact = read_alpha_from_file("/hpc2hdd/home/jtang163/work/BACKUP-0905/MC-small_bead_1/hic_data.dat", N)
# initial_alpha_0 = read_alpha_from_line("/hpc2hdd/home/jtang163/work/MC/alpha_0-ba/alpha0-NB20-5_3.txt", N)

initial_alpha_0 = zeros(Float64, N, N)
initial_alpha = zeros(Float64, N, N)

# --- 3. 运行第二阶段优化主函数 ---
opt_result = optimize_alpha_main(
    initial_alpha, 
    initial_alpha_0, 
    target_contact, 
    parallel_runs=10,
    Step_start=1
)

