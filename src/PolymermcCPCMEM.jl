module PolymermcCPCMEM

    using Distributed
    using ProgressMeter
    using LinearAlgebra
    using Statistics
    using Parameters
    using DelimitedFiles
    using CSV
    using DataFrames
    using Printf
    using Random
    using StaticArrays
    using Dates  
    # 导出需要的类型和函数
    export SimulationParameters, Particle3D, read_alpha_from_file, optimize_alpha_main

    # ==============================================================================
    # 1. 数据结构
    # ==============================================================================

    @with_kw mutable struct SimulationParameters
        N::Int = 65
        tf_counts::Tuple{Int, Int} = (0, 0)
        tf_connectivities::Tuple{Int, Int} = (0, 0)
        fbead_contact::Matrix{Int} = Matrix{Int}(undef, 0, 0)
        tf_connectivity_map::Vector{Int} = Int[]
        
        # 运行步数参数
        Steps_RUN::Int = 4000
        MSEP::Int = 1000
        Steps_FINAL::Int = 2000000
        num_samples::Int = 500

        # 温度参数
        MTi::Float64 = 50.0
        MTf::Float64 = 1.0
        
        # 物理参数
        k_angle::Float64 = 0.2
        theta0_angle::Float64 = 0.5  * π

        Pcutoff_ik::Float64 = 1.5
        k_c::Float64 = 14.0
        r0::Float64 = 1.2 # free bead contact r0
        
        # 键参数
        r0_bond::Float64 = 1.4
        k_bond::Float64 = 10.0
        De_bond::Float64 = 20.0
        DD::Float64 = 0.20

        # LJ 参数
        lj_epsilon::Float64 = 0.0
        lj_sigma::Float64 = 0.5
        lj_cutoff::Float64 = 2.5
        lj_range::Int = 65
        free_bead_LJ_ε::Float64 = 0.0

        # 墙和理想染色体
        R_wall::Float64 = 10.0
        ideal_chrom_r0::Float64 = 3.0
        ideal_chrom_kc::Float64 = 0.8

        # 接触计算参数 (NB=-1时使用, 或用于输出)
        calculate_contacts_r0::Float64 = 1.6
        calculate_contacts_kc::Float64 = 4.0
        coord_num_threshold::Float64 = 1.2
        
        # 空间参数
        min_distance::Float64 = 0.01
        box_size::Float64 = 20.0
        
        # 势能矩阵
        alpha::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
        alpha_0::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
        Pij_mediated_matrix::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)

        # Loop extrusion (占位)
        loop_relax_steps::Int = 1000
        loop_strength::Float64 = 1e5
        loop_cutoff::Float64 = 0.3
        loop_anchor::Union{Nothing,Tuple{Int,Int}} = nothing
        factor_effective_cutoff_Pcutoff_ik::Float64 = 1.0
    end

    struct Particle3D
        x::Float64; y::Float64; z::Float64
    end

    struct RejectMove <: Exception end

    # ==============================================================================
    # 2. 核心物理函数 (原 model_s.jl 中的所有物理函数)
    # ==============================================================================

    # 定义存储 N 个粒子的三维坐标的数据结构
    struct ParticleSystem
        particles::Vector{Particle3D}  # 存储 N 个粒子的数组
        ParticleSystem(N::Int) = new([Particle3D(0.0, 0.0, 0.0) for _ in 1:N])  # 初始化为零坐标
    end

    # 定义存储 NB 个粒子的三维坐标的数据结构
    struct FreeBeadSystem
        beads::Vector{Particle3D}  # 存储 NB 个自由粒子的数组
        FreeBeadSystem(NB::Int) = new([Particle3D(0.0, 0.0, 0.0) for _ in 1:NB])  # 初始化为零坐标
    end

    # 定义一个链的结构体
    struct Chain
        particles::Vector{Particle3D}  # 存储 N 个粒子的数组
        Chain(N::Int) = new([Particle3D(0.0, 0.0, 0.0) for _ in 1:N])  # 初始化为零坐标
    end



    function read_alpha_from_file(filename::String, N::Int)::Matrix{Float64}
        # 初始化 alpha 矩阵（N x N）
        alpha = zeros(Float64, N, N)

        # 打开文件并逐行读取
        open(filename, "r") do file
            for line in eachline(file)
                # 解析每一行的数据
                tokens = split(line)
                if length(tokens) == 3
                    i = parse(Int, tokens[1])   # 转换为 1-based 索引
                    j = parse(Int, tokens[2])   # 转换为 1-based 索引
                    value = parse(Float64, tokens[3])
                    
                    alpha[i, j] = value
                    # alpha[j, i] = value
                end
            end
        end

        return alpha
    end


    # 读取最后一行alpha_0
    function read_alpha_from_line(filename,N)

        last_line = nothing
        open(filename, "r") do io
            for line in eachline(io)
                last_line = line
            end
        end
        stripped_line = strip(last_line)

        parts = split(stripped_line)
        values_str = parts[1:end]
        values_num = Vector{Float64}(undef, N) # 预分配向量

        for i in eachindex(values_str)
            # 1. 移除字符串末尾的逗号 (如果存在)
            clean_str = rstrip(values_str[i], ',')
        
            # 2. 解析清理后的字符串为 Float64
            try
                values_num[i] = parse(Float64, clean_str)
            catch e
                @warn "Failed to parse value: $clean_str. Skipping." exception=e
                continue
            end
            
        end
        # --- 重建 alpha_0 矩阵 ---
        alpha_0 = zeros(Float64, N, N)
        for k in 1:N
            diag_val = values_num[k]
            # 填充上对角线
            alpha_0[diagind(alpha_0, k-1)] .= diag_val
        end
        return alpha_0
    end

    function initialize_random_chain(N::Int, bond_length::Float64, min_distance::Float64, max_attempts::Int)::Vector{Particle3D}
        # 初始化链的坐标数组
        chain = Vector{Particle3D}(undef, N)
        
        # 固定第一个单体在原点
        chain[1] = Particle3D(0.0, 0.0, 0.0)
        
        # 动态球形限制半径
        R = 60.0 * bond_length
        
        for i in 2:N
            valid_position_found = false
            attempts = 0
            
            while !valid_position_found && attempts < max_attempts
                # 随机生成方向
                theta = 1.0 * π * rand()  # 极角 [0, π]
                phi = 2.0 * π * rand()    # 方位角 [0, 2π]
                
                # 新单体的坐标
                dx = bond_length * sin(theta) * cos(phi)
                dy = bond_length * sin(theta) * sin(phi)
                dz = bond_length * cos(theta)
                
                new_x = chain[i-1].x + dx
                new_y = chain[i-1].y + dy
                new_z = chain[i-1].z + dz
                
                # 检查新位置是否满足最小距离要求
                if is_position_valid(chain[1:i-1], new_x, new_y, new_z, min_distance) &&
                compute_distance(0.0, 0.0, 0.0, new_x, new_y, new_z) <= R
                    
                    chain[i] = Particle3D(new_x, new_y, new_z)
                    valid_position_found = true
                else
                    attempts += 1
                end
            end
            
            # 如果超过最大尝试次数仍未找到有效位置，增大球形限制半径并重新尝试
            if !valid_position_found
                R *= 1.1  # 动态增大球形半径
                @warn "Increasing sphere radius to $R due to generation failure."
                i -= 1  # 重新尝试当前单体
            end
        end
        
        return chain
    end

    # 计算两点之间的距离
    function compute_distance(x1::Float64, y1::Float64, z1::Float64, x2::Float64, y2::Float64, z2::Float64)::Float64
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        return sqrt(dx^2 + dy^2 + dz^2)
    end

# 计算距离的平方，避免昂贵的 sqrt
    @inline function compute_distance_sq(a::Particle3D, b::Particle3D)::Float64
        return (a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2
    end


    # # 检查新位置是否满足最小距离要求
    # function is_position_valid(chain::Vector{Particle3D}, new_x::Float64, new_y::Float64, new_z::Float64, min_distance::Float64)::Bool
    #     for particle in chain
    #         if compute_distance(particle.x, particle.y, particle.z, new_x, new_y, new_z) < min_distance
    #             return false
    #         end
    #     end
    #     return true
    # end
    
    # # # 检查新位置是否满足最小距离要求
    # function is_position_valid(chain::Vector{Particle3D}, free_beads::Vector{Particle3D}, 
    #                         new_x::Float64, new_y::Float64, new_z::Float64, min_distance::Float64)::Bool
    #     # 检查链上的粒子
    #     for particle in chain
    #         if compute_distance(particle.x, particle.y, particle.z, new_x, new_y, new_z) < min_distance
    #             return false
    #         end
    #     end
        
    #     # 检查其他自由粒子
    #     for bead in free_beads
    #         if compute_distance(bead.x, bead.y, bead.z, new_x, new_y, new_z) < min_distance
    #             return false
    #         end
    #     end
        
    #     return true
    # end

    # # 辅助函数：检查新位置是否有效
    # function is_position_valid(
    #     chain::Vector{Particle3D},
    #     free_beads::Vector{Particle3D},
    #     min_distance::Float64
    # )::Bool
    #     # 检查链内距离
    #     for i in 1:length(chain), j in i+1:length(chain)
    #         if compute_distance(chain[i], chain[j]) < min_distance
    #             return false
    #         end
    #     end
        
    #     # 检查与自由粒子的距离
    #     for bead in free_beads, particle in chain
    #         if compute_distance(bead, particle) < min_distance
    #             return false
    #         end
    #     end
        
    #     return true
    # end

    function initialize_free_beads(NB::Int, chain::Vector{Particle3D}, min_distance::Float64, box_size::Float64)::Vector{Particle3D}
        if NB == -1
            return []
        end

        free_beads = Vector{Particle3D}(undef, NB)
        
        for i in 1:NB
            valid_position_found = false
            
            while !valid_position_found
                # 随机生成自由粒子的坐标
                new_x = rand() * box_size - box_size / 2
                new_y = rand() * box_size - box_size / 2
                new_z = rand() * box_size - box_size / 2
                
                # 检查新位置是否满足最小距离要求
                if is_position_valid(chain, free_beads[1:i-1], new_x, new_y, new_z, min_distance)
                    free_beads[i] = Particle3D(new_x, new_y, new_z)
                    valid_position_found = true
                end
                # free_beads[i] = Particle3D(new_x, new_y, new_z)
                # valid_position_found = true
            end
        end
        
        return free_beads
    end
# 检查新位置是否满足最小距离要求 (用于初始化或单粒子移动检查)
    function is_position_valid(
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D}, 
        new_x::Float64, new_y::Float64, new_z::Float64, 
        min_distance::Float64
    )::Bool
        # 1. 预计算距离平方阈值，避免后续开方
        min_dist_sq = min_distance^2

        # 检查链上的粒子
        @inbounds for p in chain
            dx = p.x - new_x
            dy = p.y - new_y
            dz = p.z - new_z
            dist_sq = dx*dx + dy*dy + dz*dz
            
            if dist_sq < min_dist_sq
                return false
            end
        end
        
        # 检查其他自由粒子
        @inbounds for b in free_beads
            dx = b.x - new_x
            dy = b.y - new_y
            dz = b.z - new_z
            dist_sq = dx*dx + dy*dy + dz*dz
            
            if dist_sq < min_dist_sq
                return false
            end
        end
        
        return true
    end

# 辅助函数：检查新位置是否有效 (针对 Free Beads 移动)
    function is_position_valid(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        min_distance::Float64
    )::Bool
        min_distance_sq = min_distance^2 # 预计算平方阈值

        # 检查链内距离 (如果链不刚性移动，这部分其实在 FreeBead Move 中可以省略，
        # 但为保持通用性，我们优化它)
        for i in 1:length(chain), j in i+1:length(chain)
            if compute_distance_sq(chain[i], chain[j]) < min_distance_sq
                return false
            end
        end
        
        # 检查与自由粒子的距离
        for bead in free_beads, particle in chain
            if compute_distance_sq(bead, particle) < min_distance_sq
                return false
            end
        end
        
        # 自由粒子之间的距离检查 (原代码似乎漏了，如果需要也加上)
        # for i in 1:length(free_beads), j in i+1:length(free_beads)
        #     if compute_distance_sq(free_beads[i], free_beads[j]) < min_distance_sq
        #         return false
        #     end
        # end
        
        return true
    end

    # 针对生成链时的新位置验证函数
    function is_position_valid(chain::Vector{Particle3D}, new_x::Float64, new_y::Float64, new_z::Float64, min_distance::Float64)::Bool
        min_distance_sq = min_distance^2
        new_p = Particle3D(new_x, new_y, new_z)
        for particle in chain
            if compute_distance_sq(particle, new_p) < min_distance_sq
                return false
            end
        end
        return true
    end
    
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
            
        end

        # 计算平均值
        avg_contact_map = total_contact_map ./ num_snapshots
        avg_distance_map = total_distance_map ./ num_snapshots

        return avg_contact_map, avg_distance_map
    end

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
        total_N_B = params.tf_counts[1]

        # 如果没有TF粒子，则直接返回
        if total_N_B <= 0
            
            k_c = params.calculate_contacts_kc
            r0 = params.calculate_contacts_r0
            for i in 1:N, j in i+1:N
                r = distance_map[i, j]
                Pij = 0.5 * (1.0 - tanh(k_c * (r - r0)))
                contact_map[i, j] = contact_map[j, i] = Pij
            end

            return contact_map, distance_map
        end
        
        # 模式二: 媒介接触模型 (通用 K 邻居版)
        log_term_matrix = zeros(Float64, N, N)

        k_c = params.k_c
        r0 = params.r0
        Pcutoff_ik_sq = params.Pcutoff_ik^2
        
        # 追踪当前处理的 bead 在 total_beads 数组中的起始索引
        current_bead_start_idx = 1

        # 遍历每种类型的 TF (根据 counts 和 connectivities 配对)
        # 例如: (100, 50) counts 和 (2, 6) connectivities
        for (count, K) in zip(params.tf_counts, params.tf_connectivities)
            
            # 如果该类型数量为0，跳过
            if count == 0
                continue
            end

            range_end = current_bead_start_idx + count - 1

            # 处理该类型的每一个 bead
            for k_bead in current_bead_start_idx:range_end
                bead = free_beads[k_bead]
                
                # 初始化 K 个最近邻居列表 [(dist, index), ...]
                # 因为只用于快照分析，这里使用简单的 Vector 即可
                closest_neighbors = Vector{Tuple{Float64, Int}}(undef, K)
                fill!(closest_neighbors, (Inf, -1))

                # 扫描链上所有单体寻找 K 个最近邻
                for i_mono in 1:N
                    mono = chain[i_mono]
                    dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                    rik_sq = dx*dx + dy*dy + dz*dz
                    
                    if rik_sq < Pcutoff_ik_sq
                        rik = sqrt(rik_sq)
                        
                        # 如果比当前第 K 个近，尝试插入排序
                        if rik < closest_neighbors[K][1]
                            # 简单的插入排序逻辑
                            for insert_pos in 1:K
                                if rik < closest_neighbors[insert_pos][1]
                                    # 将 insert_pos 之后的元素后移
                                    for shift_k in K:-1:(insert_pos + 1)
                                        closest_neighbors[shift_k] = closest_neighbors[shift_k - 1]
                                    end
                                    # 插入新元素
                                    closest_neighbors[insert_pos] = (rik, i_mono)
                                    break
                                end
                            end
                        end
                    end
                end

                # 基于找到的最近邻居计算对 Log Term 矩阵的贡献
                # 遍历所有可能的唯一配对: C(K, 2)
                for i in 1:(K-1)
                    dist1, idx1 = closest_neighbors[i]
                    if idx1 == -1; break; end # 如果第 i 个邻居无效，后面的肯定也无效

                    for j in (i + 1):K
                        dist2, idx2 = closest_neighbors[j]
                        if idx2 == -1; break; end # 如果第 j 个邻居无效

                        # 找到有效配对
                        i_pair, j_pair = minmax(idx1, idx2)
                        
                        # 计算媒介概率贡献
                        P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                        P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                        
                        term = log(max(eps(Float64), 1.0 - P_ik * P_jk))
                        log_term_matrix[i_pair, j_pair] += term
                    end
                end
            end

            # 更新下一组 bead 的起始索引
            current_bead_start_idx += count
        end

        # 将 log-sum 矩阵转换为最终的接触概率矩阵 (不变)
        for i in 1:N
            for j in i+1:N
                Pij_effective = 1.0 - exp(log_term_matrix[i, j])
                contact_map[i, j] = contact_map[j, i] = Pij_effective
            end
        end

        return contact_map, distance_map
    end




    # 总能量函数
    function compute_total_energy!(
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D},
        params::SimulationParameters  # 从输入文件读取的参数

    )::Float64

        total_energy = 0.0

        # 1. 计算链的键能（Bond Energy）
        total_energy += compute_bond_energy(chain, params)

        # 2. 计算链的键角能（Bond Angle Energy）
        total_energy += compute_bond_angle_energy(chain, params)

        # 3. 计算链内非键相互作用能（Chain Nonbonded）
        total_energy += compute_chain_nonbond_energy(chain, params)

        # 4. 计算自由粒子间相互作用能（Free Bead Nonbonded）
        total_energy += compute_free_bead_energy(free_beads, params)
        
        # 5. 与球形壁的相互作用
        total_energy += compute_wall_interaction(chain, params)

        # 6. [重构] 计算所有特异性相互作用能 (alpha 项)
        total_energy += compute_specific_interaction_energy(chain, free_beads, params)

        # 7. ideal chromosome term 是一个对角线上都是一个数值的alpha矩阵计算的Pij
        
        # total_energy += ideal_chromosome_Pij(chain, params)
      

        # # 添加临时loop势
        # if !isnothing(params.loop_anchor)
        #     a, b = params.loop_anchor
        #     r = distance(chain[a], chain[b])
        #     if r > params.loop_cutoff
        #         total_energy += params.loop_strength * (r - params.loop_cutoff)^2
        #     end
        # end
            
        return total_energy
    end

    function distance(p1::Particle3D, p2::Particle3D)::Float64
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return sqrt(dx^2 + dy^2 + dz^2)
    end

    function compute_bond_energy(chain::Vector{Particle3D}, params::SimulationParameters)::Float64
        energy = 0.0
        # Dₑ: 势阱深度。描述了键的强度。决定了断开时的能量。
        De = params.De_bond 
        # r₀: 平衡键长，即势能最低点的位置。
        r0 = params.r0_bond
        # k: 键在平衡点附近的有效力常数。
        k = params.k_bond

        # a: 莫尔斯势的宽度参数。
        #    我们不直接设置它，而是通过 Dₑ 和 k 计算得出，这样更直观。
        #    关系式为: k = 2 * a² * Dₑ  =>  a = sqrt(k / (2 * Dₑ))
        #    注意：需要确保 k 和 De 都是正数，以避免计算错误。
        if k <= 0.0 || De <= 0.0
            error("k_bond and De_bond must be positive for Morse potential calculation.")
        end
        a = sqrt(k / (2.0 * De))

        # --- 遍历链中的所有键 ---
        for i in 1:length(chain)-1
            p1 = chain[i]
            p2 = chain[i+1]

            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            
            distance = sqrt(dx^2 + dy^2 + dz^2)

            # 计算莫尔斯势能 U(r) = Dₑ * (1 - exp(-a * (r - r₀)))²
            exponent_term = exp(-a * (distance - r0))
            potential_term = 1.0 - exponent_term
            
            bond_energy = De * potential_term^2
            
            energy += bond_energy
        end
        
        return energy
    end

    # 2. 键角能计算
    function compute_bond_angle_energy(chain::Vector{Particle3D}, params::SimulationParameters)::Float64
        energy = 0.0
        k_angle = params.k_angle  # 从参数读取键角常数
        theta0 = params.theta0_angle 
        # theta0 = 0.5 * π
        EPS = eps(Float64)

        @inbounds for i in 2:length(chain)-1
            p_prev = chain[i-1]
            p_curr = chain[i]
            p_next = chain[i+1]

            # --- 不使用 [x,y,z] 数组，直接用标量 ---
            
            # 向量 1: p_prev - p_curr
            v1x = p_prev.x - p_curr.x
            v1y = p_prev.y - p_curr.y
            v1z = p_prev.z - p_curr.z
            
            # 向量 2: p_next - p_curr
            v2x = p_next.x - p_curr.x
            v2y = p_next.y - p_curr.y
            v2z = p_next.z - p_curr.z
            
            # 模长
            r1 = sqrt(v1x^2 + v1y^2 + v1z^2)
            r2 = sqrt(v2x^2 + v2y^2 + v2z^2)
            
            denominator = r1 * r2
            
            if denominator < EPS
                continue
            end

            # 点积
            dot_val = v1x*v2x + v1y*v2y + v1z*v2z

            # 计算余弦并截断
            cos_theta = dot_val / denominator
            if cos_theta > 1.0
                cos_theta = 1.0
            elseif cos_theta < -1.0
                cos_theta = -1.0
            end

            # 键角能量公式
            theta_diff = acos(cos_theta) - theta0
            energy += k_angle * (1.0 - cos(theta_diff))
            # --- 优化结束 ---
        end

        return energy
    end

    # 3. 链内非键相互作用能, 使用标准的Lennard-Jones势能
    function compute_chain_nonbond_energy(chain::Vector{Particle3D}, params::SimulationParameters)::Float64
        energy = 0.0

        ε = params.lj_epsilon
        σ = params.lj_sigma
        lj_range = params.lj_range
        cutoff_sq = params.lj_cutoff^2  # 使用距离的平方进行比较以避免开方运算

        # 如果epsilon为0, 则没有非键相互作用, 直接返回0以提高效率
        if ε == 0.0
            return 0.0
        end

        # 预计算常数, 确保在截断处能量为0 (Shifted LJ potential)
        # 这可以避免能量在截断点发生跳跃
        inv_cutoff_sq = 1.0 / cutoff_sq
        inv_cutoff_6 = inv_cutoff_sq^3
        inv_cutoff_12 = inv_cutoff_6^2
        shift_energy = 4.0 * ε * (inv_cutoff_12 - inv_cutoff_6)

        for i in 1:length(chain)
            # 内层循环只到 i + 5，同时要确保不超出链的长度
            # 并且 j 仍然需要大于 i+1 以避免键和角
            start_j = i + 2 
            end_j = min(i + lj_range, length(chain)) # 关键修改：限制j的最大值
            for j in start_j:end_j

                dx = chain[i].x - chain[j].x
                dy = chain[i].y - chain[j].y
                dz = chain[i].z - chain[j].z
                r_sq = dx^2 + dy^2 + dz^2 # 使用距离的平方

                if r_sq < cutoff_sq
                    # --- [MODIFIED] 使用标准的12-6 LJ势能 ---
                    inv_r_sq = 1.0 / r_sq
                    inv_r_6 = inv_r_sq^3
                    inv_r_12 = inv_r_6^2
                    
                    # 计算LJ势并加上能量偏移
                    lj_potential = 4.0 * ε * (σ^12 * inv_r_12 - σ^6 * inv_r_6)
                    energy += lj_potential - shift_energy
                end
            end
        end
        
        return energy
    end

    # 4. 自由粒子间相互作用能, 标准LJ势能
    function compute_free_bead_energy(free_beads::Vector{Particle3D}, params::SimulationParameters)::Float64
        energy = 0.0
        
        # 假设 σ 是基础长度单位，我们可以直接定义它。
        # 例如，我们可以从 params 中获取 σ，或者在这里为了演示而设定一个。
        # 为了与您之前的代码兼容，我们继续从 r_min 推导 σ。
        r_min = params.factor_effective_cutoff_Pcutoff_ik * params.Pcutoff_ik
        ε = params.free_bead_LJ_ε  
        σ = r_min / (2.0^(1.0/6.0))

        # **关键改动**: 选择一个更大的截断距离，典型的选择是 2.5σ
        cutoff_lj = 2.5 * σ
        cutoff_lj_sq = cutoff_lj^2

        # LJ 参数
        σ_pow6 = σ^6
        
        four_ε = 4.0 * ε

        # **关键改动**: 计算在新的、更大的截断距离 cutoff_lj 处的LJ势能值，作为平移量
        # V_LJ(r_c) = 4ε * [ (σ/r_c)^12 - (σ/r_c)^6 ]
        term_at_cutoff = σ_pow6 / (cutoff_lj^6)
        V_lj_at_cutoff = four_ε * (term_at_cutoff^2 - term_at_cutoff)

        for i in 1:length(free_beads)
            for j in i+1:length(free_beads)
                dx = free_beads[i].x - free_beads[j].x
                dy = free_beads[i].y - free_beads[j].y
                dz = free_beads[i].z - free_beads[j].z
                r_sq = dx^2 + dy^2 + dz^2
                
                # 使用新的截断距离进行判断
                if r_sq < cutoff_lj_sq
                    r_pow6 = r_sq^3
                    term = σ_pow6 / r_pow6
                    
                    # V_shifted(r) = V_LJ(r) - V_LJ(r_c)
                    lj_potential = four_ε * (term^2 - term)

                    if ε ==0 && lj_potential < 0 # 处理 ε 为 0 的情况，为纯排斥势能
                        lj_potential = 0.0
                    end

                    energy += (lj_potential - V_lj_at_cutoff)
                end
            end
        end
        return energy
    end


    #5. 计算单体与球形壁的相互作用势能
    function compute_wall_interaction(chain::Vector{Particle3D}, params::SimulationParameters)::Float64
        energy = 0.0
        σ = params.r0_bond      # Lennard-Jones 参数
        ε = 1.0      # Lennard-Jones 参数
        Rc = [0.0, 0.0, 0.0]    # 球形壁的中心坐标
        Rw = params.R_wall    # 球形壁的半径
        cutoff_distance = σ * 2^(1/6)  # 截断距离
        
        for monomer in chain
            # 计算单体到球形壁的最近距离
            dx = monomer.x - Rc[1]
            dy = monomer.y - Rc[2]
            dz = monomer.z - Rc[3]
            distance_to_center = sqrt(dx^2 + dy^2 + dz^2)
            r_np = distance_to_center - Rw
            
            # 处理单体在球形壁外部的情况
            if r_np < 0
                r_np = abs(r_np)
            else
                energy += 0.5 * ε * (r_np)^6
                continue
            end
            
            # 计算相互作用势能
            if r_np <= cutoff_distance
                LJ_term = 4ε * ((σ / r_np)^12 - (σ / r_np)^6 + 1/4)
                energy += LJ_term
            end
        end
        
        return energy
    end

    function compute_specific_interaction_energy(
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D},
        params::SimulationParameters
    )::Float64
    #   这里删除了alpha 全0 的检测以及loop的检测

        if params.tf_counts[1] == -1
           return compute_total_Pij_IJ(chain, free_beads, params)
        end

        N_chain = length(chain)
        # 注意：这里我们只清空一次矩阵！
        Pij_mediated_matrix = params.Pij_mediated_matrix 
        fill!(Pij_mediated_matrix, 0.0)

        # --- 2. 分发任务 ---
        current_idx = 1
        
        # 遍历配置 (例如: counts=(50, 50), conns=(2, 6))
        for (count, K) in zip(params.tf_counts, params.tf_connectivities)
            if count == 0; continue; end
            
            # 确定这一批粒子的索引范围 (例如 1:50 或 51:100)
            range_end = current_idx + count - 1
            bead_range = current_idx:range_end

            # 调用修改后的 Kernel 函数
            if K == 2
                _process_batch_N2!(bead_range, chain, free_beads, params)
            elseif K == 3
                _process_batch_N3!(bead_range, chain, free_beads, params)
            elseif K == 4
                _process_batch_N4!(bead_range, chain, free_beads, params)
            elseif K == 5
                _process_batch_N5!(bead_range, chain, free_beads, params)
            elseif K == 6
                _process_batch_N6!(bead_range, chain, free_beads, params)
            end

            current_idx += count
        end

        # --- 3. 最终统一计算总能量 (SIMD优化) ---
        total_Pij = 0.0
        @inbounds for i in 1:N_chain
            alpha_row = view(params.alpha, i, :)
            Pij_mediated_row = view(Pij_mediated_matrix, i, :)
            sum_val = 0.0
            @simd for j in (i + 2):N_chain
                # 此时 Pij_mediated_row 包含了 N2, N6 等所有粒子的贡献总和
                Pij_effective = 1.0 - exp(Pij_mediated_row[j])
                sum_val += alpha_row[j] * Pij_effective
            end
            total_Pij += sum_val
        end
        
        return total_Pij
    end

    # 原来的最大熵方法
    function compute_total_Pij_IJ(
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D},
        params::SimulationParameters
    )::Float64

        N = length(chain)
        total_Pij = 0.0
        k_c, r0 = params.calculate_contacts_kc, params.calculate_contacts_r0  # 预存参数
        alpha = params.alpha

        for i in 1:N-2  # 外层循环优化范围
            a = chain[i]
            ax, ay, az = a.x, a.y, a.z  # 预存坐标
            
            for j in i+2:N
                b = chain[j]
                dx = ax - b.x
                dy = ay - b.y
                dz = az - b.z
                r = sqrt(dx*dx + dy*dy + dz*dz)  # 内联距离计算
                Pij = 0.5 * (1.0 - tanh(k_c * (r - r0)))
                total_Pij += alpha[i, j] * Pij
            end
        end

        return total_Pij
    end

    # Pij 计算的核心函数，N=4,5,6的专用版本
    @inline function insert_sorted_tuple_N2(
        current::NTuple{2, Tuple{Float64, Int}}, dist::Float64, idx::Int
    )
        new = (dist, idx)
        # If the new distance is less than the first, it becomes the new first.
        # Otherwise, it must be the second (because the outer check ensures it's smaller than the current second).
        if dist < current[1][1]
            return (new, current[1])
        else
            return (current[1], new)
        end
    end

    # 和 insert_sorted_tuple_N4 等函数放在一起
    @inline function insert_sorted_tuple_N3(
        current::NTuple{3, Tuple{Float64, Int}}, dist::Float64, idx::Int
    )
        new = (dist, idx)
        dist < current[1][1] && return (new, current[1], current[2])
        dist < current[2][1] && return (current[1], new, current[2])
        return (current[1], current[2], new)
    end

    @inline function insert_sorted_tuple_N4(
        current::NTuple{4, Tuple{Float64, Int}}, dist::Float64, idx::Int
    )
        new = (dist, idx)
        dist < current[1][1] && return (new, current[1], current[2], current[3])
        dist < current[2][1] && return (current[1], new, current[2], current[3])
        dist < current[3][1] && return (current[1], current[2], new, current[3])
        return (current[1], current[2], current[3], new)
    end

    @inline function insert_sorted_tuple_N5(
        current::NTuple{5, Tuple{Float64, Int}}, dist::Float64, idx::Int
    )
        new = (dist, idx)
        dist < current[1][1] && return (new, current[1], current[2], current[3], current[4])
        dist < current[2][1] && return (current[1], new, current[2], current[3], current[4])
        dist < current[3][1] && return (current[1], current[2], new, current[3], current[4])
        dist < current[4][1] && return (current[1], current[2], current[3], new, current[4])
        return (current[1], current[2], current[3], current[4], new)
    end

    @inline function insert_sorted_tuple_N6(
        current::NTuple{6, Tuple{Float64, Int}}, dist::Float64, idx::Int
    )
        new = (dist, idx)
        dist < current[1][1] && return (new, current[1], current[2], current[3], current[4], current[5])
        dist < current[2][1] && return (current[1], new, current[2], current[3], current[4], current[5])
        dist < current[3][1] && return (current[1], current[2], new, current[3], current[4], current[5])
        dist < current[4][1] && return (current[1], current[2], current[3], new, current[4], current[5])
        dist < current[5][1] && return (current[1], current[2], current[3], current[4], new, current[5])
        return (current[1], current[2], current[3], current[4], current[5], new)
    end

    function _process_batch_N2!(
        bead_range::UnitRange{Int}, 
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D}, 
        params::SimulationParameters
    )

        N_MAX_C = 2
        N_chain = length(chain) # 必须定义，否则循环报错
        

        Pij_mediated_matrix = params.Pij_mediated_matrix 

        Pcutoff_ik_sq = params.Pcutoff_ik^2
        k_c = params.k_c
        r0 = params.r0

        initial_neighbors = ((Inf, -1), (Inf, -1))

        for k_bead in bead_range
            bead = free_beads[k_bead]
            closest_neighbors = initial_neighbors

            for i_mono in 1:N_chain
                mono = chain[i_mono]
                dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                rik_sq = dx*dx + dy*dy + dz*dz

                if rik_sq < Pcutoff_ik_sq && rik_sq < closest_neighbors[N_MAX_C][1]^2
                    rik = sqrt(max(0.0, rik_sq))
                    if rik < closest_neighbors[N_MAX_C][1]
                        closest_neighbors = insert_sorted_tuple_N2(closest_neighbors, rik, i_mono)
                    end
                end
            end

            params.fbead_contact[k_bead, 1] = closest_neighbors[1][2]
            params.fbead_contact[k_bead, 2] = closest_neighbors[2][2]
            
            if closest_neighbors[2][2] == -1
                continue
            end

            dist1, idx1 = closest_neighbors[1]
            dist2, idx2 = closest_neighbors[2]

            i_pair, j_pair = minmax(idx1, idx2)
            
            if (j_pair - i_pair) >= 2
                @fastmath begin
                    P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                    P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                    arg = 1.0 - P_ik * P_jk
                    term = log(max(eps(Float64), arg))
                end
                @inbounds Pij_mediated_matrix[i_pair, j_pair] += term
            end
        end
        return nothing
    end

    function _process_batch_N3!(
        bead_range::UnitRange{Int}, # 新增参数
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D}, 
        params::SimulationParameters
    )
        N_MAX_C = 3
        N_chain = length(chain)
        
        # 不要清零矩阵
        Pij_mediated_matrix = params.Pij_mediated_matrix 

        Pcutoff_ik_sq = params.Pcutoff_ik^2
        k_c = params.k_c
        r0 = params.r0

        initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

        for k_bead in bead_range # 修改循环范围
            bead = free_beads[k_bead]
            closest_neighbors = initial_neighbors

            for i_mono in 1:N_chain
                mono = chain[i_mono]
                dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                rik_sq = dx*dx + dy*dy + dz*dz

                if rik_sq < Pcutoff_ik_sq && rik_sq < closest_neighbors[N_MAX_C][1]^2
                    rik = sqrt(max(0.0, rik_sq))
                    if rik < closest_neighbors[N_MAX_C][1]
                        closest_neighbors = insert_sorted_tuple_N3(closest_neighbors, rik, i_mono)
                    end
                end
            end

            params.fbead_contact[k_bead, 1] = closest_neighbors[1][2]
            params.fbead_contact[k_bead, 2] = closest_neighbors[2][2]
            params.fbead_contact[k_bead, 3] = closest_neighbors[3][2]
            
            closest_neighbors[2][2] == -1 && continue

            # N=3 特定逻辑 C(3,2)
            for i in 1:(N_MAX_C-1)
                dist1, idx1 = closest_neighbors[i]; idx1 == -1 && break
                for j in (i + 1):N_MAX_C
                    dist2, idx2 = closest_neighbors[j]; idx2 == -1 && break
                    i_pair, j_pair = minmax(idx1, idx2)
                    (j_pair - i_pair) < 2 && continue

                    @fastmath begin
                        P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                        P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                        arg = 1.0 - P_ik * P_jk
                        term = log(max(eps(Float64), arg))
                    end
                    @inbounds Pij_mediated_matrix[i_pair, j_pair] += term
                end
            end
        end
        return nothing
    end

    function _process_batch_N4!(
        bead_range::UnitRange{Int}, 
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D}, 
        params::SimulationParameters
    )
        N_MAX_C = 4
        N_chain = length(chain)
        Pij_mediated_matrix = params.Pij_mediated_matrix 

        Pcutoff_ik_sq = params.Pcutoff_ik^2
        k_c = params.k_c
        r0 = params.r0

        initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

        for k_bead in bead_range
            bead = free_beads[k_bead]
            closest_neighbors = initial_neighbors

            for i_mono in 1:N_chain
                mono = chain[i_mono]
                dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                rik_sq = dx*dx + dy*dy + dz*dz

                if rik_sq < Pcutoff_ik_sq && rik_sq < closest_neighbors[N_MAX_C][1]^2
                    rik = sqrt(max(0.0, rik_sq))
                    if rik < closest_neighbors[N_MAX_C][1]
                        closest_neighbors = insert_sorted_tuple_N4(closest_neighbors, rik, i_mono)
                    end
                end
            end

            # Unroll saving contacts
            params.fbead_contact[k_bead, 1] = closest_neighbors[1][2]
            params.fbead_contact[k_bead, 2] = closest_neighbors[2][2]
            params.fbead_contact[k_bead, 3] = closest_neighbors[3][2]
            params.fbead_contact[k_bead, 4] = closest_neighbors[4][2]
            
            closest_neighbors[2][2] == -1 && continue

            for i in 1:(N_MAX_C-1)
                dist1, idx1 = closest_neighbors[i]; idx1 == -1 && break
                for j in (i + 1):N_MAX_C
                    dist2, idx2 = closest_neighbors[j]; idx2 == -1 && break
                    i_pair, j_pair = minmax(idx1, idx2)
                    (j_pair - i_pair) < 2 && continue

                    @fastmath begin
                        P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                        P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                        arg = 1.0 - P_ik * P_jk
                        term = log(max(eps(Float64), arg))
                    end
                    @inbounds Pij_mediated_matrix[i_pair, j_pair] += term
                end
            end
        end
        return nothing
    end

    function _process_batch_N5!(
        bead_range::UnitRange{Int}, 
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D}, 
        params::SimulationParameters
    )
        N_MAX_C = 5
        N_chain = length(chain)
        Pij_mediated_matrix = params.Pij_mediated_matrix 

        Pcutoff_ik_sq = params.Pcutoff_ik^2
        k_c = params.k_c
        r0 = params.r0

        initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

        for k_bead in bead_range
            bead = free_beads[k_bead]
            closest_neighbors = initial_neighbors

            for i_mono in 1:N_chain
                mono = chain[i_mono]
                dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                rik_sq = dx*dx + dy*dy + dz*dz

                if rik_sq < Pcutoff_ik_sq && rik_sq < closest_neighbors[N_MAX_C][1]^2
                    rik = sqrt(max(0.0, rik_sq))
                    if rik < closest_neighbors[N_MAX_C][1]
                        closest_neighbors = insert_sorted_tuple_N5(closest_neighbors, rik, i_mono)
                    end
                end
            end

            params.fbead_contact[k_bead, 1] = closest_neighbors[1][2]
            params.fbead_contact[k_bead, 2] = closest_neighbors[2][2]
            params.fbead_contact[k_bead, 3] = closest_neighbors[3][2]
            params.fbead_contact[k_bead, 4] = closest_neighbors[4][2]
            params.fbead_contact[k_bead, 5] = closest_neighbors[5][2]

            closest_neighbors[2][2] == -1 && continue

            for i in 1:(N_MAX_C-1)
                dist1, idx1 = closest_neighbors[i]; idx1 == -1 && break
                for j in (i + 1):N_MAX_C
                    dist2, idx2 = closest_neighbors[j]; idx2 == -1 && break
                    i_pair, j_pair = minmax(idx1, idx2)
                    (j_pair - i_pair) < 2 && continue

                    @fastmath begin
                        P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                        P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                        arg = 1.0 - P_ik * P_jk
                        term = log(max(eps(Float64), arg))
                    end
                    @inbounds Pij_mediated_matrix[i_pair, j_pair] += term
                end
            end
        end
        return nothing
    end

    function _process_batch_N6!(
        bead_range::UnitRange{Int}, 
        chain::Vector{Particle3D}, 
        free_beads::Vector{Particle3D}, 
        params::SimulationParameters
    )
        N_MAX_C = 6
        N_chain = length(chain)
        Pij_mediated_matrix = params.Pij_mediated_matrix 

        Pcutoff_ik_sq = params.Pcutoff_ik^2
        k_c = params.k_c
        r0 = params.r0

        initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

        for k_bead in bead_range
            bead = free_beads[k_bead]
            closest_neighbors = initial_neighbors

            for i_mono in 1:N_chain
                mono = chain[i_mono]
                dx = mono.x - bead.x; dy = mono.y - bead.y; dz = mono.z - bead.z
                rik_sq = dx*dx + dy*dy + dz*dz

                if rik_sq < Pcutoff_ik_sq && rik_sq < closest_neighbors[N_MAX_C][1]^2
                    rik = sqrt(max(0.0, rik_sq))
                    if rik < closest_neighbors[N_MAX_C][1]
                        closest_neighbors = insert_sorted_tuple_N6(closest_neighbors, rik, i_mono)
                    end
                end
            end

            params.fbead_contact[k_bead, 1] = closest_neighbors[1][2]
            params.fbead_contact[k_bead, 2] = closest_neighbors[2][2]
            params.fbead_contact[k_bead, 3] = closest_neighbors[3][2]
            params.fbead_contact[k_bead, 4] = closest_neighbors[4][2]
            params.fbead_contact[k_bead, 5] = closest_neighbors[5][2]
            params.fbead_contact[k_bead, 6] = closest_neighbors[6][2]

            closest_neighbors[2][2] == -1 && continue

            for i in 1:(N_MAX_C-1)
                dist1, idx1 = closest_neighbors[i]; idx1 == -1 && break
                for j in (i + 1):N_MAX_C
                    dist2, idx2 = closest_neighbors[j]; idx2 == -1 && break
                    i_pair, j_pair = minmax(idx1, idx2)
                    (j_pair - i_pair) < 2 && continue

                    @fastmath begin
                        P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                        P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                        arg = 1.0 - P_ik * P_jk
                        term = log(max(eps(Float64), arg))
                    end
                    @inbounds Pij_mediated_matrix[i_pair, j_pair] += term
                end
            end
        end
        return nothing
    end


    function ideal_chromosome_Pij(
        chain::Vector{Particle3D}, 
        params::SimulationParameters
    )::Float64
        N = length(chain)
        total_Pij = 0.0

        # 预存参数到局部变量
        kc, r0 = params.ideal_chrom_kc, params.ideal_chrom_r0
        alpha_0 = params.alpha_0

        # 优化外层循环范围，避免无效迭代
        for i in 1:N-2
            a = chain[i]
            ax, ay, az = a.x, a.y, a.z  # 预存当前粒子坐标
            
            for j in i+2:N
                b = chain[j]
                dx = ax - b.x
                dy = ay - b.y
                dz = az - b.z
                r = sqrt(dx*dx + dy*dy + dz*dz)  # 内联距离计算
                
                # 计算 Pij 并累加
                Pij = 0.5 * (1.0 - tanh(kc * (r - r0)))
                total_Pij += alpha_0[i, j] * Pij
            end
        end

        return total_Pij
    end

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # 辅助函数：计算两个粒子之间的距离
    compute_distance(a::Particle3D, b::Particle3D) = 
        sqrt((a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2)

    function metropolis_accept(ΔE::Float64, T::Float64)::Bool
        if ΔE < 0
            return true
        else
            x=rand()

            return x < exp(-ΔE / T)

            # return false
        end
    end

    function mcdiff!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        params::SimulationParameters,
        current_energy::Float64,
        current_temperature::Float64
    )::Tuple{Int, Float64}
        N = length(chain)
        i = rand(1:length(chain))
        # 保存旧坐标（链和自由粒子）
        # 1. 【记录】只保存当前被移动的单体的旧坐标
        # Particle3D 是 immutable struct，赋值即为值拷贝，非常快
        old_monomer_pos = chain[i]
        
        # 2. 【记录】查找并保存受影响的自由粒子的旧坐标
        # 使用轻量级的 Vector{Tuple} 来存储，避免 Any 类型
        # 大多数情况下 moved_beads 的长度为 0 或 1，开销极小
        moved_beads_backup = Vector{Tuple{Int, Particle3D}}()
        
        NB = length(free_beads)
        if NB > 0
            # 这里的逻辑保留你原代码的意图：
            # 如果自由粒子 "挂" 在当前移动的单体 i 上 (contact[k,1] == i)，则随之移动
            for k in 1:NB
                if params.fbead_contact[k, 1] == i
                    push!(moved_beads_backup, (k, free_beads[k]))
                end
            end
        end
        
        dx = params.DD * (2rand() - 1)
        dy = params.DD * (2rand() - 1)
        dz = params.DD * (2rand() - 1)
        
        chain[i] = Particle3D(old_monomer_pos.x + dx, old_monomer_pos.y + dy, old_monomer_pos.z + dz)

        # 移动关联的自由粒子并记录
        for (k, old_bead) in moved_beads_backup
            free_beads[k] = Particle3D(
                old_bead.x + dx,
                old_bead.y + dy,
                old_bead.z + dz
            )
        end
        
        new_energy = compute_total_energy!(chain, free_beads, params)
        ΔE = new_energy - current_energy
        
        if metropolis_accept(ΔE, current_temperature)
            return (1, new_energy)
        else
            # 7. 【回滚】拒绝：恢复旧坐标
            chain[i] = old_monomer_pos
            
            for (k, old_bead) in moved_beads_backup
                free_beads[k] = old_bead
            end

            return (0, current_energy)
        end
    end

    # function mcpivot!(
    #     chain::Vector{Particle3D},
    #     free_beads::Vector{Particle3D},
    #     params::SimulationParameters,
    #     current_energy::Float64,
    #     current_temperature::Float64
    # )::Tuple{Int, Float64}
    #     # 随机选择一个枢轴点
    #     # current_energy = compute_total_energy!(chain, free_beads, params)
    #     pivot = rand(1:length(chain)-1)  # 枢轴点不能是链的最后一个单体
    #     N = length(chain)

    #     # 随机生成旋转角度和轴
    #     θ = rand() * π  # 旋转角度 [0, π]
    #     axis = normalize([randn(), randn(), randn()])  # 随机旋转轴（归一化）
        
    #     # 创建旋转矩阵
    #     rotation_matrix = compute_rotation_matrix(axis, θ)
        
    #     # 保存旧坐标（链和自由粒子）
    #     old_chain = deepcopy(chain)
    #     old_free_beads = deepcopy(free_beads)
        
    #     # 获取枢轴点的坐标
    #     pivot_coords = [chain[pivot].x, chain[pivot].y, chain[pivot].z]
    #     N_max_c = maximum(params.tf_connectivities)
    #     # 旋转移动：仅旋转枢轴点之后的部分
    #     for i in (pivot+1):length(chain)
    #         # 将单体相对于枢轴点平移到原点
    #         relative_coords = [
    #             chain[i].x - pivot_coords[1],
    #             chain[i].y - pivot_coords[2],
    #             chain[i].z - pivot_coords[3]
    #         ]
            
    #         # 应用旋转矩阵
    #         rotated_coords = rotation_matrix * relative_coords
            
    #         # 平移回原始位置
    #         chain[i] = Particle3D(
    #             rotated_coords[1] + pivot_coords[1],
    #             rotated_coords[2] + pivot_coords[2],
    #             rotated_coords[3] + pivot_coords[3]
    #         )
    #     end
        
    #     # 同时旋转与受影响单体相关的自由粒子
    #     for k in 1:length(free_beads)
    #         bead = free_beads[k]
            
    #         # 检查该自由粒子是否与枢轴点之后的单体相关
    #         for idx in 1:N_max_c
    #             monomer_idx = params.fbead_contact[k, idx]
    #             if monomer_idx > pivot && monomer_idx != -1
    #                 # 将自由粒子相对于枢轴点平移到原点
    #                 relative_coords = [
    #                     bead.x - pivot_coords[1],
    #                     bead.y - pivot_coords[2],
    #                     bead.z - pivot_coords[3]
    #                 ]
                    
    #                 # 应用旋转矩阵
    #                 rotated_coords = rotation_matrix * relative_coords
                    
    #                 # 平移回原始位置
    #                 free_beads[k] = Particle3D(
    #                     rotated_coords[1] + pivot_coords[1],
    #                     rotated_coords[2] + pivot_coords[2],
    #                     rotated_coords[3] + pivot_coords[3]
    #                 )
                    
    #                 # 每个自由粒子只需旋转一次
    #                 break
    #             end
    #         end
    #     end
        
    #     # 计算能量变化（仅计算受影响的能量项）
    #     new_energy = compute_total_energy!(chain, free_beads, params)
    #     ΔE = new_energy - current_energy
        
    #     # Metropolis 判据
    #     if metropolis_accept(ΔE, current_temperature)
    #         return (1, new_energy)  # 接受移动
    #     else
    #         chain[:] = old_chain
    #         free_beads[:] = old_free_beads
    #         return (0, current_energy)  # 拒绝移动
    #     end
    # end

function mcpivot!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        params::SimulationParameters,
        current_energy::Float64,
        current_temperature::Float64
    )::Tuple{Int, Float64}
        
        N = length(chain)
        pivot = rand(1:N-1) 
        
        axis = normalize(SVector(randn(), randn(), randn()))
        θ = rand() * π
        
        # 2. 【优化】计算旋转矩阵 (返回 SMatrix)
        Rot = compute_rotation_matrix(axis, θ)
        
        old_chain_tail = chain[pivot+1:end] # 这里会有一次 copy，但比 loop 里每次 copy 好
        
        # 找出受影响的 beads
        affected_bead_indices = Int[]
        old_beads_backup = Particle3D[]
        N_max_c = maximum(params.tf_connectivities)
        
        for k in 1:length(free_beads)
            is_affected = false
            for idx in 1:N_max_c
                monomer_idx = params.fbead_contact[k, idx]
                if monomer_idx > pivot && monomer_idx != -1
                    is_affected = true
                    break
                end
            end
            if is_affected
                push!(affected_bead_indices, k)
                push!(old_beads_backup, free_beads[k])
            end
        end

        # --- 开始旋转 (无内存分配循环) ---
        p_piv = chain[pivot]
        pivot_pos = SVector(p_piv.x, p_piv.y, p_piv.z)

        # 旋转链
        @inbounds for i in (pivot+1):N
            p = chain[i]
            pos = SVector(p.x, p.y, p.z)
            # 核心优化：StaticArrays 矩阵乘法，无堆内存分配
            new_pos = pivot_pos + Rot * (pos - pivot_pos)
            chain[i] = Particle3D(new_pos[1], new_pos[2], new_pos[3])
        end
        
        # 旋转 Beads
        @inbounds for k in affected_bead_indices
            b = free_beads[k]
            pos = SVector(b.x, b.y, b.z)
            new_pos = pivot_pos + Rot * (pos - pivot_pos)
            free_beads[k] = Particle3D(new_pos[1], new_pos[2], new_pos[3])
        end
        
        # 计算能量
        new_energy = compute_total_energy!(chain, free_beads, params)
        ΔE = new_energy - current_energy
        
        if metropolis_accept(ΔE, current_temperature)
            return (1, new_energy)
        else
            # 回滚
            chain[pivot+1:end] = old_chain_tail
            for (i, k) in enumerate(affected_bead_indices)
                free_beads[k] = old_beads_backup[i]
            end
            return (0, current_energy)
        end
    end

    function mcdiff_free_bead!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        params::SimulationParameters,
        current_energy::Float64,
        current_temperature::Float64
    )::Tuple{Int, Float64}
        # 随机选择一个自由粒子
        if length(free_beads) == 0
            return (0, current_energy)
        end
        # current_energy = compute_total_energy!(chain, free_beads, params)
        k = rand(1:length(free_beads))
        
        # 保存旧坐标
        old_x, old_y, old_z = free_beads[k].x, free_beads[k].y, free_beads[k].z
        
        # 根据温度决定移动模式的概率
        mode = rand() < 0.5 ? 1 : 2
        
        if mode == 1

            # 模式 1: 随机选择两个不相邻的链单体索引
            N = length(chain)
            i, j = 0, 0
            
            # 确保选中的索引不相邻
            while true
                i = rand(1:N)
                j = rand(1:N)
                if abs(i - j) >= 2  # 确保两个单体不相邻
                    break
                end
            end
            # 确保索引有效

            mid_x = (chain[i].x + chain[j].x) / 2
            mid_y = (chain[i].y + chain[j].y) / 2
            mid_z = (chain[i].z + chain[j].z) / 2
            
            # 更新自由粒子位置
            free_beads[k] = Particle3D(mid_x, mid_y, mid_z)


        end
        
        if mode == 2
            # 模式 2: 自由随机移动，基于链质心的PBC
            dx = params.DD * (2rand() - 1)
            dy = params.DD * (2rand() - 1)
            dz = params.DD * (2rand() - 1)
            
            # 计算链的质心
            cm_x = sum(p.x for p in chain) / length(chain)
            cm_y = sum(p.y for p in chain) / length(chain)
            cm_z = sum(p.z for p in chain) / length(chain)
            
            # 更新坐标
            new_x = free_beads[k].x + dx
            new_y = free_beads[k].y + dy
            new_z = free_beads[k].z + dz
            
            # 应用以质心为中心的PBC
            L_box = params.box_size
            function apply_pbc(coord, cm)
                adjusted = coord - cm
                adjusted = mod(adjusted + L_box/2, L_box) - L_box/2
                return adjusted + cm
            end
            
            new_x = apply_pbc(new_x, cm_x)
            new_y = apply_pbc(new_y, cm_y)
            new_z = apply_pbc(new_z, cm_z)
            
            # 更新自由粒子位置
            free_beads[k] = Particle3D(new_x, new_y, new_z)
        end
        
        if !is_position_valid(chain, free_beads, params.min_distance)
            # 恢复旧坐标
            free_beads[k] = Particle3D(old_x, old_y, old_z)
            return (0, current_energy)  # 拒绝移动
        end

        # 计算能量变化（仅计算受影响的能量项）
        new_energy = compute_total_energy!(chain, free_beads, params)
        ΔE = new_energy - current_energy
        
        # Metropolis 判据
        if metropolis_accept(ΔE, current_temperature)
            return (1, new_energy)  # 接受移动
        else
            # 恢复旧坐标
            free_beads[k] = Particle3D(old_x, old_y, old_z)
            return (0, current_energy)  # 拒绝移动
        end
    end

    function freebead_snake!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        params::SimulationParameters,
        current_energy::Float64,
        current_temperature::Float64
    )::Tuple{Int, Float64}
        if length(free_beads) == 0
            return (0, current_energy)
        end
        # current_energy = compute_total_energy!(chain, free_beads, params)
        # 随机选择一个自由粒子
        k = rand(1:length(free_beads))

        # 获取与该自由粒子相关的链单体索引
        monomer_indices = params.fbead_contact[k, :]

        i, j = monomer_indices[1], monomer_indices[2]

        
        # 确保索引有效
        if i == -1 || j == -1
            return (0, current_energy)  # 如果无效，直接返回拒绝
        end
        
        # 随机选择移动方向：+1 表示沿链正方向，-1 表示沿链负方向
        direction = rand([-1, 1])
        
        # 计算目标位置
        target_monomer = direction == 1 ? j : i
        next_monomer = target_monomer + direction
        
        # 检查边界条件
        if next_monomer < 1 || next_monomer > length(chain)
            return (0, current_energy)  # 如果超出链范围，直接返回拒绝
        end
        
        # 计算新位置（沿链移动一个键的长度）
        old_x, old_y, old_z = free_beads[k].x, free_beads[k].y, free_beads[k].z

        # 目标键的方向向量（从当前单体指向下一个单体）
        dx = chain[next_monomer].x - chain[target_monomer].x
        dy = chain[next_monomer].y - chain[target_monomer].y
        dz = chain[next_monomer].z - chain[target_monomer].z
        
        
        # 新位置：沿着键方向移动一个键长
        new_x = old_x + dx 
        new_y = old_y + dy 
        new_z = old_z + dz 
        
        # 更新自由粒子位置
        free_beads[k] = Particle3D(new_x, new_y, new_z)
        
        if !is_position_valid(chain, free_beads, params.min_distance)
            # 如果新位置无效，恢复旧坐标并返回拒绝
            free_beads[k] = Particle3D(old_x, old_y, old_z)
            return (0, current_energy)
        end

        # 计算能量变化（仅计算受影响的能量项）
        new_energy = compute_total_energy!(chain, free_beads, params)
        ΔE = new_energy - current_energy
        
        # Metropolis 判据
        if metropolis_accept(ΔE, current_temperature)
            return (1, new_energy)  # 接受移动
        else
            # 恢复旧坐标
            free_beads[k] = Particle3D(old_x, old_y, old_z)
            return (0, current_energy)  # 拒绝移动
        end
    end


    function move_to_origin!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D}
    )
        # 移动到polymer质心
        all_particles = chain

        # 计算系统几何中心
        cm_x = sum(p.x for p in all_particles) / length(all_particles)
        cm_y = sum(p.y for p in all_particles) / length(all_particles)
        cm_z = sum(p.z for p in all_particles) / length(all_particles)

        # 原地更新chain坐标
        for i in eachindex(chain)
            p = chain[i]
            chain[i] = Particle3D(p.x - cm_x, p.y - cm_y, p.z - cm_z)
        end

        # 原地更新free_beads坐标
        for i in eachindex(free_beads)
            p = free_beads[i]
            free_beads[i] = Particle3D(p.x - cm_x, p.y - cm_y, p.z - cm_z)
        end
    end


    function write_pdb(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        filename::String
    )
        # 打开文件以写入模式
        open(filename, "w") do file
            # 写入链中的单体坐标
            for (i, particle) in enumerate(chain)
                # PDB 格式：ATOM  序号  名称  残基名  链ID  坐标
                println(
                    file,
                    @sprintf(
                        "ATOM  %5d  C   POLY %4d    %8.3f%8.3f%8.3f  1.00  0.00",
                        i, i, particle.x, particle.y, particle.z
                    )
                )
            end
            
            # 写入自由粒子的坐标
            for (j, bead) in enumerate(free_beads)
                # 使用不同的名称（例如 "B"）区分自由粒子
                println(
                    file,
                    @sprintf(
                        "ATOM  %5d  B   FREE %4d    %8.3f%8.3f%8.3f  1.00  0.00",
                        length(chain) + j, length(chain) + j, bead.x, bead.y, bead.z
                    )
                )
            end
            
            # 结束标记
            println(file, "END")
        end
        
        println("Coordinates written to $filename")
    end
    function write_pdb_multiframe(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        filename::String,
        frame::Int,
        params::SimulationParameters
    )
        # 如果是第一帧，打开文件以写入模式；否则以追加模式打开
        mode = frame == 1 ? "w" : "a"
        open(filename, mode) do file
            if frame == 1
                println(file, "REMARK Multi-frame PDB file generated by simulation")
            end

            println(file, "MODEL     $frame")

            # 写入链中的单体坐标，使用与您提供的“正确”版本完全相同的格式
            for (i, particle) in enumerate(chain)
                println(
                    file,
                    @sprintf(
                        "ATOM  %5d  C   POLY %4d    %8.3f%8.3f%8.3f  1.00  0.00",
                        i, i, particle.x, particle.y, particle.z
                    )
                )
            end

            # --- [核心修改] 写入可以区分类型的自由粒子，但严格遵守旧格式 ---
            
            # 从params中获取TF粒子的配置信息
            count1, _ = params.tf_counts
            conn1, conn2 = params.tf_connectivities

            # 遍历所有自由粒子
            for (j, bead) in enumerate(free_beads)
                local res_name::String

                # 根据粒子索引判断其类型，并设置残基名
                if j <= count1
                    # 这是第一种粒子 (CN2)
                    res_name = @sprintf("TF%d", conn1) # "TF2"
                else
                    # 这是第二种粒子 (CN6)
                    res_name = @sprintf("TF%d", conn2) # "TF6"
                end

                # 写入PDB行，使用与您提供的“正确”版本完全相同的格式结构
                # 唯一的区别是用动态的 res_name 替换了静态的 "FREE"
                # 使用 %-4s 确保残基名（如"TF2"）右侧用空格填充，总共占4个字符
                println(
                    file,
                    @sprintf(
                        "ATOM  %5d  B   %-4s %4d    %8.3f%8.3f%8.3f  1.00  0.00",
                        length(chain) + j, res_name, j, bead.x, bead.y, bead.z
                    )
                )
            end

            println(file, "ENDMDL")
        end
    end

    function mcloop_extrusion!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        params::SimulationParameters,
        current_energy::Float64,
        current_temperature::Float64
    )::Tuple{Int, Float64}
        # current_energy = compute_total_energy!(chain, free_beads, params)
        # Step 0: 选择free_bead并获取关联的a和b
        if length(free_beads) == 0
            return (0, current_energy)
        end
        k = 1
        for _ in 1:10
            k = rand(1:length(free_beads))
            if params.fbead_contact[k,1] == -1
                break
            end
        end

            
        # Step 1: 根据alpha矩阵选择a,b并找到对应的k
        # 收集所有可能的有效(a,b)对
        negative_pairs = Tuple{Int,Int}[]
        for i in 1:size(params.alpha,1)
            for j in 1:size(params.alpha,2)
                # 只考虑i < j避免重复，且alpha值为负,只考虑较远的
                if i+4 < j && params.alpha[i,j] < 0
                    push!(negative_pairs, (i,j))
                end
            end
        end

        # 没有有效对则拒绝移动
        isempty(negative_pairs) && return (0, current_energy)

        # 随机选择一个a,b对
        a, b = rand(negative_pairs)

        # 验证有效性
        (a == b || a ∉ 1:length(chain) || b ∉ 1:length(chain)) && return (0, current_energy)
        
        for line in params.fbead_contact
            if a in line && b in line
                return (0, current_energy)
            end
        end

        # 保存原始状态
        old_chain = deepcopy(chain)
        old_free_beads = deepcopy(free_beads)
        original_params = deepcopy(params)  # 保存原始参数
        old_energy = current_energy
        try
            # Step 2: 添加临时强束缚势
            params.loop_anchor = (a, b)  # 标记激活状态

            current_energy = compute_total_energy!(chain, free_beads, params) # 将当前能量带loop势能重新计算
            # Step 3: 进行结构调整
            accepted_moves = 0
            for _ in 1:params.loop_relax_steps

                if distance(chain[a], chain[b]) < params.loop_cutoff
                    break  # 如果a,b之间距离小于1.0则退出
                end
                
                # print(distance(chain[a], chain[b]))
                # 随机选择移动方式
                move_type = rand()
                if move_type < 0.3  # 30%概率使用pivot
                    acc, current_energy = mcpivot!(chain, free_beads, params, current_energy, 
                                    1.0)
                    accepted_moves += acc
                else  # 70%概率使用扩散移动
                    acc, current_energy = mcdiff!(chain, free_beads, params, current_energy,
                                    1.0)
                    accepted_moves += acc
                end
            end
            
            if distance(chain[a], chain[b]) > params.loop_cutoff
                throw(RejectMove())  # 进入回滚流程
            end

            # 将k移动到a和b的中间
            free_beads[k] = Particle3D(
                (chain[a].x + chain[b].x) / 2.0,
                (chain[a].y + chain[b].y) / 2.0,
                (chain[a].z + chain[b].z) / 2.0
            )
            
            # Step 4: 计算最终能量
            # 移除临时势后重新计算真实能量
            params.loop_anchor = nothing
            new_energy = compute_total_energy!(chain, free_beads, params)
            ΔE = new_energy - old_energy
            
            # Metropolis准则（考虑临时势的影响）
            if metropolis_accept(ΔE, current_temperature)

                return (1, new_energy)
            else
                throw(RejectMove())  # 进入回滚流程
            end
            
        catch e
            # 回滚所有修改
            isa(e, RejectMove) || rethrow(e)
            chain[:] = old_chain
            free_beads[:] = old_free_beads
            params.loop_anchor = nothing
            return (0, old_energy)
        end
    end


    # function mcdoublepivot!(
    #     chain::Vector{Particle3D},
    #     free_beads::Vector{Particle3D},
    #     params::SimulationParameters,
    #     current_energy::Float64,
    #     current_temperature::Float64
    # )::Tuple{Int, Float64}
    #     # current_energy = compute_total_energy!(chain, free_beads, params)
    #     # Step 1: 随机选择两个有效枢纽点
    #     a, b = 0, 0
    #     valid = false
    #     N = length(chain)
    #     for _ in 1:100  # 防止无限循环
    #         a = rand(1:N-2)
    #         b = rand(a+2:N)  # 确保至少有一个中间单体
    #         if b <= N && (b - a) >= 2
    #             valid = true
    #             break
    #         end
    #     end
    #     !valid && return (0, current_energy)

    #     # Step 2: 计算旋转轴（基于a到b的向量）
    #     vec_ab = [chain[b].x - chain[a].x,
    #             chain[b].y - chain[a].y,
    #             chain[b].z - chain[a].z]
    #     if norm(vec_ab) < 1e-8  # 防止零向量
    #         return (0, current_energy)
    #     end
    #     axis = normalize(vec_ab)
    #     θ = 2π * rand()  # 完整旋转范围[0, 2π]

    #     # 构建旋转矩阵（使用Rodrigues公式）
    #     rotation_matrix = compute_rotation_matrix(axis, θ)

    #     # 保存旧状态
    #     old_chain = deepcopy(chain)
    #     old_free_beads = deepcopy(free_beads)
    #     a_coords = [chain[a].x, chain[a].y, chain[a].z]

    #     # Step 3: 旋转中间区域
    #     for i in (a+1):(b-1)
    #         rel_pos = [chain[i].x - a_coords[1],
    #                 chain[i].y - a_coords[2],
    #                 chain[i].z - a_coords[3]]
    #         rot_pos = rotation_matrix * rel_pos
    #         chain[i] = Particle3D(
    #             rot_pos[1] + a_coords[1],
    #             rot_pos[2] + a_coords[2],
    #             rot_pos[3] + a_coords[3]
    #         )
    #     end

    #     # Step 4: 更新关联的free beads
    #     for fb in 1:length(free_beads)
    #         for j in 1:1
    #             m_idx = params.fbead_contact[fb, j]
    #             if m_idx in (a+1):(b-1)
    #                 rel_pos = [free_beads[fb].x - a_coords[1],
    #                         free_beads[fb].y - a_coords[2],
    #                         free_beads[fb].z - a_coords[3]]
    #                 rot_pos = rotation_matrix * rel_pos
    #                 free_beads[fb] = Particle3D(
    #                     rot_pos[1] + a_coords[1],
    #                     rot_pos[2] + a_coords[2],
    #                     rot_pos[3] + a_coords[3]
    #                 )
    #                 break
    #             end
    #         end
    #     end

    #     # Step 5: 能量计算和判据
    #     new_energy = compute_total_energy!(chain, free_beads, params)
    #     ΔE = new_energy - current_energy

    #     if metropolis_accept(ΔE, current_temperature)
    #         return (1, new_energy)
    #     else
    #         chain[:] = old_chain
    #         free_beads[:] = old_free_beads
    #         return (0, current_energy)
    #     end
    # end
function mcdoublepivot!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        params::SimulationParameters,
        current_energy::Float64,
        current_temperature::Float64
    )::Tuple{Int, Float64}

        N = length(chain)
        # Step 1: 选择点 (保持不变)
        a, b = 0, 0
        valid = false
        for _ in 1:100
            a = rand(1:N-2)
            b = rand(a+2:N)
            if b <= N && (b - a) >= 2
                valid = true
                break
            end
        end
        !valid && return (0, current_energy)

        # Step 2: 计算轴 (使用 SVector)
        p_a = chain[a]
        p_b = chain[b]
        vec_ab = SVector(p_b.x - p_a.x, p_b.y - p_a.y, p_b.z - p_a.z)
        
        if norm(vec_ab) < 1e-8
            return (0, current_energy)
        end
        
        axis = normalize(vec_ab)
        θ = 2π * rand()

        Rot = compute_rotation_matrix(axis, θ)

        # 备份
        # Double pivot 只影响 a+1 到 b-1 之间的部分，通常很短
        # 所以这里由 copy 产生的开销很小
        range_idx = (a+1):(b-1)
        old_segment = chain[range_idx]
        
        affected_bead_indices = Int[]
        old_beads_backup = Particle3D[]
        # 注意：这里简单假设只要连在区间内就动。逻辑需与你原代码一致。
        for k in 1:length(free_beads)
            # 这里简化逻辑：只要有一个接触点在区间内，就受影响
            # 你原来的逻辑是 `for j in 1:1 ... break`，看起来只检查第一个接触点？
            # 我这里保留严谨性：检查所有接触点
            is_affected = false
            for j in 1:size(params.fbead_contact, 2)
                m_idx = params.fbead_contact[k, j]
                if m_idx in range_idx
                    is_affected = true
                    break
                end
            end
            if is_affected
                push!(affected_bead_indices, k)
                push!(old_beads_backup, free_beads[k])
            end
        end

        # Step 3: 旋转 (SMatrix, 零分配)
        origin = SVector(p_a.x, p_a.y, p_a.z)
        
        @inbounds for i in range_idx
            p = chain[i]
            pos = SVector(p.x, p.y, p.z)
            new_pos = origin + Rot * (pos - origin)
            chain[i] = Particle3D(new_pos[1], new_pos[2], new_pos[3])
        end

        @inbounds for k in affected_bead_indices
            b = free_beads[k]
            pos = SVector(b.x, b.y, b.z)
            new_pos = origin + Rot * (pos - origin)
            free_beads[k] = Particle3D(new_pos[1], new_pos[2], new_pos[3])
        end

        # Step 5: 能量
        new_energy = compute_total_energy!(chain, free_beads, params)
        ΔE = new_energy - current_energy

        if metropolis_accept(ΔE, current_temperature)
            return (1, new_energy)
        else
            chain[range_idx] = old_segment
            for (i, k) in enumerate(affected_bead_indices)
                free_beads[k] = old_beads_backup[i]
            end
            return (0, current_energy)
        end
    end


    # # Rodrigues旋转矩阵计算公式
    # function compute_rotation_matrix(axis::Vector{Float64}, θ::Float64)
    #     u = normalize(axis)
    #     ux, uy, uz = u
    #     cosθ = cos(θ)
    #     sinθ = sin(θ)
        
    #     [cosθ + ux^2*(1-cosθ)      ux*uy*(1-cosθ) - uz*sinθ   ux*uz*(1-cosθ) + uy*sinθ;
    #     uy*ux*(1-cosθ) + uz*sinθ  cosθ + uy^2*(1-cosθ)       uy*uz*(1-cosθ) - ux*sinθ;
    #     uz*ux*(1-cosθ) - uy*sinθ  uz*uy*(1-cosθ) + ux*sinθ   cosθ + uz^2*(1-cosθ)]
    # end

    # Rodrigues旋转矩阵计算公式 (高性能版)
    function compute_rotation_matrix(axis::SVector{3, Float64}, θ::Float64)
        # axis 已经是归一化的 SVector
        ux, uy, uz = axis
        cosθ = cos(θ)
        sinθ = sin(θ)
        c1 = 1 - cosθ

        # 使用 @SMatrix 宏构建静态矩阵 (零分配)
        return @SMatrix [
            cosθ + ux^2*c1        ux*uy*c1 - uz*sinθ    ux*uz*c1 + uy*sinθ;
            uy*ux*c1 + uz*sinθ    cosθ + uy^2*c1        uy*uz*c1 - ux*sinθ;
            uz*ux*c1 - uy*sinθ    uz*uy*c1 + ux*sinθ    cosθ + uz^2*c1
        ]
    end

    # ==============================================================================
    # 优化性能
    # ==============================================================================

    # 计算一个 Bead 的所有相关数据：Contacts 和 Matrix Contributions
    function analyze_bead!(
        k::Int, 
        bead_pos::Particle3D, 
        chain::Vector{Particle3D}, 
        params::SimulationParameters,
        # 缓存容器
        buf_pairs::Vector{Tuple{Int, Int, Float64}}, 
        buf_contacts::Vector{Int}
    )
        get_bead_interactions!(k, bead_pos, chain, params, buf_pairs, buf_contacts)
    end

    # 单体移动的高性能版
    function mcdiff_fast!(
        chain::Vector{Particle3D},
        free_beads::Vector{Particle3D},
        params::SimulationParameters,
        current_total_E::Float64,
        T::Float64
    )::Tuple{Int, Float64} # 返回 (Accepted, NewTotalEnergy)
        
        N = length(chain)
        i = rand(1:N)
        old_pos = chain[i]
        
        # 1. 试探移动 (生成新坐标，但暂时不更新 chain)
        dx = params.DD * (2rand() - 1)
        dy = params.DD * (2rand() - 1)
        dz = params.DD * (2rand() - 1)
        new_pos = Particle3D(old_pos.x + dx, old_pos.y + dy, old_pos.z + dz)
        
        # 2. 识别所有受影响的 Beads
        # 只要 Beads 的旧位置或新位置在截断范围内，就需要更新
        affected_indices = Int[]
        NB = length(free_beads)
        Pcut_sq = params.Pcutoff_ik^2
        
        for k in 1:NB
            b = free_beads[k]
            dist_old_sq = (b.x - old_pos.x)^2 + (b.y - old_pos.y)^2 + (b.z - old_pos.z)^2
            dist_new_sq = (b.x - new_pos.x)^2 + (b.y - new_pos.y)^2 + (b.z - new_pos.z)^2
            
            should_update = false
            # 几何判断
            if dist_old_sq < Pcut_sq || dist_new_sq < Pcut_sq
                should_update = true
            else
                # 拓扑判断：如果原来就连接在这个单体上
                # 注意：params.fbead_contact 的列数取决于最大连接数
                for c_idx in view(params.fbead_contact, k, :)
                    if c_idx == i
                        should_update = true
                        break
                    end
                end
            end
            
            if should_update
                push!(affected_indices, k)
            end
        end
        
        # 3. 准备缓存 (在循环外分配，极重要！)
        # 假设最大连接数为 6，Pair 数 C(6,2)=15
        _buf_pairs = Vector{Tuple{Int, Int, Float64}}(); sizehint!(_buf_pairs, 15)
        _buf_contacts = zeros(Int, 6)
        
        # 保存旧贡献 (用于回滚和计算 Delta)
        # old_contribs[j] 对应 affected_indices[j] 的旧 Pair 数据
        old_contribs = Vector{Vector{Tuple{Int, Int, Float64}}}(undef, length(affected_indices))
        
        # 4. 计算旧的 Specific Energy 贡献 (Chain 仍为 Old)
        for (idx, k) in enumerate(affected_indices)
            get_bead_interactions!(k, free_beads[k], chain, params, _buf_pairs, _buf_contacts)
            old_contribs[idx] = copy(_buf_pairs) # 必须 Copy
        end
        
        # 5. 计算 Standard Energy Delta (Old)
        E_std_old = compute_local_standard_energy(i, old_pos, chain, params)
        
        # === 状态变更 ===
        chain[i] = new_pos
        # ================
        
        # 6. 计算新的 Specific Energy 贡献并累积 Delta
        # 我们需要保存新贡献以便在接受时不做处理，或在拒绝时移除
        new_contribs = Vector{Vector{Tuple{Int, Int, Float64}}}(undef, length(affected_indices))
        new_contacts_list = Vector{Vector{Int}}(undef, length(affected_indices))
        
        M = params.Pij_mediated_matrix
        Alpha = params.alpha
        delta_spec = 0.0
        
        for (idx, k) in enumerate(affected_indices)
            get_bead_interactions!(k, free_beads[k], chain, params, _buf_pairs, _buf_contacts)
            new_contribs[idx] = copy(_buf_pairs)
            new_contacts_list[idx] = copy(_buf_contacts) # 保存新连接关系，用于更新 fbead_contact
            
            # --- 增量更新 Matrix 和 Energy ---
            
            # A. 移除旧贡献
            for (u, v, term) in old_contribs[idx]
                old_M = M[u, v]
                M[u, v] -= term
                new_M = M[u, v]
                # E = alpha * (1 - exp(M))
                # Delta = E_new - E_old = alpha * (exp(M_old) - exp(M_new))
                delta_spec += Alpha[u, v] * (exp(old_M) - exp(new_M))
            end
            
            # B. 添加新贡献
            for (u, v, term) in new_contribs[idx]
                old_M = M[u, v]
                M[u, v] += term
                new_M = M[u, v]
                delta_spec += Alpha[u, v] * (exp(old_M) - exp(new_M))
            end
        end
        
        # 7. 计算 Standard Energy Delta (New)
        E_std_new = compute_local_standard_energy(i, new_pos, chain, params)
        
        # 8. 总能量判定
        delta_total = (E_std_new - E_std_old) + delta_spec
        
        if metropolis_accept(delta_total, T)
            # --- 接受 ---
            
            # Matrix M 已经被更新到位了，无需额外操作。
            
            # 更新 fbead_contact 表
            for (idx, k) in enumerate(affected_indices)
                contacts = new_contacts_list[idx]
                # 能够安全地填入，因为 _buf_contacts 大小固定
                # 注意：如果 contacts 中含有 -1，也要填入
                for ci in eachindex(contacts)
                    if ci <= size(params.fbead_contact, 2)
                        params.fbead_contact[k, ci] = contacts[ci]
                    end
                end
            end
            
            return 1, current_total_E + delta_total
        else
            # --- 拒绝：回滚 ---
            
            # 1. 恢复 Chain
            chain[i] = old_pos
            
            # 2. 回滚 Matrix M
            # 反向操作：移除 New，加回 Old
            for (idx, k) in enumerate(affected_indices)
                # 撤销新贡献
                for (u, v, term) in new_contribs[idx]
                    M[u, v] -= term 
                end
                # 恢复旧贡献
                for (u, v, term) in old_contribs[idx]
                    M[u, v] += term 
                end
            end
            
            return 0, current_total_E
        end
    end

    # ==============================================================================
    # 3. 辅助函数
    # ==============================================================================

    function get_bead_interactions!(
        bead_idx::Int,
        bead_pos::Particle3D,
        chain::Vector{Particle3D},
        params::SimulationParameters,
        # 输出容器
        out_pairs::Vector{Tuple{Int, Int, Float64}}, 
        out_contacts::Vector{Int}
    )
        empty!(out_pairs)
        
        # 1. 确定 K
        count1, count2 = params.tf_counts
        conn1, conn2 = params.tf_connectivities
        K = (bead_idx <= count1) ? conn1 : conn2

        # 2. 分发到专用内核
        if K == 2
            _kernel_N2!(bead_pos, chain, params, out_pairs, out_contacts)
        elseif K == 3
            _kernel_N3!(bead_pos, chain, params, out_pairs, out_contacts)
        elseif K == 4
            _kernel_N4!(bead_pos, chain, params, out_pairs, out_contacts)
        elseif K == 5
            _kernel_N5!(bead_pos, chain, params, out_pairs, out_contacts)
        elseif K == 6
            _kernel_N6!(bead_pos, chain, params, out_pairs, out_contacts)
        end
    end

    # --- Kernel for K=2 ---
    function _kernel_N2!(
        pos::Particle3D, chain::Vector{Particle3D}, params::SimulationParameters,
        out_pairs::Vector{Tuple{Int, Int, Float64}}, out_contacts::Vector{Int}
    )
        # 初始化 Tuple: ((r^2, idx), ...)
        best = ( (Inf, -1), (Inf, -1) )
        Pcut_sq = params.Pcutoff_ik^2
        bx, by, bz = pos.x, pos.y, pos.z

        # 1. 极速搜索
        @inbounds for i in 1:length(chain)
            p = chain[i]
            dx, dy, dz = p.x - bx, p.y - by, p.z - bz
            r_sq = dx*dx + dy*dy + dz*dz
            
            # 比较距离平方。注意：best[2] 是当前第 2 大的
            if r_sq < Pcut_sq && r_sq < best[2][1]
                best = insert_sorted_tuple_N2(best, r_sq, i)
            end
        end

        # 2. 填充 Contacts (确保 out_contacts 足够大)
        fill!(out_contacts, -1) 
        out_contacts[1], out_contacts[2] = best[1][2], best[2][2]

        # 3. 计算能量 (只有在找到足够邻居时)
        if best[2][2] != -1
            # 在这里才进行开方运算
            _compute_pairs_generic!(best, 2, params, out_pairs)
        end
    end

    # --- Kernel for K=3 ---
    function _kernel_N3!(
        pos::Particle3D, chain::Vector{Particle3D}, params::SimulationParameters,
        out_pairs::Vector{Tuple{Int, Int, Float64}}, out_contacts::Vector{Int}
    )
        best = ( (Inf, -1), (Inf, -1), (Inf, -1) )
        Pcut_sq = params.Pcutoff_ik^2
        bx, by, bz = pos.x, pos.y, pos.z

        @inbounds for i in 1:length(chain)
            p = chain[i]
            r_sq = (p.x-bx)^2 + (p.y-by)^2 + (p.z-bz)^2
            
            if r_sq < Pcut_sq && r_sq < best[3][1]
                best = insert_sorted_tuple_N3(best, r_sq, i)
            end
        end


        fill!(out_contacts, -1)
        out_contacts[1], out_contacts[2], out_contacts[3] = best[1][2], best[2][2], best[3][2]

        if best[2][2] != -1 # 至少需要2个邻居才能形成Pair
            _compute_pairs_generic!(best, 3, params, out_pairs)
        end
    end

    # --- Kernel for K=4 ---
    function _kernel_N4!(
        pos::Particle3D, chain::Vector{Particle3D}, params::SimulationParameters,
        out_pairs::Vector{Tuple{Int, Int, Float64}}, out_contacts::Vector{Int}
    )
        best = ( (Inf, -1), (Inf, -1), (Inf, -1), (Inf, -1) )
        Pcut_sq = params.Pcutoff_ik^2
        bx, by, bz = pos.x, pos.y, pos.z

        @inbounds for i in 1:length(chain)
            p = chain[i]
            r_sq = (p.x-bx)^2 + (p.y-by)^2 + (p.z-bz)^2
            if r_sq < Pcut_sq && r_sq < best[4][1]
                best = insert_sorted_tuple_N4(best, r_sq, i)
            end
        end

        fill!(out_contacts, -1)
        for k in 1:4; out_contacts[k] = best[k][2]; end

        if best[2][2] != -1
            _compute_pairs_generic!(best, 4, params, out_pairs)
        end
    end

    # --- Kernel for K=5 ---
    function _kernel_N5!(
        pos::Particle3D, chain::Vector{Particle3D}, params::SimulationParameters,
        out_pairs::Vector{Tuple{Int, Int, Float64}}, out_contacts::Vector{Int}
    )
        best = ( (Inf, -1), (Inf, -1), (Inf, -1), (Inf, -1), (Inf, -1) )
        Pcut_sq = params.Pcutoff_ik^2
        bx, by, bz = pos.x, pos.y, pos.z

        @inbounds for i in 1:length(chain)
            p = chain[i]
            r_sq = (p.x-bx)^2 + (p.y-by)^2 + (p.z-bz)^2
            if r_sq < Pcut_sq && r_sq < best[5][1]
                best = insert_sorted_tuple_N5(best, r_sq, i)
            end
        end

        fill!(out_contacts, -1)
        for k in 1:5; out_contacts[k] = best[k][2]; end

        if best[2][2] != -1
            _compute_pairs_generic!(best, 5, params, out_pairs)
        end
    end

    # --- Kernel for K=6 ---
    function _kernel_N6!(
        pos::Particle3D, chain::Vector{Particle3D}, params::SimulationParameters,
        out_pairs::Vector{Tuple{Int, Int, Float64}}, out_contacts::Vector{Int}
    )
        best = ( (Inf, -1), (Inf, -1), (Inf, -1), (Inf, -1), (Inf, -1), (Inf, -1) )
        Pcut_sq = params.Pcutoff_ik^2
        bx, by, bz = pos.x, pos.y, pos.z

        @inbounds for i in 1:length(chain)
            p = chain[i]
            r_sq = (p.x-bx)^2 + (p.y-by)^2 + (p.z-bz)^2
            if r_sq < Pcut_sq && r_sq < best[6][1]
                best = insert_sorted_tuple_N6(best, r_sq, i)
            end
        end

        fill!(out_contacts, -1)
        for k in 1:6; out_contacts[k] = best[k][2]; end

        if best[2][2] != -1
            _compute_pairs_generic!(best, 6, params, out_pairs)
        end
    end
    @inline function _compute_pairs_generic!(
        best_tuple::Tuple, 
        K::Int, 
        params::SimulationParameters, 
        out_pairs::Vector{Tuple{Int, Int, Float64}}
    )
        k_c = params.k_c
        r0 = params.r0

        # 外层循环: 第一个点
        for a in 1:(K-1)
            item_a = best_tuple[a]
            idx_a = item_a[2]
            if idx_a == -1 break end 
            dist_a = sqrt(item_a[1]) # 此时才开方

            # 内层循环: 第二个点
            for b in (a+1):K
                item_b = best_tuple[b]
                idx_b = item_b[2]
                if idx_b == -1 break end
                dist_b = sqrt(item_b[1]) 

                i_pair, j_pair = minmax(idx_a, idx_b)
                
                # 物理限制
                if (j_pair - i_pair) >= 2
                    P_ik = 0.5 * (1.0 - tanh(k_c * (dist_a - r0)))
                    P_jk = 0.5 * (1.0 - tanh(k_c * (dist_b - r0)))
                    
                    term = log(max(eps(Float64), 1.0 - P_ik * P_jk))
                    push!(out_pairs, (i_pair, j_pair, term))
                end
            end
        end
    end

    # ==============================================================================
    # 优化后的局部能量计算函数
    # ==============================================================================

    """
    只计算与单个粒子 index 相关的 Bond, Angle, LJ, Wall 能量。
    不包含 FreeBeads 和 Specific Interaction (因为它们涉及全局排序，需单独处理)。
    """
    function compute_local_standard_energy(
        idx::Int, 
        p_new::Particle3D, 
        chain::Vector{Particle3D}, 
        params::SimulationParameters
    )::Float64
        E = 0.0
        N = length(chain)

        # 1. Bond Energy (涉及 i-1 和 i, 以及 i 和 i+1)
        # 注意：如果 idx=1，没有 i-1；如果 idx=N，没有 i+1
        if idx > 1
            E += bond_pair_energy(chain[idx-1], p_new, params)
        end
        if idx < N
            E += bond_pair_energy(p_new, chain[idx+1], params)
        end

        # 2. Angle Energy (涉及 idx-1, idx, idx+1 三元组，以及以 idx 为端点的角)
        # Case A: idx 是中心 (idx-1, idx, idx+1)
        if idx > 1 && idx < N
            E += angle_triplet_energy(chain[idx-1], p_new, chain[idx+1], params)
        end
        # Case B: idx 是右端点 (idx-2, idx-1, idx)
        if idx > 2
            E += angle_triplet_energy(chain[idx-2], chain[idx-1], p_new, params)
        end
        # Case C: idx 是左端点 (idx, idx+1, idx+2)
        if idx < N - 1
            E += angle_triplet_energy(p_new, chain[idx+1], chain[idx+2], params)
        end

        # 3. Wall Interaction
        E += wall_particle_energy(p_new, params)

        # 4. Chain Non-bonded (LJ) - O(N) instead of O(N^2)
        # 只计算 p_new 与链上其他粒子的相互作用
        E += lj_particle_chain_energy(idx, p_new, chain, params)

        # # 5. Ideal Chromosome (Alpha_0)
        # # 类似 LJ，只计算 p_new 与其他粒子的对
        # if !all(iszero, params.alpha_0)
        #     E += ideal_chrom_particle_energy(idx, p_new, chain, params)
        # end

        return E
    end

    # --- 辅助微观能量函数 (Inline 以提高速度) ---

    @inline function bond_pair_energy(p1::Particle3D, p2::Particle3D, params::SimulationParameters)
        dist_sq = (p1.x-p2.x)^2 + (p1.y-p2.y)^2 + (p1.z-p2.z)^2
        dist = sqrt(dist_sq)
        # Morse Potential
        k, De, r0 = params.k_bond, params.De_bond, params.r0_bond
        a = sqrt(k / (2.0 * De))
        val = 1.0 - exp(-a * (dist - r0))
        return De * val^2
    end

    @inline function angle_triplet_energy(p1::Particle3D, p2::Particle3D, p3::Particle3D, params::SimulationParameters)
        v1x, v1y, v1z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        v2x, v2y, v2z = p3.x - p2.x, p3.y - p2.y, p3.z - p2.z
        
        dot_prod = v1x*v2x + v1y*v2y + v1z*v2z
        norm1 = sqrt(v1x^2 + v1y^2 + v1z^2)
        norm2 = sqrt(v2x^2 + v2y^2 + v2z^2)
        
        denom = norm1 * norm2
        if denom < 1e-12 return 0.0 end
        
        cos_theta = clamp(dot_prod / denom, -1.0, 1.0)
        theta_diff = acos(cos_theta) - params.theta0_angle # theta0 已经是弧度
        return params.k_angle * (1.0 - cos(theta_diff))
    end

    @inline function wall_particle_energy(p::Particle3D, params::SimulationParameters)
        # Rc assumed [0,0,0]
        r_center = sqrt(p.x^2 + p.y^2 + p.z^2)
        Rw = params.R_wall
        r_np = r_center - Rw
        
        if r_np >= 0 # Outside or on boundary
            return 0.5 * 1.0 * r_np^6 # ε=1.0
        end
        
        # Inside
        r_np_abs = abs(r_np)
        σ = params.r0_bond
        cutoff = σ * 1.122462 # 2^(1/6)
        
        if r_np_abs <= cutoff
            inv_r = σ / r_np_abs
            inv_r6 = inv_r^6
            return 4.0 * 1.0 * (inv_r6^2 - inv_r6 + 0.25)
        end
        return 0.0
    end

    @inline function lj_particle_chain_energy(idx::Int, p::Particle3D, chain::Vector{Particle3D}, params::SimulationParameters)
        ε = params.lj_epsilon
        if ε == 0.0 return 0.0 end
        
        E = 0.0
        σ = params.lj_sigma
        cutoff_sq = params.lj_cutoff^2
        lj_range = params.lj_range
        
        # Shift energy constants
        inv_cut2 = 1.0 / cutoff_sq
        inv_cut6 = inv_cut2^3
        shift = 4.0 * ε * (inv_cut6^2 - inv_cut6)
        
        N = length(chain)
        # Loop limits based on lj_range
        start_j = max(1, idx - lj_range)
        end_j = min(N, idx + lj_range)
        
        @inbounds for j in start_j:end_j
            # Skip self and neighbors involved in bonds (usually |i-j| > 1 or 2 depending on physics)
            # Assuming 1-2 and 1-3 exclusion handled by bond/angle, standard LJ often excludes |i-j| <= 2?
            # Your original code used: start_j = i + 2. Let's keep consistency: exclude |i-j| <= 1
            if abs(idx - j) < 2 continue end 

            pj = chain[j]
            dx, dy, dz = p.x - pj.x, p.y - pj.y, p.z - pj.z
            r_sq = dx^2 + dy^2 + dz^2
            
            if r_sq < cutoff_sq
                inv_r2 = 1.0 / r_sq
                inv_r6 = inv_r2^3
                E += 4.0 * ε * (inv_r6^2 - inv_r6) - shift
            end
        end
        return E
    end

    @inline function ideal_chrom_particle_energy(idx::Int, p::Particle3D, chain::Vector{Particle3D}, params::SimulationParameters)
        E = 0.0
        kc, r0 = params.ideal_chrom_kc, params.ideal_chrom_r0
        alpha_0 = params.alpha_0
        
        # 只计算涉及 idx 的对
        # 注意：alpha_0 通常是对称的，或者只定义了上三角。
        # 这里我们扫描所有 j，除了 idx 本身和近邻（如果 ideal model 排除近邻的话，原代码是 j in i+2:N）
        
        N = length(chain)
        @inbounds for j in 1:N
            if abs(idx - j) < 2 continue end # 保持一致性，排除近邻
            
            # 获取 alpha_0 值，处理对称性
            val_alpha = (idx < j) ? alpha_0[idx, j] : alpha_0[j, idx]
            if val_alpha == 0.0 continue end

            pj = chain[j]
            dist = sqrt((p.x-pj.x)^2 + (p.y-pj.y)^2 + (p.z-pj.z)^2)
            Pij = 0.5 * (1.0 - tanh(kc * (dist - r0)))
            E += val_alpha * Pij
        end
        return E
    end

    # 修改后的接触矩阵计算函数
    function calculate_contacts(chains, params, cutoff::Real)
        N = params.N
        
        # 初始化时指定类型
        contacts = Vector{Matrix{Float64}}()
        distance_maps = Vector{Matrix{Float64}}()
        # 预分配大小以提高效率
        sizehint!(contacts, length(chains))
        sizehint!(distance_maps, length(chains))

        for chain in chains
            contact = zeros(Float64, N, N)      # 初始化为浮点数矩阵
            distance_map = zeros(Float64, N, N)
            for i in 1:N
                for j in i:N
                    distance_ij = distance(chain[i], chain[j])
                    
                    local val::Float64 # 声明变量 val

                    # --- 主要修改部分：根据 cutoff 的值选择计算模式 ---
                    if cutoff != 0.0
                        # 模式一：使用 cutoff 进行二元判断
                        # 如果距离 <= cutoff，则接触为 1.0，否则为 0.0
                        val = distance_ij <= cutoff ? 1.0 : 0.0
                    else
                        # 模式二：使用原函数（tanh平滑函数）计算
                        # 仅在此模式下才需要 k_c 和 r_0 参数
                        k_c = params.calculate_contacts_kc
                        r_0 = params.calculate_contacts_r0
                        val = 0.5 * (1 + tanh(k_c * (r_0 - distance_ij)))
                    end
                    # --- 修改结束 ---

                    contact[i, j] = val
                    contact[j, i] = val          # 对称赋值
                    distance_map[i, j] = distance_ij
                    distance_map[j, i] = distance_ij
                end
            end
            push!(contacts, contact) # 将计算结果添加到列表
            push!(distance_maps, distance_map)
        end
        
        # @show typeof(contacts) # 这行可以保留用于调试，或在最终代码中移除
        return contacts, distance_maps
    end



    function save_contact_map(contact_matrix, filename)
        N = size(contact_matrix, 1)
        open(filename
            , "w") do io
            for i in 1:N
                for j in 1:N
                    println(io,i," ",j," ", contact_matrix[i, j])
                end
            end
        end
    end



    # ==============================================================================
    # 3. 模拟流程控制 (Run Core & Parallel)
    # ==============================================================================

    function run_simulation_core(params::SimulationParameters, snapshot_base_dir)
        worker_id = myid()
        pdb_filename = joinpath(snapshot_base_dir, "worker_$(worker_id).pdb")
        energy_filename = joinpath(snapshot_base_dir, "energy_worker_$(worker_id).txt")
        
        # 缓冲区
        output_buffer = String[]
        sizehint!(output_buffer, params.num_samples)

        total_N_B = sum(params.tf_counts)

        # 初始化链和Beads
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
            max_conn = maximum(params.tf_connectivities)
            params.fbead_contact = fill(-1, total_N_B, max_conn)
        end

        current_energy = compute_total_energy!(chain, free_beads, params)
        move_functions = (mcdiff!, mcdiff_free_bead!, mcpivot!, mcdoublepivot!, freebead_snake!)
        
        # move_functions = (mcdiff_fast!, mcdiff_free_bead!, mcpivot!, mcdoublepivot!, freebead_snake!)
        # 移动权重
        move_weights = [100, 100, 20, 20, 5]
        cumulative_weights = cumsum(move_weights)
        total_weight = cumulative_weights[end]

        # 时间步设置
        T = params.MTf
        equilibration_steps = round(Int, params.Steps_FINAL * 0.5)
        sampling_steps = params.Steps_FINAL - equilibration_steps
        sampling_interval = round(Int, max(1, sampling_steps / params.num_samples))

        for step in 1:params.Steps_FINAL
            rand_val = rand(1:total_weight)
            move_idx = searchsortedfirst(cumulative_weights, rand_val)
            _, current_energy = move_functions[move_idx](chain, free_beads, params, current_energy, T)
            
            if step > equilibration_steps && (step - equilibration_steps) % sampling_interval == 0
                push!(output_chains, deepcopy(chain))
                push!(tem_beads, deepcopy(free_beads))
                
                # 计算能量输出
                energy_out = [
                    compute_total_energy!(chain, free_beads, params),
                    compute_bond_energy(chain, params), 
                    compute_bond_angle_energy(chain, params), 
                    compute_chain_nonbond_energy(chain, params), 
                    compute_wall_interaction(chain,params), 
                    compute_free_bead_energy(free_beads, params), 
                    compute_specific_interaction_energy(chain, free_beads, params)
                ]
                mean_coord_num = calculate_mean_coordination_number(chain, params.coord_num_threshold)
                
                output_step = step - equilibration_steps
                energy_str = join([@sprintf("%.4f", e) for e in energy_out], " ")
                push!(output_buffer, @sprintf("%d %d %s %.4f", worker_id, output_step, energy_str, mean_coord_num))
            end
        end
        
        open(energy_filename, "w") do f
            for line in output_buffer; println(f, line); end
        end

        if isempty(output_chains)
            error("No valid chains recorded on worker $(worker_id)")
        end
        for i in 1:length(output_chains)
            write_pdb_multiframe(output_chains[i], tem_beads[i], pdb_filename, i, params)
        end
        ensemble_contacts, ensemble_distances = calculate_modulated_contact_probability(output_chains, tem_beads, params)
        
        return ensemble_contacts, ensemble_distances
    end

    # -------------------------------------------------------------------------
    # 并行模拟调度器 (从 Config 构建参数)
    function run_parallel_simulations(alpha_matrix, alpha_0_matrix, config::Dict; iter_num=0)
        
        # 输出路径处理
        output_dir = config["experiment"]["output_dir"]
        snapshot_base_dir = joinpath(output_dir, "sim_out", "iter_$(iter_num)")
        mkpath(snapshot_base_dir)
        
        # 提取 Simulation 部分配置
        sim_conf = config["simulation"]
        phys_conf = config["simulation"]["physics"]
        
        # 1. 提取 N 和 TF 配置
        N = sim_conf["N"]
        
        # 处理 TF 计数和连接数 (从 Array 转为 Tuple)
        tf_counts = Tuple(sim_conf["tf_counts"])
        tf_connectivities = Tuple(sim_conf["tf_connectivities"])
        
        # 计算 connectivity_map
        if tf_counts[1] == -1
            connectivity_map = Int[]
        else
            connectivity_map = vcat(
                fill(tf_connectivities[1], tf_counts[1]),
                fill(tf_connectivities[2], tf_counts[2])
            )
        end
        
        # 2. 确定并行运行次数
        parallel_runs = config["parallel"]["runs_per_iter"]

        results_per_worker = @distributed (append!) for _ in 1:parallel_runs
            # 3. 动态构建 SimulationParameters
            params = SimulationParameters(
                # 动态变量
                alpha = alpha_matrix,
                alpha_0 = alpha_0_matrix,
                tf_counts = tf_counts,
                tf_connectivities = tf_connectivities,
                tf_connectivity_map = connectivity_map,
                
                # 基础模拟参数
                N = N,
                Steps_FINAL = sim_conf["steps_final"],
                num_samples = sim_conf["num_samples"],
                MTf = sim_conf["temperature"],
                
                # 物理参数 (映射 config -> struct)
                lj_epsilon = phys_conf["lj_epsilon"],
                k_angle = phys_conf["k_angle"],
                theta0_angle = phys_conf["theta0_angle_pai"] * π,

                lj_range = phys_conf["lj_range"],
                r0_bond = phys_conf["r0_bond"],
                k_bond = phys_conf["k_bond"],
                De_bond = phys_conf["De_bond"],
                lj_sigma = phys_conf["lj_sigma"],
                lj_cutoff = phys_conf["lj_cutoff"],
                R_wall = phys_conf["R_wall"],
                
                # Free bead 参数
                r0 = phys_conf["fbead_r0"],
                k_c = phys_conf["fbead_kc"],
                Pcutoff_ik = phys_conf["Pcutoff_ik"],
                free_bead_LJ_ε = phys_conf["free_bead_LJ_epsilon"],
                
                # 接触计算与Loop
                loop_cutoff = phys_conf["loop_cutoff"],
                calculate_contacts_r0 = phys_conf["calc_contact_r0"],
                calculate_contacts_kc = phys_conf["calc_contact_kc"]
            )
            
            [run_simulation_core(params, snapshot_base_dir)]
        end

        # 聚合结果
        all_worker_means = [res[1] for res in results_per_worker]
        mean_contact = mean(all_worker_means)
        
        all_worker_distances = [res[2] for res in results_per_worker]
        mean_distance = mean(all_worker_distances)

        return mean_contact, mean_distance
    end

    # -------------------------------------------------------------------------
    # 优化主循环
    function optimize_alpha_main(initial_alpha, initial_alpha_0, target_contact, config::Dict)

        # 从 Config 提取优化参数
        opt_conf = config["optimization"]
        output_dir = config["experiment"]["output_dir"]
        
        parallel_runs = config["parallel"]["runs_per_iter"]
        maxiter = opt_conf["max_iter"]
        Step_start = opt_conf["step_start"]
        
        correction_exponent = opt_conf["correction_exponent"]
        lambda_alpha_base = opt_conf["lambda_alpha_base"]
        lambda_alpha0_base = opt_conf["lambda_alpha0_base"]
        N = size(initial_alpha, 1)

        # 创建必要的子目录
        mkpath(joinpath(output_dir, "alpha_log"))
        mkpath(joinpath(output_dir, "contacts"))

        if Step_start == 1
            alpha = copy(initial_alpha)
            alpha_0 = copy(initial_alpha_0)
        else
            alpha = read_alpha_from_file(joinpath(output_dir, "alpha_log", "$Step_start.txt"),N)
            alpha_0 = copy(initial_alpha_0)
        end
        k_max = N - 1



        # 预计算距离校正矩阵
        CorrectionMatrix = ones(Float64, N, N)
        if correction_exponent > 0
            for k in 2:k_max
                correction_factor = Float64(k)^correction_exponent
                for i in 1:(N-k)
                    j = i + k
                    CorrectionMatrix[i, j] = correction_factor
                    CorrectionMatrix[j, i] = correction_factor
                end
            end
        end

        for t in Step_start:maxiter
            iter_start_time = time() # 获取高精度时间戳用于计算耗时
            start_dt = Dates.now()   # 获取当前日期时间用于显示
            println("\n--- Iteration: $t --- Start Time: $(Dates.format(start_dt, "yyyy-mm-dd HH:MM:SS"))")

            # 1. 保存当前 alpha
            save_contact_map(alpha, joinpath(output_dir, "alpha_log", "$t.txt"))

            # 2. 运行并行模拟 (传入 config)
            simulated_contact, _ = run_parallel_simulations(
                alpha, alpha_0, config; iter_num=t
            )
            println("Simulations complete.")

            # 3. 计算 Loss
            # ... (与原代码相同，省略部分重复逻辑以保持简洁) ...
            numerator = 0.0; denominator = 0.0
            sim_vals = Float64[]; target_vals = Float64[]
            for i in 1:N, j in (i+2):N
                s_val = simulated_contact[i, j]; t_val = target_contact[i, j]
                numerator += abs(s_val - t_val)
                denominator += t_val
                push!(sim_vals, s_val); push!(target_vals, t_val)
            end
            loss = denominator ≈ 0.0 ? 0.0 : numerator / denominator
            correlation = cor(sim_vals, target_vals)
            
            # 2. 获取当前时间戳 (建议使用下划线连接，保持列的完整性)
            timestamp_str = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
            current_duration = time() - iter_start_time


            println("Loss: $loss, Correlation: $correlation")
            # 3. 写入文件
            open(joinpath(output_dir, "loss.dat"), "a") do io
                # 输出格式: Iteration | Loss | Correlation | Duration(s) | Timestamp
                println(io, t, " ", loss, " ", correlation, " ", round(current_duration, digits=2), " ", timestamp_str)
            end
            save_contact_map(simulated_contact, joinpath(output_dir, "contacts", "$t.txt"))

            # 4. 更新 Alpha
            delta = simulated_contact - target_contact
            λ_alpha = lambda_alpha_base / sqrt(t)
            λ_alpha0 = lambda_alpha0_base / sqrt(t)

            update_term = λ_alpha * (delta .* CorrectionMatrix)
            for i in 1:N, j in (i+1):N
                alpha[i,j] += update_term[i,j]
            end

            # 5. 更新 Alpha_0
            sim_means = [mean(diag(simulated_contact, k)) for k in 1:k_max]
            target_means = [mean(diag(target_contact, k)) for k in 1:k_max]
            
            for k in 1:k_max
                isnan(sim_means[k]) && continue
                delta_k = sim_means[k] - target_means[k]
                current_alpha0_k = alpha_0[1, 1+k]
                eff_lambda = k > 50 ? λ_alpha0 * 1.2 : λ_alpha0
                new_val = current_alpha0_k + eff_lambda * delta_k
                alpha_0[diagind(alpha_0, k)] .= new_val
            end

            open(joinpath(output_dir, "alpha_0_log.txt"), "a") do io
                println(io, t, " ", join(round.(alpha_0[1, 2:end], digits=6), " "))
            end
            
        end
        
        println("Optimization finished.")
        return (best_alpha=alpha, best_alpha_0=alpha_0)
    end

end # module