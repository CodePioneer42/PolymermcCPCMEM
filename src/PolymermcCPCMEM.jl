module PolymermcCPCMEM

using Parameters
using LinearAlgebra
using Random
using TupleTools
using StaticArrays
using Distributed

export SimulationParameters, Particle3D
export optimize_alpha_main, run_parallel_simulations

@with_kw mutable struct SimulationParameters
    N::Int = 65

    # --- [新增] 使用元组定义TF粒子混合种群 ---
    # 例如 tf_counts = (10, 10) 表示两种TF各有10个
    tf_counts::Tuple{Int, Int} = (0, 0) 
    # 例如 tf_connectivities = (2, 3) 表示第一类TF连接数为2，第二类为3
    tf_connectivities::Tuple{Int, Int} = (0, 0) 
    fbead_contact ::Matrix{Int} = Matrix{Int}(undef, 0, 0)
    
    # --- [新增] 用于在模拟中快速查找每个TF粒子的连接数 ---
    # 这个数组将在初始化时被填充，例如 [2, 2, ..., 3, 3, ...]
    tf_connectivity_map::Vector{Int} = Int[]

    Steps_RUN::Int = 4000 # 每一个温度的步数
    MSEP::Int = 1000 # 输出pdb文件的频率

    Steps_FINAL::Int = 1000000

    MTi::Float64 = 50.0
    MTf::Float64 = 1.0
    MdT::Float64 = -1.0
    num_samples::Int = 10

    k_angle::Float64 = 1.0

    Pcutoff_ik::Float64 = 0.8 # Pij 计算的tanh的参数
    k_c::Float64 = 3.22  # Pij 计算的tanh的参数
    r0::Float64 = 1.78 # Pij 计算的tanh的参数 在计算NB的时候会被更改
    factor_effective_cutoff_Pcutoff_ik ::Float64 = 2.1 # TF的排斥直径是 Pcutoff_ik 的x倍
    free_bead_LJ_ε ::Float64 = 0.5 # 自由粒子LJ势能的ε参数
    lj_range::Int = 5 # LJ势的范围
    
    r0_bond::Float64 = 1.6 # 键长
    k_bond::Float64 = 15.0
    De_bond::Float64 = 10.0
    DD::Float64 = 0.20

    # bead之间的LJ 势能ε参数
    lj_epsilon::Float64 = 1.0 
    lj_sigma::Float64 = 0.1
    lj_cutoff::Float64 = 2.5
    
    R_wall::Float64 = 9.7 # 球形壁的半径，需要减去键长才是真实的空间的半径 9.7-1.6
    ideal_chrom_r0::Float64 = 3.0
    ideal_chrom_kc::Float64 = 0.8

    calculate_contacts_r0::Float64 = 1.78
    calculate_contacts_kc::Float64 = 3.22
    coord_num_threshold::Float64 = 1.2 # 定义配位数计算的接触阈值

    min_distance::Float64 = 0.01
    box_size::Float64 = 20.0 # 自由粒子的盒子大小
    alpha::Matrix{Float64} = Matrix{Float64}(undef, 0, 0) 

    alpha_0::Matrix{Float64} = Matrix{Float64}(undef, 0, 0) #使用的是Ps填充的矩阵

    # Loop extrusion参数
    loop_relax_steps::Int = 1000
    loop_strength::Float64 = 1e5
    loop_cutoff::Float64 = 0.3 # loop势的截断距离,loop节点的最小距离
    loop_anchor::Union{Nothing,Tuple{Int,Int}} = nothing

    Pij_mediated_matrix ::Matrix{Float64}  = Matrix{Float64}(undef, 0, 0) 

end



# 定义一个粒子的三维坐标结构体
struct Particle3D
    x::Float64  # x 坐标
    y::Float64  # y 坐标
    z::Float64  # z 坐标
end

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

    for i in 1:length(values_str)
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

@everywhere function initialize_random_chain(N::Int, bond_length::Float64, min_distance::Float64, max_attempts::Int)::Vector{Particle3D}
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

# 检查新位置是否满足最小距离要求
function is_position_valid(chain::Vector{Particle3D}, new_x::Float64, new_y::Float64, new_z::Float64, min_distance::Float64)::Bool
    for particle in chain
        if compute_distance(particle.x, particle.y, particle.z, new_x, new_y, new_z) < min_distance
            return false
        end
    end
    return true
end



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

# 检查新位置是否满足最小距离要求
function is_position_valid(chain::Vector{Particle3D}, free_beads::Vector{Particle3D}, 
                        new_x::Float64, new_y::Float64, new_z::Float64, min_distance::Float64)::Bool
    # 检查链上的粒子
    for particle in chain
        if compute_distance(particle.x, particle.y, particle.z, new_x, new_y, new_z) < min_distance
            return false
        end
    end
    
    # 检查其他自由粒子
    for bead in free_beads
        if compute_distance(bead.x, bead.y, bead.z, new_x, new_y, new_z) < min_distance
            return false
        end
    end
    
    return true
end

# 计算两点之间的距离
function compute_distance(x1::Float64, y1::Float64, z1::Float64, x2::Float64, y2::Float64, z2::Float64)::Float64
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return sqrt(dx^2 + dy^2 + dz^2)
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
    if !all(iszero, params.alpha_0)
        total_energy += ideal_chromosome_Pij(chain, params)
    end

    # 添加临时loop势
    if !isnothing(params.loop_anchor)
        a, b = params.loop_anchor
        r = distance(chain[a], chain[b])
        if r > params.loop_cutoff
            total_energy += params.loop_strength * (r - params.loop_cutoff)^2
        end
    end
        
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
    # Dₑ: 势阱深度。这是一个关键的能量参数，描述了键的强度。
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
    theta0 = π/2       
    
    for i in 2:length(chain)-1
        # 计算向量
        vec1 = [
            chain[i-1].x - chain[i].x,
            chain[i-1].y - chain[i].y,
            chain[i-1].z - chain[i].z
        ]
        vec2 = [
            chain[i+1].x - chain[i].x,
            chain[i+1].y - chain[i].y,
            chain[i+1].z - chain[i].z
        ]
        
        # 计算向量模长并处理分母为零的情况
        norm_vec1 = norm(vec1)
        norm_vec2 = norm(vec2)
        denominator = norm_vec1 * norm_vec2
        
        if denominator < eps(Float64)  # 避免除以零
            @warn "Vectors too short or colinear at index $i, skipping angle calculation."
            continue
        end

        # 计算余弦值并限制在有效范围内
        cos_theta = dot(vec1, vec2) / denominator
        cos_theta = clamp(cos_theta, -1.0, 1.0)  # 防止数值误差导致超出[-1,1]

        # 计算角度差
        theta_diff = acos(cos_theta) - theta0
        
        # 键角能量公式
        energy += k_angle * (1 - cos(theta_diff))
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


"""
    compute_specific_interaction_energy(...)

这是一个为特定场景（固定顺序的CN2-CN3混合）进行超优化的函数。
它假定 free_beads 数组的前 `count1` 个是CN2粒子，
接下来的 `count2` 个是CN3粒子。
"""
function compute_specific_interaction_energy(
    chain::Vector{Particle3D}, 
    free_beads::Vector{Particle3D},
    params::SimulationParameters
)::Float64
    # --- 1. 前置检查与参数提取 ---
    if all(iszero, params.alpha) || !isnothing(params.loop_anchor)
        return 0.0
    end

    N_chain = length(chain)
    Pij_mediated_matrix = params.Pij_mediated_matrix 
    fill!(Pij_mediated_matrix, 0.0)

    Pcutoff_ik_sq = params.Pcutoff_ik^2
    k_c = params.k_c
    r0 = params.r0
    
    # 根据约定，我们知道 count1 是 CN2 的数量, count2 是 CN3 的数量
    count1, count2 = params.tf_counts

    # --- 2. [第一部分] 处理所有 CN2 粒子 (硬编码优化) ---
    initial_neighbors_N2 = ntuple(_ -> (Inf, -1), Val(2))
    for k_bead in 1:count1
        bead = free_beads[k_bead]
        closest_neighbors = initial_neighbors_N2
        for i_mono in 1:N_chain
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
        for i in 1:2; params.fbead_contact[k_bead, i] = closest_neighbors[i][2]; end
        closest_neighbors[2][2] == -1 && continue
        
        dist1, idx1 = closest_neighbors[1]
        dist2, idx2 = closest_neighbors[2]
        i_pair, j_pair = minmax(idx1, idx2)
        if (j_pair - i_pair) >= 2
            P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
            P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
            term = log(max(eps(Float64), 1.0 - P_ik * P_jk))
            Pij_mediated_matrix[i_pair, j_pair] += term
        end
    end

    # --- 3. [第二部分] 处理所有 CN3 粒子 (硬编码优化) ---
    start_index = count1 + 1
    end_index = count1 + count2
    initial_neighbors_N3 = ntuple(_ -> (Inf, -1), Val(3))
    for k_bead in start_index:end_index
        bead = free_beads[k_bead]
        closest_neighbors = initial_neighbors_N3
        for i_mono in 1:N_chain
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
        for i in 1:3; params.fbead_contact[k_bead, i] = closest_neighbors[i][2]; end
        closest_neighbors[2][2] == -1 && continue

        # 硬编码循环范围以获得最佳性能: C(3,2) = 3 对
        for i in 1:2
            dist1, idx1 = closest_neighbors[i]; idx1 == -1 && break
            for j in (i + 1):3
                dist2, idx2 = closest_neighbors[j]; idx2 == -1 && break
                i_pair, j_pair = minmax(idx1, idx2)
                (j_pair - i_pair) < 2 && continue
                P_ik = 0.5 * (1.0 - tanh(k_c * (dist1 - r0)))
                P_jk = 0.5 * (1.0 - tanh(k_c * (dist2 - r0)))
                term = log(max(eps(Float64), 1.0 - P_ik * P_jk))
                Pij_mediated_matrix[i_pair, j_pair] += term
            end
        end
    end

    # --- 4. 最终能量计算 (不变) ---
    total_Pij = 0.0
    @inbounds for i in 1:N_chain
        alpha_row = view(params.alpha, i, :); Pij_mediated_row = view(Pij_mediated_matrix, i, :)
        sum_val = 0.0
        @simd for j in (i + 2):N_chain
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
    params.N_B != -1 && return 0.0  # 提前返回

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

function _compute_pij_core_N2!(
    chain::Vector{Particle3D}, free_beads::Vector{Particle3D}, params::SimulationParameters
)::Float64
    N_MAX_C = 2
    N_chain = length(chain)
    NB = length(free_beads)
    total_Pij = 0.0
    Pij_mediated_matrix = params.Pij_mediated_matrix 
    fill!(Pij_mediated_matrix, 0.0)

    Pcutoff_ik_sq = params.Pcutoff_ik^2
    k_c = params.k_c
    r0 = params.r0

    initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

    for k_bead in 1:NB
        bead = free_beads[k_bead]
        closest_neighbors = initial_neighbors

        # --- High-performance neighbor search using Tuples ---
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

        # Update contact matrix
        params.fbead_contact[k_bead, 1] = closest_neighbors[1][2]
        params.fbead_contact[k_bead, 2] = closest_neighbors[2][2]
        
        # If we didn't find at least two neighbors, continue
        if closest_neighbors[2][2] == -1
            continue
        end

        # --- Simplified Pairing Logic for N=2 (only one pair) ---
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

    # Final energy calculation (identical to all other versions)
    @inbounds for i in 1:N_chain
        alpha_row = view(params.alpha, i, :); Pij_mediated_row = view(Pij_mediated_matrix, i, :)
        sum_val = 0.0
        @simd for j in (i + 2):N_chain
            Pij_effective = 1.0 - exp(Pij_mediated_row[j])
            sum_val += alpha_row[j] * Pij_effective
        end
        total_Pij += sum_val
    end
    return total_Pij
end

function _compute_pij_core_N3!(
    chain::Vector{Particle3D}, free_beads::Vector{Particle3D}, params::SimulationParameters
)::Float64
    N_MAX_C = 3
    N_chain = length(chain)
    NB = length(free_beads)
    total_Pij = 0.0
    Pij_mediated_matrix = params.Pij_mediated_matrix 
    fill!(Pij_mediated_matrix, 0.0)

    Pcutoff_ik_sq = params.Pcutoff_ik^2
    k_c = params.k_c
    r0 = params.r0

    initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

    for k_bead in 1:NB
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

        # 编译器将完全展开这个双重循环 (只有3个配对)
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

    # 总能量计算部分与 N=4,5,6 版本完全一致
    @inbounds for i in 1:N_chain
        alpha_row = view(params.alpha, i, :); Pij_mediated_row = view(Pij_mediated_matrix, i, :)
        sum_val = 0.0
        @simd for j in (i + 2):N_chain
            Pij_effective = 1.0 - exp(Pij_mediated_row[j])
            sum_val += alpha_row[j] * Pij_effective
        end
        total_Pij += sum_val
    end
    return total_Pij
end

function _compute_pij_core_N4!(
    chain::Vector{Particle3D}, free_beads::Vector{Particle3D}, params::SimulationParameters
)::Float64
    N_MAX_C = 4
    N_chain = length(chain)
    NB = length(free_beads)
    total_Pij = 0.0
    Pij_mediated_matrix = params.Pij_mediated_matrix 
    fill!(Pij_mediated_matrix, 0.0)

    Pcutoff_ik_sq = params.Pcutoff_ik^2
    k_c = params.k_c
    r0 = params.r0

    initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

    for k_bead in 1:NB
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

        params.fbead_contact[k_bead, 1] = closest_neighbors[1][2]
        params.fbead_contact[k_bead, 2] = closest_neighbors[2][2]
        params.fbead_contact[k_bead, 3] = closest_neighbors[3][2]
        params.fbead_contact[k_bead, 4] = closest_neighbors[4][2]
        
        closest_neighbors[2][2] == -1 && continue

        # 编译器将完全展开这个双重循环
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

    @inbounds for i in 1:N_chain
        alpha_row = view(params.alpha, i, :); Pij_mediated_row = view(Pij_mediated_matrix, i, :)
        sum_val = 0.0
        @simd for j in (i + 2):N_chain
            Pij_effective = 1.0 - exp(Pij_mediated_row[j])
            sum_val += alpha_row[j] * Pij_effective
        end
        total_Pij += sum_val
    end
    return total_Pij
end

function _compute_pij_core_N5!(
    chain::Vector{Particle3D}, free_beads::Vector{Particle3D}, params::SimulationParameters
)::Float64
    N_MAX_C = 5
    N_chain = length(chain)
    NB = length(free_beads)
    total_Pij = 0.0
    Pij_mediated_matrix = params.Pij_mediated_matrix 
    fill!(Pij_mediated_matrix, 0.0)

    Pcutoff_ik_sq = params.Pcutoff_ik^2
    k_c = params.k_c
    r0 = params.r0

    initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

    for k_bead in 1:NB
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

    @inbounds for i in 1:N_chain
        alpha_row = view(params.alpha, i, :); Pij_mediated_row = view(Pij_mediated_matrix, i, :)
        sum_val = 0.0
        @simd for j in (i + 2):N_chain
            Pij_effective = 1.0 - exp(Pij_mediated_row[j])
            sum_val += alpha_row[j] * Pij_effective
        end
        total_Pij += sum_val
    end
    return total_Pij
end

function _compute_pij_core_N6!(
    chain::Vector{Particle3D}, free_beads::Vector{Particle3D}, params::SimulationParameters
)::Float64
    N_MAX_C = 6
    N_chain = length(chain)
    NB = length(free_beads)
    total_Pij = 0.0
    Pij_mediated_matrix = params.Pij_mediated_matrix 
    fill!(Pij_mediated_matrix, 0.0)

    Pcutoff_ik_sq = params.Pcutoff_ik^2
    k_c = params.k_c
    r0 = params.r0

    initial_neighbors = ntuple(_ -> (Inf, -1), Val(N_MAX_C))

    for k_bead in 1:NB
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

    @inbounds for i in 1:N_chain
        alpha_row = view(params.alpha, i, :); Pij_mediated_row = view(Pij_mediated_matrix, i, :)
        sum_val = 0.0
        @simd for j in (i + 2):N_chain
            Pij_effective = 1.0 - exp(Pij_mediated_row[j])
            sum_val += alpha_row[j] * Pij_effective
        end
        total_Pij += sum_val
    end
    return total_Pij
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
    i = rand(1:length(chain))
    # 保存旧坐标（链和自由粒子）
    old_chain = deepcopy(chain)
    old_free_beads = deepcopy(free_beads)
    
    # 保存被移动的自由粒子的旧坐标
    moved_beads = []
    for k in 1:length(free_beads)
        if params.fbead_contact[k, 1] == i 
            push!(moved_beads, (k, free_beads[k]))
        end
    end
    
    dx = params.DD * (2rand() - 1)
    dy = params.DD * (2rand() - 1)
    dz = params.DD * (2rand() - 1)
    
    p = chain[i]
    chain[i] = Particle3D(p.x + dx, p.y +dy, p.z + dz)



    # 移动关联的自由粒子并记录
    for (k, old_bead) in moved_beads
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
        chain[:] = old_chain
        free_beads[:] = old_free_beads
        return (0, current_energy)
    end
end

function mcpivot!(
    chain::Vector{Particle3D},
    free_beads::Vector{Particle3D},
    params::SimulationParameters,
    current_energy::Float64,
    current_temperature::Float64
)::Tuple{Int, Float64}
    # 随机选择一个枢轴点
    # current_energy = compute_total_energy!(chain, free_beads, params)
    pivot = rand(1:length(chain)-1)  # 枢轴点不能是链的最后一个单体
    
    # 随机生成旋转角度和轴
    θ = rand() * π  # 旋转角度 [0, π]
    axis = normalize([randn(), randn(), randn()])  # 随机旋转轴（归一化）
    
    # 创建旋转矩阵
    rotation_matrix = compute_rotation_matrix(axis, θ)
    
    # 保存旧坐标（链和自由粒子）
    old_chain = deepcopy(chain)
    old_free_beads = deepcopy(free_beads)
    
    # 获取枢轴点的坐标
    pivot_coords = [chain[pivot].x, chain[pivot].y, chain[pivot].z]
    N_max_c = maximum(params.tf_connectivities)
    # 旋转移动：仅旋转枢轴点之后的部分
    for i in (pivot+1):length(chain)
        # 将单体相对于枢轴点平移到原点
        relative_coords = [
            chain[i].x - pivot_coords[1],
            chain[i].y - pivot_coords[2],
            chain[i].z - pivot_coords[3]
        ]
        
        # 应用旋转矩阵
        rotated_coords = rotation_matrix * relative_coords
        
        # 平移回原始位置
        chain[i] = Particle3D(
            rotated_coords[1] + pivot_coords[1],
            rotated_coords[2] + pivot_coords[2],
            rotated_coords[3] + pivot_coords[3]
        )
    end
    
    # 同时旋转与受影响单体相关的自由粒子
    for k in 1:length(free_beads)
        bead = free_beads[k]
        
        # 检查该自由粒子是否与枢轴点之后的单体相关
        for idx in 1:N_max_c
            monomer_idx = params.fbead_contact[k, idx]
            if monomer_idx > pivot && monomer_idx != -1
                # 将自由粒子相对于枢轴点平移到原点
                relative_coords = [
                    bead.x - pivot_coords[1],
                    bead.y - pivot_coords[2],
                    bead.z - pivot_coords[3]
                ]
                
                # 应用旋转矩阵
                rotated_coords = rotation_matrix * relative_coords
                
                # 平移回原始位置
                free_beads[k] = Particle3D(
                    rotated_coords[1] + pivot_coords[1],
                    rotated_coords[2] + pivot_coords[2],
                    rotated_coords[3] + pivot_coords[3]
                )
                
                # 每个自由粒子只需旋转一次
                break
            end
        end
    end
    
    # 计算能量变化（仅计算受影响的能量项）
    new_energy = compute_total_energy!(chain, free_beads, params)
    ΔE = new_energy - current_energy
    
    # Metropolis 判据
    if metropolis_accept(ΔE, current_temperature)
        return (1, new_energy)  # 接受移动
    else
        chain[:] = old_chain
        free_beads[:] = old_free_beads
        return (0, current_energy)  # 拒绝移动
    end
end

# 辅助函数：计算旋转矩阵
function compute_rotation_matrix(axis::Vector{Float64}, θ::Float64)::Matrix{Float64}
    cosθ = cos(θ)
    sinθ = sin(θ)
    ux, uy, uz = axis
    
    R = [
        cosθ + ux^2*(1-cosθ)       ux*uy*(1-cosθ) - uz*sinθ   ux*uz*(1-cosθ) + uy*sinθ;
        uy*ux*(1-cosθ) + uz*sinθ   cosθ + uy^2*(1-cosθ)       uy*uz*(1-cosθ) - ux*sinθ;
        uz*ux*(1-cosθ) - uy*sinθ   uz*uy*(1-cosθ) + ux*sinθ   cosθ + uz^2*(1-cosθ)
    ]
    return R
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

# 辅助函数：检查新位置是否有效
function is_position_valid(
    chain::Vector{Particle3D},
    free_beads::Vector{Particle3D},
    min_distance::Float64
)::Bool
    # 检查链内距离
    for i in 1:length(chain), j in i+1:length(chain)
        if compute_distance(chain[i], chain[j]) < min_distance
            return false
        end
    end
    
    # 检查与自由粒子的距离
    for bead in free_beads, particle in chain
        if compute_distance(bead, particle) < min_distance
            return false
        end
    end
    
    return true
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


# 辅助结构
struct RejectMove <: Exception end


function mcdoublepivot!(
    chain::Vector{Particle3D},
    free_beads::Vector{Particle3D},
    params::SimulationParameters,
    current_energy::Float64,
    current_temperature::Float64
)::Tuple{Int, Float64}
    # current_energy = compute_total_energy!(chain, free_beads, params)
    # Step 1: 随机选择两个有效枢纽点
    a, b = 0, 0
    valid = false
    N = length(chain)
    for _ in 1:100  # 防止无限循环
        a = rand(1:N-2)
        b = rand(a+2:N)  # 确保至少有一个中间单体
        if b <= N && (b - a) >= 2
            valid = true
            break
        end
    end
    !valid && return (0, current_energy)

    # Step 2: 计算旋转轴（基于a到b的向量）
    vec_ab = [chain[b].x - chain[a].x,
            chain[b].y - chain[a].y,
            chain[b].z - chain[a].z]
    if norm(vec_ab) < 1e-8  # 防止零向量
        return (0, current_energy)
    end
    axis = normalize(vec_ab)
    θ = 2π * rand()  # 完整旋转范围[0, 2π]

    # 构建旋转矩阵（使用Rodrigues公式）
    rotation_matrix = compute_rotation_matrix(axis, θ)

    # 保存旧状态
    old_chain = deepcopy(chain)
    old_free_beads = deepcopy(free_beads)
    a_coords = [chain[a].x, chain[a].y, chain[a].z]

    # Step 3: 旋转中间区域
    for i in (a+1):(b-1)
        rel_pos = [chain[i].x - a_coords[1],
                chain[i].y - a_coords[2],
                chain[i].z - a_coords[3]]
        rot_pos = rotation_matrix * rel_pos
        chain[i] = Particle3D(
            rot_pos[1] + a_coords[1],
            rot_pos[2] + a_coords[2],
            rot_pos[3] + a_coords[3]
        )
    end

    # Step 4: 更新关联的free beads
    for fb in 1:length(free_beads)
        for j in 1:1
            m_idx = params.fbead_contact[fb, j]
            if m_idx in (a+1):(b-1)
                rel_pos = [free_beads[fb].x - a_coords[1],
                        free_beads[fb].y - a_coords[2],
                        free_beads[fb].z - a_coords[3]]
                rot_pos = rotation_matrix * rel_pos
                free_beads[fb] = Particle3D(
                    rot_pos[1] + a_coords[1],
                    rot_pos[2] + a_coords[2],
                    rot_pos[3] + a_coords[3]
                )
                break
            end
        end
    end

    # Step 5: 能量计算和判据
    new_energy = compute_total_energy!(chain, free_beads, params)
    ΔE = new_energy - current_energy

    if metropolis_accept(ΔE, current_temperature)
        return (1, new_energy)
    else
        chain[:] = old_chain
        free_beads[:] = old_free_beads
        return (0, current_energy)
    end
end

# Rodrigues旋转矩阵计算公式
function compute_rotation_matrix(axis::Vector{Float64}, θ::Float64)
    u = normalize(axis)
    ux, uy, uz = u
    cosθ = cos(θ)
    sinθ = sin(θ)
    
    [cosθ + ux^2*(1-cosθ)      ux*uy*(1-cosθ) - uz*sinθ   ux*uz*(1-cosθ) + uy*sinθ;
    uy*ux*(1-cosθ) + uz*sinθ  cosθ + uy^2*(1-cosθ)       uy*uz*(1-cosθ) - ux*sinθ;
    uz*ux*(1-cosθ) - uy*sinθ  uz*uy*(1-cosθ) + ux*sinθ   cosθ + uz^2*(1-cosθ)]
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

    for i in 1:length(values_str)
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


end # module PolymermcCPCMEM
