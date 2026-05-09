#!/usr/bin/env julia

#SBATCH --job-name CPCMEM_Opt
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --partition i64m512r
#SBATCH --output %j-out.txt
#SBATCH --error %j-err.txt

using Pkg
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()

using Distributed
using TOML
using ArgParse
using Printf

# --- 1. 参数解析 ---
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config", "-c"
            help = "Path to the TOML configuration file"
            required = true
            arg_type = String
    end
    return parse_args(s)
end

args = parse_commandline()
config_path = args["config"]

if !isfile(config_path)
    error("Config file not found: $config_path")
end

println("--- Loading Config: $config_path ---")
config = TOML.parsefile(config_path)

# --- 2. 设置输出与备份 ---
output_dir = config["experiment"]["output_dir"]
if !isdir(output_dir)
    mkpath(output_dir)
end
cp(config_path, joinpath(output_dir, "run_config.toml"), force=true)

# --- 3. 设置并行环境 ---
desired_workers = config["parallel"]["n_workers"]
if nworkers() < desired_workers
    addprocs(desired_workers - nworkers())
end
println("Total workers: $(nworkers())")

# 在所有核心上加载包
@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PolymermcCPCMEM
    using LinearAlgebra
    using Statistics
    using Printf
end

# --- 4. 准备数据 ---
N = config["simulation"]["N"]
hic_data_path = config["input"]["hic_data"]

println("Reading Hi-C target from: $hic_data_path")
target_contact = read_alpha_from_file(hic_data_path, N)

# 初始化 alpha (全0，不再读取文件)
input_alpha = config["input"]["input_alpha"]
initial_alpha = input_alpha
initial_alpha_0 = zeros(Float64, N, N)

# --- 5. 运行优化 ---
# 直接将 config 字典传入，不再需要在脚本里拆包
optimize_alpha_main(
    initial_alpha,
    initial_alpha_0,
    target_contact,
    config
)