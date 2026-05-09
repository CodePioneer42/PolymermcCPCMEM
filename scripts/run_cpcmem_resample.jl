#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Distributed
using TOML
using ArgParse
using Printf
using PolymermcCPCMEM

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--sim-dir"
            help = "Path to one fitted simulation folder, e.g. CN2_70"
            required = true
            arg_type = String
        "--scales"
            help = "Comma-separated alpha scaling factors"
            default = "0.5,0.8,1.2,1.5,1.8"
            arg_type = String
        "--alpha-iter"
            help = "Iteration index in alpha_log; if <= 0, auto-detect the latest one"
            default = 0
            arg_type = Int
        "--dry-run"
            help = "Only print detected inputs and planned outputs"
            action = :store_true
    end
    return parse_args(settings)
end

function parse_scales(text::String)
    values = Float64[]
    for item in split(text, ",")
        stripped = strip(item)
        isempty(stripped) && continue
        push!(values, parse(Float64, stripped))
    end
    return values
end

function scale_token(scale::Real)
    return replace(replace(string(scale), "-" => "m"), "." => "p")
end

function find_latest_alpha_iter(alpha_dir::String)
    candidates = Int[]
    for name in readdir(alpha_dir)
        endswith(name, ".txt") || continue
        stem = first(splitext(name))
        value = tryparse(Int, stem)
        value === nothing && continue
        push!(candidates, value)
    end
    isempty(candidates) && error("No numeric alpha files found in $alpha_dir")
    return maximum(candidates)
end

function ensure_clean_tmp_dir(tmp_dir::String)
    isdir(tmp_dir) && rm(tmp_dir; recursive=true, force=true)
    mkpath(tmp_dir)
end

function run_one_scale(base_config::Dict, sim_dir::String, alpha_file::String, scale::Float64)
    N = base_config["simulation"]["N"]
    token = scale_token(scale)

    final_root = joinpath(sim_dir, "sim_out_alpha")
    final_iter_dir = joinpath(final_root, "iter_" * token)
    tmp_root = joinpath(final_root, "__tmp_" * token)
    tmp_iter_dir = joinpath(tmp_root, "sim_out", "iter_" * token)

    if isdir(final_iter_dir)
        println("skip scale=", scale, " because output already exists: ", final_iter_dir)
        return false
    end

    ensure_clean_tmp_dir(tmp_root)

    config = deepcopy(base_config)
    config["experiment"]["output_dir"] = tmp_root

    alpha = PolymermcCPCMEM.read_alpha_from_file(alpha_file, N)
    alpha .*= scale
    alpha0 = zeros(Float64, N, N)

    println("running scale=", scale, " -> ", final_iter_dir)
    mean_contact, mean_distance = PolymermcCPCMEM.run_parallel_simulations(
        alpha,
        alpha0,
        config;
        iter_num = token,
    )

    isdir(tmp_iter_dir) || error("Expected simulation output not found: $tmp_iter_dir")
    mkpath(final_root)
    mv(tmp_iter_dir, final_iter_dir)

    PolymermcCPCMEM.save_contact_map(alpha, joinpath(final_iter_dir, "alpha_scaled.txt"))
    PolymermcCPCMEM.save_contact_map(mean_contact, joinpath(final_iter_dir, "contact_resampled.txt"))
    PolymermcCPCMEM.save_contact_map(mean_distance, joinpath(final_iter_dir, "distance_resampled.txt"))

    config_to_save = deepcopy(base_config)
    config_to_save["experiment"]["output_dir"] = final_iter_dir
    write_config(joinpath(final_iter_dir, "run_config.toml"), config_to_save)

    rm(tmp_root; recursive=true, force=true)
    return true
end

function write_config(path::String, config::Dict)
    open(path, "w") do io
        TOML.print(io, config)
    end
end

args = parse_commandline()

sim_dir = abspath(expanduser(args["sim-dir"]))
scales = parse_scales(args["scales"])
alpha_iter = args["alpha-iter"]
dry_run = args["dry-run"]

isdir(sim_dir) || error("Simulation directory not found: $sim_dir")
isempty(scales) && error("No valid scales were provided.")

config_path = joinpath(sim_dir, "run_config.toml")
alpha_dir = joinpath(sim_dir, "alpha_log")

isfile(config_path) || error("Missing run_config.toml in $sim_dir")
isdir(alpha_dir) || error("Missing alpha_log directory in $sim_dir")

if alpha_iter <= 0
    alpha_iter = find_latest_alpha_iter(alpha_dir)
end

alpha_file = joinpath(alpha_dir, string(alpha_iter) * ".txt")
isfile(alpha_file) || error("Alpha file not found: $alpha_file")

println("--- Loading Config: $config_path ---")
config = TOML.parsefile(config_path)

println("Detected simulation folder: ", sim_dir)
println("Detected alpha file:        ", alpha_file)
println("Scales:                     ", scales)
println("Output root:                ", joinpath(sim_dir, "sim_out_alpha"))

if dry_run
    for scale in scales
        println("plan -> ", joinpath(sim_dir, "sim_out_alpha", "iter_" * scale_token(scale)))
    end
    exit(0)
end

desired_workers = config["parallel"]["n_workers"]
if nworkers() < desired_workers
    addprocs(desired_workers - nworkers())
end
println("Total workers: $(nworkers())")

project_root = joinpath(@__DIR__, "..")
@everywhere begin
    using Pkg
    Pkg.activate($project_root)
    using PolymermcCPCMEM
    using LinearAlgebra
    using Statistics
    using Printf
end


for scale in scales
    success = run_one_scale(config, sim_dir, alpha_file, scale)

end

println("Finished scales: ", "/", length(scales))
