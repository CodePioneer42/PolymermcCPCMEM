#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Distributed
using TOML
using ArgParse
using PolymermcCPCMEM

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--config", "-c"
            help = "Path to the TOML configuration file"
            required = true
            arg_type = String
        "--alpha-file", "-a"
            help = "Path to the alpha matrix file"
            required = true
            arg_type = String
        "--output-dir", "-o"
            help = "Override experiment.output_dir from the config"
            default = ""
            arg_type = String
        "--iter-token"
            help = "Token used for sim_out/iter_<token>"
            default = "single_alpha"
            arg_type = String
        "--dry-run"
            help = "Print detected inputs and planned outputs without running simulation"
            action = :store_true
    end
    return parse_args(settings)
end

function write_config(path::String, config::Dict)
    open(path, "w") do io
        TOML.print(io, config)
    end
end

args = parse_commandline()

config_path = abspath(expanduser(args["config"]))
alpha_file = abspath(expanduser(args["alpha-file"]))
iter_token = args["iter-token"]

isfile(config_path) || error("Config file not found: $config_path")
isfile(alpha_file) || error("Alpha file not found: $alpha_file")

println("--- Loading Config: $config_path ---")
config = TOML.parsefile(config_path)

if !isempty(args["output-dir"])
    config["experiment"]["output_dir"] = abspath(expanduser(args["output-dir"]))
end

output_dir = config["experiment"]["output_dir"]
snapshot_dir = joinpath(output_dir, "sim_out", "iter_" * iter_token)

if args["dry-run"]
    println("Dry run: single alpha simulation")
    println("Config:       ", config_path)
    println("Alpha file:   ", alpha_file)
    println("Output dir:   ", output_dir)
    println("Snapshot dir: ", snapshot_dir)
    exit(0)
end

mkpath(output_dir)
cp(config_path, joinpath(output_dir, "run_config.toml"), force=true)
write_config(joinpath(output_dir, "run_config_effective.toml"), config)

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

N = config["simulation"]["N"]
alpha = PolymermcCPCMEM.read_alpha_from_file(alpha_file, N)
alpha_0 = zeros(Float64, N, N)

mean_contact, mean_distance = PolymermcCPCMEM.run_parallel_simulations(
    alpha,
    alpha_0,
    config;
    iter_num = iter_token,
)

mkpath(snapshot_dir)
PolymermcCPCMEM.save_contact_map(alpha, joinpath(snapshot_dir, "alpha_input.txt"))
PolymermcCPCMEM.save_contact_map(mean_contact, joinpath(snapshot_dir, "contact_single_alpha.txt"))
PolymermcCPCMEM.save_contact_map(mean_distance, joinpath(snapshot_dir, "distance_single_alpha.txt"))

println("Single alpha simulation complete: ", snapshot_dir)
