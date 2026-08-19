using Test
using PolymermcCPCMEM

const PMC = PolymermcCPCMEM

function shifted_lj(r, epsilon, sigma, cutoff)
    r >= cutoff && return 0.0
    sr6 = (sigma / r)^6
    src6 = (sigma / cutoff)^6
    return 4epsilon * (sr6^2 - sr6 - src6^2 + src6)
end

@testset "Polymer chain Lennard-Jones energy" begin
    epsilon = 0.7
    sigma = 0.9
    cutoff = 2.25
    params = SimulationParameters(
        N = 6,
        lj_epsilon = epsilon,
        lj_sigma = sigma,
        lj_cutoff = cutoff,
        lj_range = 80,
    )

    @testset "shifted potential and exclusions" begin
        r = 1.1
        chain = [
            Particle3D(0.0, 0.0, 0.0),
            Particle3D(10.0, 0.0, 0.0),
            Particle3D(20.0, 0.0, 0.0),
            Particle3D(r, 0.0, 0.0),
        ]
        expected = shifted_lj(r, epsilon, sigma, cutoff)
        @test PMC.compute_chain_nonbond_energy(chain, params) ≈ expected atol=1e-12
        @test PMC.lj_particle_chain_energy(1, chain[1], chain, params) ≈ expected atol=1e-12

        angle_neighbor_chain = [
            Particle3D(0.0, 0.0, 0.0),
            Particle3D(10.0, 0.0, 0.0),
            Particle3D(r, 0.0, 0.0),
        ]
        @test PMC.compute_chain_nonbond_energy(angle_neighbor_chain, params) == 0.0
        @test PMC.lj_particle_chain_energy(1, angle_neighbor_chain[1], angle_neighbor_chain, params) == 0.0

        cutoff_chain = [
            Particle3D(0.0, 0.0, 0.0),
            Particle3D(10.0, 0.0, 0.0),
            Particle3D(20.0, 0.0, 0.0),
            Particle3D(cutoff, 0.0, 0.0),
        ]
        @test PMC.compute_chain_nonbond_energy(cutoff_chain, params) == 0.0
    end

    @testset "global and local energies agree" begin
        chain = [
            Particle3D(0.0, 0.0, 0.0),
            Particle3D(1.4, 0.0, 0.0),
            Particle3D(2.2, 0.8, 0.0),
            Particle3D(1.0, 1.2, 0.2),
            Particle3D(2.4, 1.5, 0.4),
            Particle3D(0.5, 2.0, 0.6),
        ]
        global_energy = PMC.compute_chain_nonbond_energy(chain, params)
        local_sum = sum(
            PMC.lj_particle_chain_energy(i, chain[i], chain, params)
            for i in eachindex(chain)
        )
        @test global_energy ≈ local_sum / 2 atol=1e-12

        moved_idx = 4
        moved = Particle3D(1.2, 1.0, 0.4)
        moved_chain = copy(chain)
        moved_chain[moved_idx] = moved
        global_delta = PMC.compute_chain_nonbond_energy(moved_chain, params) - global_energy
        local_delta = (
            PMC.lj_particle_chain_energy(moved_idx, moved, moved_chain, params) -
            PMC.lj_particle_chain_energy(moved_idx, chain[moved_idx], chain, params)
        )
        @test global_delta ≈ local_delta atol=1e-12
    end
end
