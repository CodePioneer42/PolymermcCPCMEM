using Test
using PolymermcCPCMEM

const PMC = PolymermcCPCMEM

function reference_soft_core(r, epsilon, sigma, E_cut)
    r_min = sigma * 2.0^(1.0 / 6.0)
    r > r_min && return 0.0
    r == 0.0 && return E_cut

    U_lj = 4epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon
    sigma_over_r_o_6 = 0.5 + sqrt(E_cut / (8epsilon))
    r_o = sigma / sigma_over_r_o_6^(1.0 / 6.0)
    if r <= r_o
        return 0.5E_cut * (1.0 + tanh(2.0U_lj / E_cut - 1.0))
    end
    return max(U_lj, 0.0)
end

@testset "Polymer soft-core repulsive energy" begin
    epsilon = 3.0
    sigma = 0.9
    E_cut = 6.0
    params = SimulationParameters(
        N = 6,
        lj_epsilon = epsilon,
        lj_sigma = sigma,
        soft_core_E_cut = E_cut,
        lj_range = 80,
    )
    sigma_sq, r_o_sq, r_min_sq = PMC.soft_core_squared_distances(epsilon, sigma, E_cut)
    r_o = sqrt(r_o_sq)
    r_min = sqrt(r_min_sq)

    @testset "finite core and continuous boundaries" begin
        @test r_o ≈ sigma atol=1e-12
        @test PMC.soft_core_repulsive_pair_energy(0.0, epsilon, sigma_sq, E_cut, r_o_sq, r_min_sq) == E_cut
        @test reference_soft_core(r_o, epsilon, sigma, E_cut) ≈ 0.5E_cut atol=1e-12
        @test reference_soft_core(r_o * (1.0 - 1e-8), epsilon, sigma, E_cut) ≈ 0.5E_cut atol=1e-6
        @test reference_soft_core(r_o * (1.0 + 1e-8), epsilon, sigma, E_cut) ≈ 0.5E_cut atol=1e-6
        @test reference_soft_core(r_min, epsilon, sigma, E_cut) ≈ 0.0 atol=1e-12
        @test reference_soft_core(r_min + 1e-8, epsilon, sigma, E_cut) == 0.0
        @test all(reference_soft_core(r, epsilon, sigma, E_cut) >= 0.0 for r in range(0.0, r_min * 1.1; length=100))
    end

    @testset "pair energy and exclusions" begin
        r = 0.95
        chain = [
            Particle3D(0.0, 0.0, 0.0),
            Particle3D(10.0, 0.0, 0.0),
            Particle3D(20.0, 0.0, 0.0),
            Particle3D(r, 0.0, 0.0),
        ]
        expected = reference_soft_core(r, epsilon, sigma, E_cut)
        @test PMC.compute_chain_nonbond_energy(chain, params) ≈ expected atol=1e-12
        @test PMC.lj_particle_chain_energy(1, chain[1], chain, params) ≈ expected atol=1e-12

        angle_neighbor_chain = [
            Particle3D(0.0, 0.0, 0.0),
            Particle3D(10.0, 0.0, 0.0),
            Particle3D(r, 0.0, 0.0),
        ]
        @test PMC.compute_chain_nonbond_energy(angle_neighbor_chain, params) == 0.0
        @test PMC.lj_particle_chain_energy(1, angle_neighbor_chain[1], angle_neighbor_chain, params) == 0.0
    end

    @testset "global and local energies agree" begin
        chain = [
            Particle3D(0.0, 0.0, 0.0),
            Particle3D(1.4, 0.0, 0.0),
            Particle3D(2.2, 0.8, 0.0),
            Particle3D(0.8, 0.5, 0.2),
            Particle3D(2.4, 1.5, 0.4),
            Particle3D(0.5, 1.0, 0.6),
        ]
        global_energy = PMC.compute_chain_nonbond_energy(chain, params)
        local_sum = sum(
            PMC.lj_particle_chain_energy(i, chain[i], chain, params)
            for i in eachindex(chain)
        )
        @test global_energy ≈ local_sum / 2 atol=1e-12

        moved_idx = 4
        moved = Particle3D(1.0, 0.8, 0.4)
        moved_chain = copy(chain)
        moved_chain[moved_idx] = moved
        global_delta = PMC.compute_chain_nonbond_energy(moved_chain, params) - global_energy
        local_delta = (
            PMC.lj_particle_chain_energy(moved_idx, moved, moved_chain, params) -
            PMC.lj_particle_chain_energy(moved_idx, chain[moved_idx], chain, params)
        )
        @test global_delta ≈ local_delta atol=1e-12
    end

    @testset "disabled and invalid parameters" begin
        disabled = SimulationParameters(lj_epsilon=0.0, soft_core_E_cut=0.0)
        @test PMC.compute_chain_nonbond_energy([Particle3D(0.0, 0.0, 0.0)], disabled) == 0.0
        @test_throws ArgumentError PMC.soft_core_squared_distances(-1.0, sigma, E_cut)
        @test_throws ArgumentError PMC.soft_core_squared_distances(epsilon, 0.0, E_cut)
        @test_throws ArgumentError PMC.soft_core_squared_distances(epsilon, sigma, 0.0)
    end
end
