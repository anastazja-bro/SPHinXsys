/**
 * @file    tethered_dead_fish_in_flow.cpp
 * @brief   fish flapping passively in flow
 * @author  Xiangyu Hu and Chi Zhang
 */
#include "sphinxsys.h"
#include "fish_and_bones.h"	//shapes for fish and bones
using namespace SPH;
//------------------------------------------------------------------------------
// global parameters for the case
//------------------------------------------------------------------------------
Real DL = 11.0;							/**< Channel length. */
Real DH = 8.0;							/**< Channel height. */
Real resolution_ref = 0.1;				/** Initial particle spacing. */
Real DL_sponge = resolution_ref * 20.0; /**< Sponge region to impose inflow condition. */
Real BW = resolution_ref * 4.0;			/**< Extending width for BCs. */
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-DL_sponge - BW, -BW), Vec2d(DL + BW, DH + BW));
Real cx = 2.0;			  /**< Center of fish in x direction. */
Real cy = 4.0;			  /**< Center of fish in y direction. */
Real fish_length = 3.738; /**< Length of fish. */
Real fish_shape_resolution = resolution_ref * 0.5;
Vecd tethering_point(-1.0, cy); /**< The tethering point. */
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real rho0_f = 1.0;
Real U_f = 1.0;
Real c_f = 10.0 * U_f;
Real Re = 5.0e3;
Real mu_f = rho0_f * U_f * (fish_length) / Re;
//----------------------------------------------------------------------
//	Material properties of the fish body.
//----------------------------------------------------------------------
Real rho0_s = 1.0;
Real poisson = 0.49;
Real Ae = 2.0e2;
Real Youngs_modulus = Ae * rho0_f * U_f * U_f;
//------------------------------------------------------------------------------
// geometric shape elements used in the case
//------------------------------------------------------------------------------
std::vector<Vecd> createWaterBlockShape()
{
	std::vector<Vecd> water_block_shape;
	water_block_shape.push_back(Vecd(-DL_sponge, 0.0));
	water_block_shape.push_back(Vecd(-DL_sponge, DH));
	water_block_shape.push_back(Vecd(DL, DH));
	water_block_shape.push_back(Vecd(DL, 0.0));
	water_block_shape.push_back(Vecd(-DL_sponge, 0.0));

	return water_block_shape;
}
Vec2d buffer_halfsize = Vec2d(0.5 * DL_sponge, 0.5 * DH);
Vec2d buffer_translation = Vec2d(-DL_sponge, 0.0) + buffer_halfsize;

std::vector<Vecd> createOuterWallShape()
{
	std::vector<Vecd> outer_wall_shape;
	outer_wall_shape.push_back(Vecd(-DL_sponge - BW, -BW));
	outer_wall_shape.push_back(Vecd(-DL_sponge - BW, DH + BW));
	outer_wall_shape.push_back(Vecd(DL + BW, DH + BW));
	outer_wall_shape.push_back(Vecd(DL + BW, -BW));
	outer_wall_shape.push_back(Vecd(-DL_sponge - BW, -BW));

	return outer_wall_shape;
}

std::vector<Vecd> createInnerWallShape()
{
	std::vector<Vecd> inner_wall_shape;
	inner_wall_shape.push_back(Vecd(-DL_sponge - 2.0 * BW, 0.0));
	inner_wall_shape.push_back(Vecd(-DL_sponge - 2.0 * BW, DH));
	inner_wall_shape.push_back(Vecd(DL + 2.0 * BW, DH));
	inner_wall_shape.push_back(Vecd(DL + 2.0 * BW, 0.0));
	inner_wall_shape.push_back(Vecd(-DL_sponge - 2.0 * BW, 0.0));

	return inner_wall_shape;
}

Real head_size = 1.0;
std::vector<Vecd> createFishBlockingShape()
{
	std::vector<Vecd> fish_blocking_shape;
	fish_blocking_shape.push_back(Vecd(cx + head_size, cy - 0.4));
	fish_blocking_shape.push_back(Vecd(cx + head_size, cy + 0.4));
	fish_blocking_shape.push_back(Vecd(cx + 5.0, cy + 0.4));
	fish_blocking_shape.push_back(Vecd(cx + 5.0, cy - 0.4));
	fish_blocking_shape.push_back(Vecd(cx + head_size, cy - 0.4));

	return fish_blocking_shape;
}
//----------------------------------------------------------------------
//	Complex shape for wall boundary, note that no partial overlap is allowed
//	for the shapes in a complex shape.
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
public:
	explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createWaterBlockShape(), ShapeBooleanOps::add);
		/** Exclude the fish body. */
		std::vector<Vecd> fish_shape = CreatFishShape(cx, cy, fish_length, fish_shape_resolution);
		multi_polygon_.addAPolygon(fish_shape, ShapeBooleanOps::sub);
	}
};

class WallBoundary : public MultiPolygonShape
{
public:
	explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		std::vector<Vecd> outer_shape = createOuterWallShape();
		std::vector<Vecd> inner_shape = createInnerWallShape();
		multi_polygon_.addAPolygon(outer_shape, ShapeBooleanOps::add);
		multi_polygon_.addAPolygon(inner_shape, ShapeBooleanOps::sub);
	}
};

class FishBody : public MultiPolygonShape
{
public:
	explicit FishBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		std::vector<Vecd> fish_shape = CreatFishShape(cx, cy, fish_length, fish_shape_resolution);
		multi_polygon_.addAPolygon(fish_shape, ShapeBooleanOps::add);
	}
};

MultiPolygon createFishHeadShape(SPHBody &sph_body)
{
	std::vector<Vecd> fish_shape = CreatFishShape(cx, cy, fish_length, sph_body.sph_adaptation_->ReferenceSpacing());
	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(fish_shape, ShapeBooleanOps::add);
	multi_polygon.addAPolygon(createFishBlockingShape(), ShapeBooleanOps::sub);
	return multi_polygon;
};

class FishObserverParticleGenerator : public ObserverParticleGenerator
{
public:
	explicit FishObserverParticleGenerator(SPHBody &sph_body) : ObserverParticleGenerator(sph_body)
	{
		positions_.push_back(Vecd(cx + resolution_ref, cy));
		positions_.push_back(Vecd(cx + fish_length - resolution_ref, cy));
	}
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
	Real u_ref_, t_ref_;
	AlignedBoxShape &aligned_box_;
	Vecd halfsize_;

	template <class BoundaryConditionType>
	explicit InflowVelocity(BoundaryConditionType &boundary_condition)
		: u_ref_(U_f), t_ref_(4.0),
		  aligned_box_(boundary_condition.getAlignedBox()),
		  halfsize_(aligned_box_.HalfSize()) {}

	Vecd operator()(const Vecd &position, const Vecd &velocity)
	{
		Vecd target_velocity = velocity;
		Real run_time = GlobalStaticVariables::physical_time_;
		Real u_ave = run_time < t_ref_ ? 0.5 * u_ref_ * (1.0 - cos(Pi * run_time / t_ref_)) : u_ref_;
		if (aligned_box_.checkInBounds(0, position))
		{
			target_velocity[0] = 1.5 * u_ave * (1.0 - position[1] * position[1] / halfsize_[1] / halfsize_[1]);
		}
		return target_velocity;
	}
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
	//----------------------------------------------------------------------
	//	Build up an SPHSystem.
	//----------------------------------------------------------------------
	SPHSystem system(system_domain_bounds, resolution_ref);
	system.setRunParticleRelaxation(false);
	system.setReloadParticles(true);
	system.handleCommandlineOptions(ac, av);
	IOEnvironment io_environment(system);
	//----------------------------------------------------------------------
	//	Creating bodies with corresponding materials and particles.
	//----------------------------------------------------------------------
	FluidBody water_block(system, makeShared<WaterBlock>("WaterBody"));
	water_block.defineParticlesAndMaterial<FluidParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
	water_block.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary(system, makeShared<WallBoundary>("Wall"));
	wall_boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
	wall_boundary.generateParticles<ParticleGeneratorLattice>();

	SolidBody fish_body(system, makeShared<FishBody>("FishBody"));
	fish_body.defineAdaptationRatios(1.15, 2.0);
	fish_body.defineBodyLevelSetShape();
	fish_body.defineParticlesAndMaterial<ElasticSolidParticles, NeoHookeanSolid>(rho0_s, Youngs_modulus, poisson);
	(!system.RunParticleRelaxation() && system.ReloadParticles())
		? fish_body.generateParticles<ParticleGeneratorReload>(io_environment, fish_body.getName())
		: fish_body.generateParticles<ParticleGeneratorLattice>();

	ObserverBody fish_observer(system, "Observer");
	fish_observer.generateParticles<FishObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation water_block_inner(water_block);
	InnerRelation fish_body_inner(fish_body);
	ComplexRelation water_block_complex(water_block_inner, {&wall_boundary, &fish_body});
	ContactRelation fish_body_contact(fish_body, {&water_block});
	ContactRelation fish_observer_contact(fish_observer, {&fish_body});
	//----------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//----------------------------------------------------------------------
	if (system.RunParticleRelaxation())
	{
		//----------------------------------------------------------------------
		//	Methods used for particle relaxation.
		//----------------------------------------------------------------------
		/** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_fish_body_particles(fish_body);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_fish_body(io_environment, fish_body);
		/** Write the particle reload files. */
		ReloadParticleIO write_particle_reload_files(io_environment, {&fish_body});
		/** A  Physics relaxation step. */
		relax_dynamics::RelaxationStepInner relaxation_step_inner(fish_body_inner);
		//----------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------
		random_fish_body_particles.parallel_exec(0.25);
		relaxation_step_inner.SurfaceBounding().parallel_exec();
		write_fish_body.writeToFile();
		//----------------------------------------------------------------------
		//	relax particles of the insert body.
		//----------------------------------------------------------------------
		int ite_p = 0;
		while (ite_p < 1000)
		{
			relaxation_step_inner.parallel_exec();
			ite_p += 1;
			if (ite_p % 200 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
				write_fish_body.writeToFile(ite_p);
			}
		}
		std::cout << "The physics relaxation process of inserted body finish !" << std::endl;

		/** Output results. */
		write_particle_reload_files.writeToFile();
		return 0;
	}
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	/** Periodic BCs in x direction. */
	PeriodicConditionUsingCellLinkedList periodic_condition(water_block, water_block.getBodyShapeBounds(), xAxis);
	SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
	SimpleDynamics<NormalDirectionFromBodyShape> fish_body_normal_direction(fish_body);
	/** Corrected configuration.*/
	InteractionDynamics<solid_dynamics::CorrectConfiguration> fish_body_corrected_configuration(fish_body_inner);
	SimpleDynamics<TimeStepInitialization> initialize_a_fluid_step(water_block);
	/** Evaluation of density by summation approach. */
	InteractionWithUpdate<fluid_dynamics::DensitySummationComplex> update_density_by_summation(water_block_complex);
	/** Time step size without considering sound wave speed. */
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
	/** Time step size with considering sound wave speed. */
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
	/** Pressure relaxation using verlet time stepping. */
	Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannWithWall> pressure_relaxation(water_block_complex);
	Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWall> density_relaxation(water_block_complex);
	/** Computing viscous acceleration. */
	InteractionDynamics<fluid_dynamics::ViscousAccelerationWithWall> viscous_acceleration(water_block_complex);
	/** Impose transport velocity formulation. */
	InteractionDynamics<fluid_dynamics::TransportVelocityCorrectionComplex> transport_velocity_correction(water_block_complex);
	/** Computing vorticity in the flow. */
	InteractionDynamics<fluid_dynamics::VorticityInner> compute_vorticity(water_block_inner);
	/** Inflow boundary condition. */
	BodyAlignedBoxByCell inflow_buffer(
		water_block, makeShared<AlignedBoxShape>(Transform2d(Vec2d(buffer_translation)), buffer_halfsize));
	SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>, BodyAlignedBoxByCell> parabolic_inflow(inflow_buffer, 0.3);
	//----------------------------------------------------------------------
	//	Algorithms of FSI.
	//----------------------------------------------------------------------
	InteractionDynamics<solid_dynamics::FluidViscousForceOnSolid> viscous_force_on_fish_body(fish_body_contact);
	InteractionDynamics<solid_dynamics::FluidForceOnSolidUpdate> fluid_force_on_fish_body(fish_body_contact, viscous_force_on_fish_body);
	//----------------------------------------------------------------------
	//	Algorithms of solid dynamics.
	//----------------------------------------------------------------------
	/** Time step size calculation. */
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> fish_body_computing_time_step_size(fish_body);
	/** Process of stress relaxation. */
	Dynamics1Level<solid_dynamics::Integration1stHalf>
		fish_body_stress_relaxation_first_half(fish_body_inner);
	Dynamics1Level<solid_dynamics::Integration2ndHalf>
		fish_body_stress_relaxation_second_half(fish_body_inner);
	/** Update normal direction on fish body.*/
	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection>
		fish_body_update_normal(fish_body);
	/** Compute the average velocity on fish body. */
	solid_dynamics::AverageVelocityAndAcceleration fish_body_average_velocity(fish_body);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations, observations
	//	and regression tests of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_real_body_states(io_environment, system.real_bodies_);
	ReducedQuantityRecording<ReduceDynamics<solid_dynamics::TotalForceOnSolid>> write_total_force_on_fish(io_environment, fish_body);
	ObservedQuantityRecording<Vecd> write_fish_displacement("Position", io_environment, fish_observer_contact);
	//----------------------------------------------------------------------
	//	Define the multi-body system
	//----------------------------------------------------------------------
	SimTK::MultibodySystem MBsystem;
	/** The bodies or matter of the MBsystem. */
	SimTK::SimbodyMatterSubsystem matter(MBsystem);
	/** The forces of the MBsystem.*/
	SimTK::GeneralForceSubsystem forces(MBsystem);
	SimTK::CableTrackerSubsystem cables(MBsystem);
	/** Mass properties of the fixed spot. */
	SimTK::Body::Rigid fixed_spot_info(SimTK::MassProperties(1.0, SimTK::Vec3(0), SimTK::UnitInertia(1)));
	SolidBodyPartForSimbody fish_head(fish_body, makeShared<MultiPolygonShape>(createFishHeadShape(fish_body), "FishHead"));
	/** Mass properties of the constrained spot. */
	SimTK::Body::Rigid tethered_spot_info(*fish_head.body_part_mass_properties_);
	/** Mobility of the fixed spot. */
	SimTK::MobilizedBody::Weld fixed_spot(matter.Ground(), SimTK::Transform(SimTK::Vec3(tethering_point[0], tethering_point[1], 0.0)),
										  fixed_spot_info, SimTK::Transform(SimTK::Vec3(0)));
	/** Mobility of the tethered spot.
	 * Set the mass center as the origin location of the planar mobilizer
	 */
	Vecd disp0 = fish_head.initial_mass_center_ - tethering_point;
	SimTK::MobilizedBody::Planar tethered_spot(fixed_spot, SimTK::Transform(SimTK::Vec3(disp0[0], disp0[1], 0.0)), tethered_spot_info, SimTK::Transform(SimTK::Vec3(0)));
	/** The tethering line give cable force.
	 * the start point of the cable path is at the origin location of the first mobilizer body,
	 * the end point is the tip of the fish head which has a distance to the origin
	 * location of the second mobilizer body origin location, here, the mass center
	 * of the fish head.
	 */
	Vecd disp_cable_end = Vecd(cx, cy) - fish_head.initial_mass_center_;
	SimTK::CablePath tethering_line(cables, fixed_spot, SimTK::Vec3(0), tethered_spot, SimTK::Vec3(disp_cable_end[0], disp_cable_end[1], 0.0));
	SimTK::CableSpring tethering_spring(forces, tethering_line, 100.0, 3.0, 10.0);

	// discrete forces acting on the bodies
	SimTK::Force::DiscreteForces force_on_bodies(forces, matter);
	fixed_spot_info.addDecoration(SimTK::Transform(), SimTK::DecorativeSphere(0.02));
	tethered_spot_info.addDecoration(SimTK::Transform(), SimTK::DecorativeSphere(0.4));
	/** Visualizer from simbody. */
	SimTK::Visualizer viz(MBsystem);
	/** Initialize the system and state. */
	SimTK::State state = MBsystem.realizeTopology();
	/** Time stepping method for multibody system.*/
	SimTK::RungeKuttaMersonIntegrator integ(MBsystem);
	integ.setAccuracy(1e-3);
	integ.setAllowInterpolation(false);
	integ.initialize(state);
	//----------------------------------------------------------------------
	//	Coupling between SimBody and SPH
	//----------------------------------------------------------------------
	ReduceDynamics<solid_dynamics::TotalForceForSimBody, SolidBodyPartForSimbody>
		force_on_tethered_spot(fish_head, MBsystem, tethered_spot, force_on_bodies, integ);
	SimpleDynamics<solid_dynamics::ConstraintBySimBody, SolidBodyPartForSimbody>
		constraint_tethered_spot(fish_head, MBsystem, tethered_spot, force_on_bodies, integ);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//	Prepare quantities, e.g. wall normal, fish body norm,
	//----------------------------------------------------------------------
	system.initializeSystemCellLinkedLists();
	periodic_condition.update_cell_linked_list_.parallel_exec();
	system.initializeSystemConfigurations();
	wall_boundary_normal_direction.parallel_exec();
	fish_body_normal_direction.parallel_exec();
	fish_body_corrected_configuration.parallel_exec();
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_real_body_states.writeToFile(0);
	write_fish_displacement.writeToFile(0);
	//----------------------------------------------------------------------
	//	Basic control parameters for time stepping.
	//----------------------------------------------------------------------
	int number_of_iterations = 0;
	int screen_output_interval = 100;
	Real end_time = 50.0;
	Real output_interval = end_time / 200.0;
	Real dt = 0.0;	 /**< Default acoustic time step sizes. */
	Real dt_s = 0.0; /**< Default acoustic time step sizes for solid. */
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	Main loop of time stepping starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < end_time)
	{
		Real integration_time = 0.0;
		while (integration_time < output_interval)
		{
			initialize_a_fluid_step.parallel_exec();
			Real Dt = get_fluid_advection_time_step_size.parallel_exec();
			update_density_by_summation.parallel_exec();
			viscous_acceleration.parallel_exec();
			transport_velocity_correction.parallel_exec();
			/** Viscous force exerting on fish body. */
			viscous_force_on_fish_body.parallel_exec();
			/** Update normal direction on fish body. */
			fish_body_update_normal.parallel_exec();
			Real relaxation_time = 0.0;
			while (relaxation_time < Dt)
			{
				// note that dt needs to sufficiently large to avoid divide zero
				// when computing solid average velocity for FSI
				dt = SMIN(get_fluid_time_step_size.parallel_exec(), Dt);
				/** Fluid dynamics process, first half. */
				pressure_relaxation.parallel_exec(dt);
				/** Fluid pressure force exerting on fish. */
				fluid_force_on_fish_body.parallel_exec();
				/** Fluid dynamics process, second half. */
				density_relaxation.parallel_exec(dt);
				/** Relax fish body by solid dynamics. */
				Real dt_s_sum = 0.0;
				fish_body_average_velocity.initialize_displacement_.parallel_exec();
				while (dt_s_sum < dt)
				{
					dt_s = SMIN(fish_body_computing_time_step_size.parallel_exec(), dt - dt_s_sum);
					fish_body_stress_relaxation_first_half.parallel_exec(dt_s);
					SimTK::State &state_for_update = integ.updAdvancedState();
					force_on_bodies.clearAllBodyForces(state_for_update);
					force_on_bodies.setOneBodyForce(state_for_update, tethered_spot,
													force_on_tethered_spot.parallel_exec());
					integ.stepBy(dt_s);
					constraint_tethered_spot.parallel_exec();
					fish_body_stress_relaxation_second_half.parallel_exec(dt_s);
					dt_s_sum += dt_s;
				}
				// note that dt needs to sufficiently large to avoid divide zero
				fish_body_average_velocity.update_averages_.parallel_exec(dt);
				write_total_force_on_fish.writeToFile(number_of_iterations);

				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
				parabolic_inflow.parallel_exec();
			}
			if (number_of_iterations % screen_output_interval == 0)
			{
				viz.report(integ.getState()); // visualize the motion of rigid body
				std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
						  << GlobalStaticVariables::physical_time_
						  << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";
			}
			number_of_iterations++;

			/** Water block configuration and periodic condition. */
			periodic_condition.bounding_.parallel_exec();
			water_block.updateCellLinkedListWithParticleSort(100);
			fish_body.updateCellLinkedList();
			periodic_condition.update_cell_linked_list_.parallel_exec();
			water_block_complex.updateConfiguration();
			/** Fish body contact configuration. */
			fish_body_contact.updateConfiguration();
			write_fish_displacement.writeToFile(number_of_iterations);
		}
		tick_count t2 = tick_count::now();
		compute_vorticity.parallel_exec();
		write_real_body_states.writeToFile();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	return 0;
}
