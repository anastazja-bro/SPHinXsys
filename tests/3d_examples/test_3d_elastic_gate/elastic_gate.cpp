/**
 * @file 	elastic_gate.cpp
 * @brief 	3D elastic gate deformation due to dam break force.
 * @details This is the one of the basic test cases, also the first case for
 * 			understanding SPH method for fluid-structure-interaction (FSI) simulation.
 * @author 	Anastazja Broniatowska
 */
#include "sphinxsys.h" //	SPHinXsys Library.
#include "precice/SolverInterface.hpp"
using namespace SPH;   //	Namespace cite here.
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 0.5;						/**< Tank length. */
Real DH = 0.142;						/**< Tank height. */
Real DW = 0.02;							/**< Tank width*/
Real Dam_L = 0.1;						/**< Water block width. */
Real Dam_H = 0.14;						/**< Water block height. */
Real Gate_width = 0.005;					/**< Width of the gate. */
Real Base_bottom_position = 0.079;		/**< Position of gate base. (In Y direction) */
Real resolution_ref = 0.002; /**< Initial reference particle spacing. */
Real BW = resolution_ref * 4.0;			/**< Extending width for BCs. */
Vecd observer_pos(Dam_L + Gate_width, 0, DW / 2);
StdVec<Vecd> observation_location = {observer_pos};


//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_f = 1000.0;						   /**< Reference density of fluid. */
Real gravity_g = 9.81;				   /**< Value of gravity. */
Real U_f = 2.0 * sqrt(gravity_g * Dam_H);	/**< Characteristic velocity. */
Real c_f = 10 * U_f; /**< Reference sound speed. */
//----------------------------------------------------------------------
//	Material parameters of the elastic gate.
//----------------------------------------------------------------------
Real rho0_s = 1100.0;	 /**< Reference density of gate. */
Real poisson = 0.4; /**< Poisson ratio. */
Real Youngs_modulus = 10e6;	
Real Ae = Youngs_modulus / rho0_f / U_f / U_f;  /**< Normalized Youngs Modulus. */

//	define the water block shape
class WaterBlock : public ComplexShape
{
public:
	explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
	{
		Vecd halfsize_water(0.5 * Dam_L, 0.5 * Dam_H, 0.5 * DW);
		Transformd translation_water(halfsize_water);
		add<TransformShape<GeometricShapeBox>>(Transformd(translation_water), halfsize_water);
	}
};

class WallBoundary : public ComplexShape
{
public:
	explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
	{
		Vecd halfsize_outer(0.5 * DL + BW, 0.5 * DH + BW, 0.5 * DW + BW);
		Vecd halfsize_inner(0.5 * DL, 0.5 * DH, 0.5 * DW);
		Vecd halfsize_gate(0.5 * Gate_width, 0.5 * (DH - Base_bottom_position) , 0.5 * DW);
		Vecd translation_gate(Dam_L + 0.5 * Gate_width, 0.5 * (DH + Base_bottom_position), 0.5 * DW );
		add<TransformShape<GeometricShapeBox>>(Transformd(halfsize_inner), halfsize_outer);
		subtract<TransformShape<GeometricShapeBox>>(Transformd(halfsize_inner), halfsize_inner);
		add<TransformShape<GeometricShapeBox>>(Transformd(translation_gate), halfsize_gate);
	}
};

class GateShape : public ComplexShape
{
public:
	explicit GateShape(const std::string &shape_name) : ComplexShape(shape_name)
	{
		Vecd halfsize_gate(0.26 * resolution_ref, 0.5 * (Base_bottom_position - 0.001), 0.5 * DW);
		Vecd translation_gate(Dam_L + 0.26 * resolution_ref, 0.5 * (Base_bottom_position + 0.001), 0.5 * DW );
		add<TransformShape<GeometricShapeBox>>(Transformd(translation_gate), halfsize_gate);
	}
};

//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem.
	//----------------------------------------------------------------------
	BoundingBox system_domain_bounds(Vecd(-BW, -BW, -BW), Vecd(DL + BW, DH + BW, DW + BW));
	SPHSystem system(system_domain_bounds, resolution_ref);
	IOEnvironment io_environment(system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	FluidBody water_block(system, makeShared<WaterBlock>("WaterBlock"));
	water_block.defineParticlesAndMaterial<FluidParticles, WeaklyCompressibleFluid>(rho0_f, c_f);
	water_block.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary(system, makeShared<WallBoundary>("WallBoundary"));
	wall_boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
	wall_boundary.generateParticles<ParticleGeneratorLattice>();

	SolidBody gate(system, makeShared<GateShape>("Gate"));
	gate.defineAdaptationRatios(1.15, 2.0);
	gate.defineParticlesAndMaterial<ElasticSolidParticles, SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
	gate.generateParticles<ParticleGeneratorLattice>();

	ObserverBody gate_observer(system, "Observer");
	gate_observer.defineAdaptationRatios(1.15, 2.0);
	gate_observer.generateParticles<ObserverParticleGenerator>(observation_location);
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	ComplexRelation water_block_complex_relation(water_block, RealBodyVector{&wall_boundary, &gate});
	InnerRelation gate_inner_relation(gate);
	ContactRelation gate_water_contact_relation(gate, {&water_block});
	ContactRelation gate_observer_contact_relation(gate_observer, {&gate});
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	//----------------------------------------------------------------------
	//	Algorithms of fluid dynamics.
	//----------------------------------------------------------------------
	Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannWithWall> pressure_relaxation(water_block_complex_relation);
	Dynamics1Level<fluid_dynamics::Integration2ndHalfRiemannWithWall> density_relaxation(water_block_complex_relation);
	InteractionWithUpdate<fluid_dynamics::DensitySummationFreeSurfaceComplex> update_density_by_summation(water_block_complex_relation);
	SimpleDynamics<TimeStepInitialization> initialize_a_fluid_step(water_block, makeShared<Gravity>(Vecd(0.0, -gravity_g, 0.0)));
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
	//----------------------------------------------------------------------
	//	Algorithms of FSI.
	//----------------------------------------------------------------------
	//SimpleDynamics<OffsetInitialPosition> gate_offset_position(gate, offset);
	SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
	SimpleDynamics<NormalDirectionFromBodyShape> gate_normal_direction(gate);
	InteractionDynamics<solid_dynamics::CorrectConfiguration> gate_corrected_configuration(gate_inner_relation);
	InteractionDynamics<solid_dynamics::FluidPressureForceOnSolidRiemann> fluid_pressure_force_on_gate(gate_water_contact_relation);
	solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(gate);
	//----------------------------------------------------------------------
	//	Algorithms of Elastic dynamics.
	//----------------------------------------------------------------------
	Dynamics1Level<solid_dynamics::Integration1stHalf> gate_stress_relaxation_first_half(gate_inner_relation);
	Dynamics1Level<solid_dynamics::Integration2ndHalf> gate_stress_relaxation_second_half(gate_inner_relation);
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> gate_computing_time_step_size(gate);

	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> gate_update_normal(gate);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	water_block.addBodyStateForRecording<Real>("Density");
	BodyStatesRecordingToPlt write_real_body_states_to_plt(io_environment, system.real_bodies_);
	BodyStatesRecordingToVtp write_real_body_states_to_vtp(io_environment, system.real_bodies_);
	RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
		write_beam_tip_displacement("Position", io_environment, gate_observer_contact_relation);
	//TODO: observing position is not as good observing displacement. 
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	//gate_offset_position.parallel_exec();
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	wall_boundary_normal_direction.parallel_exec();
	gate_normal_direction.parallel_exec();
	gate_corrected_configuration.parallel_exec();
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int number_of_iterations = 0;
	int screen_output_interval = 100;
	Real end_time = 20.0;
	Real output_interval = end_time / 20.0;
	Real dt = 0.0;					/**< Default acoustic time step sizes. */
	Real dt_s = 0.0;				/**< Default acoustic time step sizes for solid. */
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_real_body_states_to_vtp.writeToFile();
	write_beam_tip_displacement.writeToFile();
	//----------------------------------------------------------------------
	// preCICE set up
	//----------------------------------------------------------------------
	precice::SolverInterface precice("FluidSolver", "precice-config.xml",0,1);
	int dim = precice.getDimensions();
	int meshID = precice.getMeshID("FluidMesh");
	int vertexSize; // number of vertices at wet surface 
	vertexSize = gate.LoopRange();
	std::cout << vertexSize << std::endl;

	// coords of coupling vertices must be in format (x0,y0,z0,x1,y1,z1,...)
	vector<double> coords; 
	coords.reserve(vertexSize*dim);
	for(int index = 0; index < vertexSize; ++index)
	{
		for(int dimension = 0; dimension < dim; ++dimension)
		{
			coords.push_back(gate.getBaseParticles().pos_[index][dimension]);
		}
	}
	vector<int> vertexIDs(vertexSize);
	precice.setMeshVertices(meshID, vertexSize, coords.data(), vertexIDs.data()); 

	int displID = precice.getDataID("Displacements", meshID); 
	int forceID = precice.getDataID("Forces", meshID); 
	vector<double> forces(vertexSize*dim);
	vector<double> displacements(vertexSize*dim);

	double precice_dt; // maximum precice timestep size
	precice_dt = precice.initialize();
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < end_time)
	{
		Real integration_time = 0.0;
		/** Integrate time (loop) until the next output time. */
		while (integration_time < output_interval)
		{
			/** Acceleration due to viscous force and gravity. */
			initialize_a_fluid_step.parallel_exec();
			Real Dt = get_fluid_advection_time_step_size.parallel_exec();
			update_density_by_summation.parallel_exec();
			/** Update normal direction at elastic body surface. */
			gate_update_normal.parallel_exec();
			Real relaxation_time = 0.0;
			while (relaxation_time < Dt)
			{
				/** Fluid relaxation and force computation. */
				dt = get_fluid_time_step_size.parallel_exec();
				dt= min(precice_dt, dt);
				pressure_relaxation.parallel_exec(dt);
				fluid_pressure_force_on_gate.parallel_exec();
				density_relaxation.parallel_exec(dt);
				/* set forces vector for preCICE in format (fx0,fy0,fz0,fx1,fy1,fz1,...)*/
				for(int index = 0; index < vertexSize; ++index)
				{
					for(int dimension = 0; dimension < dim; ++dimension)
					{
						forces[index*dim+dimension] = gate.getBaseParticles().acc_prior_[index][dimension]*
												gate.getBaseParticles().ParticleMass(index);
					}
				}
				precice.writeBlockVectorData(forceID, vertexSize, vertexIDs.data(), forces.data());
				/** Solid dynamics time stepping. */
				//Real dt_s_sum = 0.0;
				average_velocity_and_acceleration.initialize_displacement_.parallel_exec();
				// read displacements from preCICE and update gate particles positions
				precice.readBlockVectorData(displID, vertexSize, vertexIDs.data(), displacements.data());
				for(int index = 0; index < vertexSize; ++index)
				{
					for(int dimension = 0; dimension < dim; ++dimension)
					{
						gate.getBaseParticles().pos_[index][dimension] += displacements[index*dim+dimension];
					}
				}
				
				// while (dt_s_sum < dt)
				// {
				// 	dt_s = gate_computing_time_step_size.parallel_exec();
				// 	if (dt - dt_s_sum < dt_s)
				// 		dt_s = dt - dt_s_sum;
				// 	gate_stress_relaxation_first_half.parallel_exec(dt_s);
				// 	gate_stress_relaxation_second_half.parallel_exec(dt_s);
				// 	dt_s_sum += dt_s;
				// }
				average_velocity_and_acceleration.update_averages_.parallel_exec(dt);
				/* advance precice, exchange and process all the data */
				precice_dt = precice.advance(dt);
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}

			if (number_of_iterations % screen_output_interval == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
						  << GlobalStaticVariables::physical_time_
						  << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";
				write_real_body_states_to_vtp.writeToFile();
			}
			number_of_iterations++;

			/** Update cell linked list and configuration. */
			water_block.updateCellLinkedListWithParticleSort(100);
			gate.updateCellLinkedList();
			water_block_complex_relation.updateConfiguration();
			gate_water_contact_relation.updateConfiguration();
			/** Output the observed data. */
			write_beam_tip_displacement.writeToFile(number_of_iterations);
		}
		tick_count t2 = tick_count::now();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();
	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	write_beam_tip_displacement.newResultTest();

	/* finalize preCICE */
	precice.finalize();

	return 0;
}
