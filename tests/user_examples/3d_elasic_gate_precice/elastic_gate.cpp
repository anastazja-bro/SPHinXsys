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
		Transformd translation_wall(halfsize_inner);
		add<TransformShape<GeometricShapeBox>>(Transformd(translation_wall), halfsize_outer);
		subtract<TransformShape<GeometricShapeBox>>(Transformd(translation_wall), halfsize_inner);
	}
};

class GateShape : public ComplexShape
{
public:
	explicit GateShape(const std::string &shape_name) : ComplexShape(shape_name)
	{
		Vecd halfsize_gate(0.5 * Gate_width, 0.5 * DH, 0.5 * DW);
		Vecd translation_gate(Dam_L + 0.5 * Gate_width, 0.5 * DH, 0.5 * DW );
		add<TransformShape<GeometricShapeBox>>(Transformd(translation_gate), halfsize_gate);
	}
};

SharedPtr<ComplexShape> createGateConstraint()
{
	auto constraint = makeShared<ComplexShape>("Constraint");
	Vecd halfsize_gate(0.5 * Gate_width, 0.5 * (DH - Base_bottom_position) , 0.5 * DW);
	Vecd translation_gate(Dam_L + 0.5 * Gate_width, 0.5 * (DH + Base_bottom_position), 0.5 * DW );
	constraint->add<TransformShape<GeometricShapeBox>>(Transformd(translation_gate), halfsize_gate);
	return constraint;
}

StdVec<Vecd> getObserversLocation()
{
	StdVec<Vecd> observation_loacation;
	observation_loacation.reserve(79*20);
	for (size_t y_pos = 0; y_pos < 79; ++y_pos)
	{
		for (size_t z_pos = 0; z_pos < 20; ++z_pos)
		{
			Vecd pos(Dam_L, y_pos * resolution_ref / 2, z_pos * resolution_ref / 2);
			observation_loacation.push_back(pos);
		}
	}
	return observation_loacation;
}


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
	StdVec<Vecd> observation_location = getObserversLocation();
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

	auto constraint = createGateConstraint();
	BodyRegionByParticle gate_base(gate, constraint);
	SimpleDynamics<solid_dynamics::FixConstraint, BodyRegionByParticle> constraint_gate_base(gate_base);
	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> gate_update_normal(gate);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	water_block.addBodyStateForRecording<Real>("Density");
	gate.addBodyStateForRecording<Vecd>("ForceFromFluid");
	BodyStatesRecordingToPlt write_real_body_states_to_plt(io_environment, system.real_bodies_);
	BodyStatesRecordingToVtp write_real_body_states_to_vtp(io_environment, system.real_bodies_);
	ObservedQuantityRecording<Vecd>
		write_force_on_gate("ForceFromFluid", io_environment, gate_observer_contact_relation);
	ObservedQuantityRecording<Vecd>
		write_gate_pos("Position", io_environment, gate_observer_contact_relation);
	//TODO: observing position is not as good observing displacement. 
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------

	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	wall_boundary_normal_direction.parallel_exec();
	gate_normal_direction.parallel_exec();
	gate_corrected_configuration.parallel_exec();
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int number_of_iterations = 0;
	int output_interval = 100;
	Real dt = 0.0;
	Real dt_s = 0.0;					/**< Default acoustic time step sizes. */
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_real_body_states_to_vtp.writeToFile();
	write_force_on_gate.writeToFile();
	//----------------------------------------------------------------------
	// preCICE set up
	//----------------------------------------------------------------------
	precice::SolverInterface precice("FluidSolver", "precice-config.xml",0,1);
	int dim = precice.getDimensions();
	int meshID = precice.getMeshID("FluidMesh");
	int vertexSize; // number of vertices at wet surface 
	vertexSize = gate_observer.LoopRange();

	// coords of coupling vertices must be in format (x0,y0,z0,x1,y1,z1,...)
	vector<double> coords; 
	coords.reserve(vertexSize*dim);
	for(int index = 0; index < vertexSize; ++index)
	{
		for(int dimension = 0; dimension < dim; ++dimension)
		{
			coords.push_back(gate_observer.getBaseParticles().pos_[index][dimension]);
		}
	}
	vector<int> vertexIDs(vertexSize);
	precice.setMeshVertices(meshID, vertexSize, coords.data(), vertexIDs.data()); 

	int displID = precice.getDataID("DisplacementDeltas", meshID); 
	int forceID = precice.getDataID("Forces", meshID); 
	vector<double> forces(vertexSize*dim);
	vector<double> displacements(vertexSize*dim);
	// this is done only for Calculix as it only sends absolute displacements
	vector<double> displacements_previous(vertexSize*dim);
	vector<double> delta(vertexSize*dim);
	//
	double precice_dt; // maximum precice timestep size
	precice_dt = precice.initialize();
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (precice.isCouplingOngoing())
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
			/* Compute and write forces */
			fluid_pressure_force_on_gate.parallel_exec();
			/* write forces vector for preCICE in format (fx0,fy0,fz0,fx1,fy1,fz1,...)*/
			//update observers values
			write_force_on_gate.writeToFile();
			for(int index = 0; index < vertexSize; ++index)
			{
				for(int dimension = 0; dimension < dim; ++dimension)
				{
					StdLargeVec<Vecd>& force_from_fluid = *(gate_observer.getBaseParticles().getVariableByName<Vecd>("ForceFromFluid"));
					forces[index*dim+dimension] = force_from_fluid[index][dimension];
				}
			}
			density_relaxation.parallel_exec(dt);
			precice.writeBlockVectorData(forceID, vertexSize, vertexIDs.data(), forces.data());
				
			/* advance precice, exchange and process all the data */
			precice_dt = precice.advance(dt);
			
			//make a copy of displ from previous step (only for Calculix)
			displacements_previous = displacements;
			// read displacements from preCICE and set Dirichlet BC 
			precice.readBlockVectorData(displID, vertexSize, vertexIDs.data(), displacements.data());
			// compute deltas (only for Calculix)
			
			for (size_t index=0; index<delta.size(); ++index)
			{
				delta[index] = displacements[index] - displacements_previous[index];
			}
			
			// set force to 0 so that gate computations are based on the Dirichlet BC
			StdLargeVec<Vecd>& force_from_fluid_gate = *(gate.getBaseParticles().getVariableByName<Vecd>("ForceFromFluid"));
			for (size_t iParicle = 0; iParicle < gate.LoopRange(); ++iParicle)
			{
				force_from_fluid_gate[iParicle]= Vecd::Zero();
			}
			
			//apply Dirichlet BCs
			average_velocity_and_acceleration.initialize_displacement_.parallel_exec();
			Real dt_s_sum = 0.0;
			while (dt_s_sum < dt)
			{
				dt_s = gate_computing_time_step_size.parallel_exec();
				if (dt - dt_s_sum < dt_s)
					dt_s = dt - dt_s_sum;
				for(int index = 0; index < vertexSize; ++index)
				{
					for(int dimension = 0; dimension < dim; ++dimension)
					{
						gate.getBaseParticles().vel_[index][dimension] = delta[index*dim+dimension]/dt;
						// another possibility is to set positions with known displ and velocities to 0
					}
				}
				gate_stress_relaxation_first_half.parallel_exec(dt_s);
				constraint_gate_base.parallel_exec();
				for(int index = 0; index < vertexSize; ++index)
				{
					for(int dimension = 0; dimension < dim; ++dimension)
					{
						gate.getBaseParticles().vel_[index][dimension] = delta[index*dim+dimension]/dt;
					}
				}
				gate_stress_relaxation_second_half.parallel_exec(dt_s);
				dt_s_sum += dt_s;
			}
			average_velocity_and_acceleration.update_averages_.parallel_exec(dt);
			write_gate_pos.writeToFile();
			
			relaxation_time += dt;
			GlobalStaticVariables::physical_time_ += dt;
			
			if (number_of_iterations % output_interval == 0)
			{
				write_real_body_states_to_vtp.writeToFile();
			}
			number_of_iterations++;

		}

		/** Update cell linked list and configuration. */
		water_block.updateCellLinkedListWithParticleSort(100);
		gate.updateCellLinkedList();
		water_block_complex_relation.updateConfiguration();
		gate_water_contact_relation.updateConfiguration();


	}
	tick_count t2 = tick_count::now();
	tick_count::interval_t tt;
	tt = t2 - t1;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;


	/* finalize preCICE */
	precice.finalize();

	return 0;
}
