/**
 * @file 	diffusion_op.cpp
 * @brief 	This is the first test to validate the optimization.
 * @author 	Bo Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH;   // Namespace cite here
//----------------------------------------------------------------------
//	Global geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0; // inner domain length
Real H = 1.0; // inner domain height
Real resolution_ref = H / 100.0;
Real BW = resolution_ref * 2.0; // boundary thickness
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Global parameters for material properties.
//----------------------------------------------------------------------
std::string coefficient_name = "ThermalConductionRate";
Real diffusion_coff = 1;
//----------------------------------------------------------------------
//	Global parameters for setting up variables.
//----------------------------------------------------------------------
std::string variable_name = "Phi";
Real lower_temperature = 300.0;
Real upper_temperature = 350.0;
Real stead_reference = upper_temperature - lower_temperature;
Real heat_source = 100.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
Vec2d solid_block_halfsize = Vec2d(0.5 * L, 0.5 * H);			 // local center at origin
Vec2d solid_block_translation = solid_block_halfsize;			 // translation to global coordinates
Vec2d constraint_halfsize = Vec2d(0.05 * L, 0.5 * BW);			 // constraint region size
Vec2d top_constraint_translation = Vec2d(0.5 * L, L + 0.5 * BW); // top constraint region
Vec2d bottom_constraint_translation = Vec2d(0.5 * L, -0.5 * BW); // bottom constraint region
class IsothermalBoundaries : public ComplexShape
{
public:
	explicit IsothermalBoundaries(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<TransformShape<GeometricShapeBox>>(Transform2d(top_constraint_translation), constraint_halfsize);
		add<TransformShape<GeometricShapeBox>>(Transform2d(bottom_constraint_translation), constraint_halfsize);
	}
};
//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class DiffusionBodyInitialCondition : public ValueAssignment<Real>
{
public:
	explicit DiffusionBodyInitialCondition(SPHBody &diffusion_body)
		: ValueAssignment<Real>(diffusion_body, variable_name){};

	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = 325.0 + 25.0 * (((double)rand() / (RAND_MAX)) - 0.5) * 2.0;
	};
};
//----------------------------------------------------------------------
//	Constraints for isothermal boundaries.
//----------------------------------------------------------------------
class IsothermalBoundariesConstraints : public ValueAssignment<Real>
{
public:
	explicit IsothermalBoundariesConstraints(SolidBody &isothermal_boundaries)
		: ValueAssignment<Real>(isothermal_boundaries, variable_name),
		  pos_(particles_->pos_){};

	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = pos_[index_i][1] > 0.0 ? lower_temperature : upper_temperature;
	}

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class DiffusionCoefficientDistribution : public ValueAssignment<Real>
{
public:
	explicit DiffusionCoefficientDistribution(SPHBody &diffusion_body)
		: ValueAssignment<Real>(diffusion_body, coefficient_name),
		  pos_(particles_->pos_){};
	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = pos_[index_i][1] < 0.5 * H ? 1.0 : 0.1;
	};

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody diffusion_body(sph_system,
							 makeShared<TransformShape<GeometricShapeBox>>(
								 Transform2d(solid_block_translation), solid_block_halfsize, "DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<SolidParticles, Solid>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();
	StdLargeVec<Real> body_temperature;
	diffusion_body.addBodyState<Real>(body_temperature, variable_name);
	diffusion_body.addBodyStateForRecording<Real>(variable_name);
	diffusion_body.addBodyStateToRestart<Real>(variable_name);
	StdLargeVec<Real> thermal_conduction_rate;
	diffusion_body.addBodyState<Real>(thermal_conduction_rate, coefficient_name);
	diffusion_body.addBodyStateForRecording<Real>(coefficient_name);
	diffusion_body.addBodyStateToRestart<Real>(coefficient_name);

	SolidBody isothermal_boundaries(sph_system, makeShared<IsothermalBoundaries>("IsoThermalBoundaries"));
	isothermal_boundaries.defineParticlesAndMaterial<SolidParticles, Solid>();
	isothermal_boundaries.generateParticles<ParticleGeneratorLattice>();
	StdLargeVec<Real> constrained_temperature;
	isothermal_boundaries.addBodyState<Real>(constrained_temperature, variable_name);
	isothermal_boundaries.addBodyStateForRecording<Real>(variable_name);
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	ComplexRelation diffusion_body_complex(diffusion_body, {&isothermal_boundaries});
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	SimpleDynamics<DiffusionBodyInitialCondition> setup_diffusion_initial_condition(diffusion_body);
	SimpleDynamics<IsothermalBoundariesConstraints> setup_boundary_condition(isothermal_boundaries);
	SimpleDynamics<DiffusionCoefficientDistribution> setup_diffusion_coefficient(diffusion_body);
	SimpleDynamics<SourceAssignment<Real>> thermal_source(diffusion_body, variable_name, heat_source);
	ReduceDynamics<SteadySolutionCheck<Real>> check_steady_temperature(diffusion_body, variable_name, stead_reference);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	RestartIO restart_io(io_environment, sph_system.real_bodies_);
	/************************************************************************/
	/*            splitting thermal diffusivity optimization                */
	/************************************************************************/
	InteractionSplit<DampingComplex<DampingPairwiseInnerVariableCoefficient<Real>,
									DampingPairwiseFromWallVariableCoefficient<Real>>>
		implicit_heat_transfer_solver(diffusion_body_complex, variable_name, coefficient_name);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	setup_diffusion_initial_condition.parallel_exec();
	setup_boundary_condition.parallel_exec();
	setup_diffusion_coefficient.parallel_exec();
	//----------------------------------------------------------------------
	//	Load restart file if necessary.
	//----------------------------------------------------------------------
	if (sph_system.RestartStep() != 0)
	{
		GlobalStaticVariables::physical_time_ = restart_io.readRestartFiles(sph_system.RestartStep());
		diffusion_body.updateCellLinkedList();
		diffusion_body_complex.updateConfiguration();
	}
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int ite = sph_system.RestartStep();
	Real T0 = 10.0;
	Real End_Time = T0;
	Real Observe_time = 0.01 * End_Time;
	Real dt = 1.0 / 1000.0;
	int restart_output_interval = 1000;

	/** Output global basic parameters.*/
	write_states.writeToFile(ite);
	//----------------------------------------------------------------------
	//	Statistics for CPU time
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real relaxation_time = 0.0;
		while (relaxation_time < Observe_time)
		{

			thermal_source.parallel_exec(dt);
			implicit_heat_transfer_solver.parallel_exec(dt);

			ite++;
			relaxation_time += dt;
			GlobalStaticVariables::physical_time_ += dt;

			if (ite % 100 == 0)
			{
				std::cout << "N= " << ite << " Time: " << GlobalStaticVariables::physical_time_ << "	dt: " << dt << "\n";
			}

			if (ite % restart_output_interval == 0)
			{
				restart_io.writeToFile(ite);
			}
		}

		write_states.writeToFile(ite);
		if (check_steady_temperature.parallel_exec())
		{
			std::cout << "Convergence is achieved at Time: " << GlobalStaticVariables::physical_time_ << "\n";
			return 0;
		}
	}

	tick_count t4 = tick_count::now();
	tick_count::interval_t tt;
	tt = t4 - t1;
	std::cout << "The computation has finished, but the solution is still not steady yet." << std::endl;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	return 0;
}