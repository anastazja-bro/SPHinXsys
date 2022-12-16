/**
 * @file 	diffusion_op.cpp
 * @brief 	This is the first test to validate the optimization.
 * @author 	Bo Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH;   // Namespace cite here
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real H = 1.0;
Real resolution_ref = H / 100.0;
Real BW = resolution_ref * 2.0;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coff = 1;
Real alpha = Pi / 4.0;
Vec2d bias_direction(cos(alpha), sin(alpha));
std::array<std::string, 1> species_name_list{"Phi"};
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_temperature = 0.0;
Real lower_temperature = 300.0;
Real upper_temperature = 350.0;
Real heat_source = 100.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
Vec2d solid_block_halfsize = Vec2d(0.5 * L, 0.5 * H);			 // local center at origin
Vec2d solid_block_translation = solid_block_halfsize;			 // translation to global coordinates
Vec2d constraint_halfsize = Vec2d(0.05 * L, 0.5 * BW);			 // top constraint region
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
class DiffusionBodyInitialCondition : public InitializationCondition<Real>
{
public:
	DiffusionBodyInitialCondition(SPHBody &diffusion_body)
		: InitializationCondition<Real>(diffusion_body, "Phi"){};

	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = 400 + 50 * (double)rand() / RAND_MAX;
	};
};

class IsothermalBoundariesConstraints
	: public InitializationCondition<Real>
{
protected:
	StdLargeVec<Vecd> &pos_;

public:
	IsothermalBoundariesConstraints(SolidBody &diffusion_body)
		: InitializationCondition<Real>(diffusion_body, "Phi"),
		pos_(particles_->pos_){};

	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = -0.0;
		if (pos_[index_i][1] < 0 && pos_[index_i][0] > 0.45 * L && pos_[index_i][0] < 0.55 * L)
		{
			variable_[index_i] = lower_temperature;
		}
		if (pos_[index_i][1] > 1 && pos_[index_i][0] > 0.45 * L && pos_[index_i][0] < 0.55 * L)
		{
			variable_[index_i] = upper_temperature;
		}
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
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody diffusion_body(sph_system, makeShared<TransformShape<GeometricShapeBox>>(
						Transform2d(solid_block_translation), solid_block_halfsize, "DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<SolidParticles, Solid>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();
	StdLargeVec<Real> body_temperature;
	diffusion_body.addBodyState<Real>(body_temperature, "Phi");
	diffusion_body.addBodyStateForRecording<Real>("Phi");
	diffusion_body.addBodyStateToRestart<Real>("Phi");

	SolidBody isothermal_boundaries(sph_system, makeShared<IsothermalBoundaries>("IsoThermalBoundaries"));
	isothermal_boundaries.defineParticlesAndMaterial<SolidParticles, Solid>();
	isothermal_boundaries.generateParticles<ParticleGeneratorLattice>();
	StdLargeVec<Real> constrained_temperature;
	isothermal_boundaries.addBodyState<Real>(constrained_temperature, "Phi");
	isothermal_boundaries.addBodyStateForRecording<Real>("Phi");
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
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	RestartIO restart_io(io_environment, sph_system.real_bodies_);
	/************************************************************************/
	/*            splitting thermal diffusivity optimization                */
	/************************************************************************/
	InteractionSplit<DampingPairwiseWithWall<Real, DampingPairwiseInner>>
		implicit_heat_transfer_solver(diffusion_body_complex, "Phi", diffusion_coff);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	setup_diffusion_initial_condition.parallel_exec();
	setup_boundary_condition.parallel_exec();
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
	Real T0 = 10;
	Real End_Time = T0;
	Real Observe_time = 0.01 * End_Time;
	Real dt = T0 / 1000.0;
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
	}

	tick_count t4 = tick_count::now();
	tick_count::interval_t tt;
	tt = t4 - t1;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	return 0;
}