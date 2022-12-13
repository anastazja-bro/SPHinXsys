/**
 * @file 	diffusion_op.cpp
 * @brief 	This is the first test to validate the optimization.
 * @author 	Bo Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH; //Namespace cite here
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
Real bias_diffusion_coff = 0.0;
Real alpha = Pi / 4.0;
Vec2d bias_direction(cos(alpha), sin(alpha));
std::array<std::string, 1> species_name_list{ "Phi" };
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
std::vector<Vecd> createThermalDomain()
{
	std::vector<Vecd> thermalDomainShape;
	thermalDomainShape.push_back(Vecd(0.0, 0.0));
	thermalDomainShape.push_back(Vecd(0.0, H));
	thermalDomainShape.push_back(Vecd(L, H));
	thermalDomainShape.push_back(Vecd(L, 0.0));
	thermalDomainShape.push_back(Vecd(0.0, 0.0));

	return thermalDomainShape;
};

std::vector<Vecd> createBoundaryDomain()
{
	std::vector<Vecd> boundaryDomain;
	boundaryDomain.push_back(Vecd(-BW, -BW));
	boundaryDomain.push_back(Vecd(-BW, -BW));
	boundaryDomain.push_back(Vecd(-BW, H + BW));
	boundaryDomain.push_back(Vecd(L + BW, H + BW));
	boundaryDomain.push_back(Vecd(L + BW, -BW));
	boundaryDomain.push_back(Vecd(-BW, -BW));

	return boundaryDomain;
};

//----------------------------------------------------------------------
//	Define SPH bodies. 
//----------------------------------------------------------------------
class DiffusionBody : public MultiPolygonShape
{
public:
	explicit DiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::add);
	}
};

class WallBoundary : public MultiPolygonShape
{
public:
	explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createBoundaryDomain(), ShapeBooleanOps::add);
		multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::sub);
	}
};

//----------------------------------------------------------------------
//	Setup diffusion material properties. 
//----------------------------------------------------------------------
class DiffusionBodyMaterial :public DiffusionReaction<Solid>
{
public:
	DiffusionBodyMaterial() :DiffusionReaction<Solid>(species_name_list)
	{
		initializeAnDiffusion<LocalDirectionalDiffusion>("Phi", "Phi", diffusion_coff, bias_diffusion_coff, bias_direction);
	}
};

//----------------------------------------------------------------------
//	Application dependent initial condition. 
//----------------------------------------------------------------------
class DiffusionBodyInitialCondition
	: public DiffusionReactionInitialCondition<SolidParticles, Solid>
{
protected:
	size_t phi_;

	void update(size_t index_i, Real dt)
	{
		species_n_[phi_][index_i] = 400 + 50 * (double)rand() / RAND_MAX;
		heat_source_[index_i] = heat_source;
	};

public:
	DiffusionBodyInitialCondition(SPHBody& diffusion_body) :
		DiffusionReactionInitialCondition<SolidParticles, Solid>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}
};


class WallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<SolidParticles, Solid>
{
protected:
	size_t phi_;

	void update(size_t index_i, Real dt)
	{
		species_n_[phi_][index_i] = -0.0;
		if (pos_[index_i][1] < 0 && pos_[index_i][0] > 0.45*L && pos_[index_i][0] < 0.55*L)
		{
			species_n_[phi_][index_i] = lower_temperature;
		}
		if (pos_[index_i][1] > 1 && pos_[index_i][0] > 0.45*L && pos_[index_i][0] < 0.55*L)
		{
			species_n_[phi_][index_i] = upper_temperature;
		}
	}
public:
	WallBoundaryInitialCondition(SolidBody &diffusion_body) :
		DiffusionReactionInitialCondition<SolidParticles, Solid>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}
};

//----------------------------------------------------------------------
//	Specify diffusion relaxation method. 
//----------------------------------------------------------------------
class DiffusionBodyRelaxation
	:public RelaxationOfAllDiffusionSpeciesRK2<
	 RelaxationOfAllDiffusionSpeciesComplex<SolidParticles, Solid, SolidParticles, Solid>>
{
public:
	DiffusionBodyRelaxation(ComplexRelation &body_complex_relation)
		:RelaxationOfAllDiffusionSpeciesRK2(body_complex_relation) {};
	virtual ~DiffusionBodyRelaxation() {};
};
//----------------------------------------------------------------------
//	An observer body to measure temperature at given positions. 
//----------------------------------------------------------------------
class ThermalDiffusivityObserverParticleGenerator : public ObserverParticleGenerator
{
public:
	ThermalDiffusivityObserverParticleGenerator(SPHBody &sph_body) : ObserverParticleGenerator(sph_body)
	{
		/** A line of measuring points at the middle line. */
		size_t number_of_observation_points = 20;
		Real range_of_measure = L;
		Real start_of_measure = 0;

		for (size_t i = 0; i < number_of_observation_points; ++i)
		{
			Vec2d point_coordinate(0.5*L, range_of_measure * (Real)i / (Real)(number_of_observation_points - 1) + start_of_measure);
			positions_.push_back(point_coordinate);
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
	SolidBody diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<DiffusionReactionParticles<SolidParticles, Solid>, DiffusionBodyMaterial>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();
	
	SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
	wall_boundary.defineParticlesAndMaterial<DiffusionReactionParticles<SolidParticles, Solid>, DiffusionBodyMaterial>();
	wall_boundary.generateParticles<ParticleGeneratorLattice>();
	//----------------------------  ------------------------------------------
	//	Particle and body creation of temperature observers.
	//----------------------------------------------------------------------
	ObserverBody thermal_diffusivity_observer(sph_system, "ThermalDiffusivityObserver");
	thermal_diffusivity_observer.generateParticles<ThermalDiffusivityObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation diffusion_body_inner(diffusion_body);
	InnerRelation wall_boundary_inner(wall_boundary);
	ComplexRelation diffusion_body_complex(diffusion_body, { &wall_boundary });
	ComplexRelation wall_boundary_complex(wall_boundary, { &diffusion_body });
	ContactRelation thermal_diffusivity_observer_contact(thermal_diffusivity_observer, { &diffusion_body });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	SimpleDynamics<DiffusionBodyInitialCondition> setup_diffusion_initial_condition(diffusion_body);
	SimpleDynamics<WallBoundaryInitialCondition> setup_boundary_condition(wall_boundary);
	InteractionDynamics<UpdateUnitNormalVector<SolidParticles, Solid, SolidParticles, Solid>>
		update_diffusion_body_normal_vector(diffusion_body_complex);
	InteractionDynamics<UpdateUnitNormalVector<SolidParticles, Solid, SolidParticles, Solid>>
		update_wall_boundary_normal_vector(wall_boundary_complex);
	InteractionDynamics<UpdateNormalDistance<SolidParticles, Solid>>
		update_normal_distance_domain(diffusion_body_inner);
	InteractionDynamics<UpdateNormalDistance<SolidParticles, Solid>>
		update_normal_distance_wall(wall_boundary_inner);
	GetDiffusionTimeStepSize<SolidParticles, Solid> get_time_step_size(diffusion_body);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	RestartIO	restart_io(io_environment, sph_system.real_bodies_);
	ObservedQuantityRecording<Real> write_solid_temperature("Phi", io_environment, thermal_diffusivity_observer_contact);
	/************************************************************************/
	/*            splitting thermal diffusivity optimization                */
	/************************************************************************/
	DiffusionBodyRelaxation thermal_relaxation_with_boundary(diffusion_body_complex);
	InteractionSplit<TemperatureSplittingByPDEWithBoundary<SolidParticles, Solid, SolidParticles, Solid, Real>>
		temperature_splitting_with_boundary(diffusion_body_complex, "Phi");
	InteractionSplit<UpdateTemperaturePDEResidual<TemperatureSplittingByPDEWithBoundary<SolidParticles, Solid,
		SolidParticles, Solid, Real>, ComplexRelation, Real>>
		update_temperature_global_residual(diffusion_body_complex, "Phi");
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_temperature_local_residual(diffusion_body, "residual_T_local");
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_temperature_global_residual(diffusion_body, "residual_T_global");
	ReduceAverage<DiffusionReactionSpeciesSummation<SolidParticles, Solid>>
		calculate_averaged_temperature(diffusion_body, "Phi");
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary. 
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	setup_diffusion_initial_condition.exec();
	setup_boundary_condition.exec();
	update_diffusion_body_normal_vector.parallel_exec();
	update_wall_boundary_normal_vector.parallel_exec();
	update_normal_distance_domain.parallel_exec();
	update_normal_distance_wall.parallel_exec();
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
	int ite_splitting = 0;
	Real T0 = 10;
	Real End_Time = T0;
	Real Observe_time = 0.01 * End_Time;
	int restart_output_interval = 1000;
	Vec2d averaged_residual_T_last(0.0, 0.0);
	Vec2d averaged_residual_T_current(0.0, 0.0);
	Real dt = 0.0;

	/** Output global basic parameters.*/
	write_solid_temperature.writeToFile(ite);
	write_states.writeToFile(ite);
	//----------------------------------------------------------------------
	//	Statistics for CPU time
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	std::string filefullpath_error_PDE = io_environment.output_folder_ + "/" + "error_PDE.dat";
	std::ofstream out_file_error_PDE(filefullpath_error_PDE.c_str(), std::ios::app);
	std::string filefullpath_observed_temperature = io_environment.output_folder_ + "/" + "observed_temperature.dat";
	std::ofstream out_file_observed_temperature(filefullpath_observed_temperature.c_str(), std::ios::app);

	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real relaxation_time = 0.0;
		while (relaxation_time < Observe_time)
		{
			if (ite % 200 == 0)
			{
				std::cout << "N= " << ite << " Time: " << GlobalStaticVariables::physical_time_ << "	dt: " << dt << "\n";
				averaged_residual_T_current[0] = calculate_temperature_local_residual.parallel_exec(dt);
				averaged_residual_T_current[1] = calculate_temperature_global_residual.parallel_exec(dt);
				out_file_error_PDE << std::fixed << std::setprecision(9) << ite << "   " << averaged_residual_T_current[1] << "\n";
				out_file_observed_temperature << std::fixed << std::setprecision(9) << ite << "   " << calculate_averaged_temperature.parallel_exec() << "\n";
			}
			
			dt = get_time_step_size.parallel_exec();
			temperature_splitting_with_boundary.parallel_exec(dt);
			update_temperature_global_residual.parallel_exec(dt);
			
			ite++; relaxation_time += dt; GlobalStaticVariables::physical_time_ += dt;
			
			if (ite % 500 == 0)
			{
				write_states.writeToFile(ite);
				write_solid_temperature.writeToFile(ite);
			}

			if (ite % restart_output_interval == 0)
			{
				restart_io.writeToFile(ite);
			}
		}
	}
	out_file_error_PDE.close();
	out_file_observed_temperature.close();

	tick_count t4 = tick_count::now();
	tick_count::interval_t tt;
	tt = t4 - t1;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	return 0;
}