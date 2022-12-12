/**
 * @file 	diffusion_heat_source_optimization.cpp
 * @brief 	This is the first test to validate the numerical optimization method.
 * @author 	Bo Zhang and Xiangyu Hu
 */
#include "sphinxsys.h"
using namespace SPH;

//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real H = 1.0;
Real resolution_ref = H / 100.0;
Real BW = resolution_ref * 4.0;
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
class DiffusionBodyMaterial : public DiffusionReaction<Solid>
{
public:
	DiffusionBodyMaterial()
		:DiffusionReaction<Solid>(species_name_list)
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
	DiffusionBodyInitialCondition(SolidBody &diffusion_body) :
		DiffusionReactionInitialCondition<SolidParticles, Solid>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}
};

class ThermalDiffusivityRandomInitialization
	: public DiffusionMaterialPropertyInitialization<SolidParticles, Solid, Real>
{
protected:
	void update(size_t index_i, Real dt)
	{
		/** setup the initial random thermal diffusivity and the averaged initial K is 1. */
		variable_[index_i] = 0.5 + (double)rand() / RAND_MAX;
	};
public:
	ThermalDiffusivityRandomInitialization(SolidBody &diffusion_body, const std::string &variable_name) :
	 	DiffusionMaterialPropertyInitialization<SolidParticles, Solid, Real>(diffusion_body, variable_name) {};
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
//  Impose constraints on the objective function
//----------------------------------------------------------------------
class ImposeObjectiveFunction
	: public DiffusionBasedMapping<SolidParticles, Solid>
{
protected:
	size_t phi_;
	void update(size_t index_i, Real learning_rate)
	{
		species_recovery_[index_i] = species_n_[phi_][index_i];
		species_modified_[index_i] = species_n_[phi_][index_i] - learning_rate * (upper_temperature - lower_temperature);
	}                                     
public:
	ImposeObjectiveFunction(SolidBody &diffusion_body) :
		DiffusionBasedMapping<SolidParticles, Solid>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}
};

class CopyOptimizedTemperature
	: public DiffusionBasedMapping<SolidParticles, Solid>
{
protected:
	size_t phi_;
	void update(size_t index_i, Real dt)
	{
		species_n_[phi_][index_i] = species_recovery_[index_i];
	}
public:
	CopyOptimizedTemperature(SolidBody& diffusion_body) :
		DiffusionBasedMapping<SolidParticles, Solid>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}
};

//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	/* Here is the initialization of the random generator. */
	srand((double)clock());
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
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation diffusion_body_inner(diffusion_body);
	ComplexRelation diffusion_body_complex(diffusion_body, { &wall_boundary });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	SimpleDynamics<DiffusionBodyInitialCondition> setup_diffusion_condition(diffusion_body);
	SimpleDynamics<WallBoundaryInitialCondition> setup_boundary_condition(wall_boundary);
	SimpleDynamics<ThermalDiffusivityRandomInitialization> 
		thermal_diffusivity_random_initialization(diffusion_body, "ThermalDiffusivity");
	/** Time step size calculation. */
	GetDiffusionTimeStepSize<SolidParticles, Solid> get_time_step_size(diffusion_body);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	RestartIO restart_io(io_environment, sph_system.real_bodies_);
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int ite = 0;                   /* define the loop of all operations for optimization. */
	int ite_T = 0;                 /* define loop index for temperature splitting iteration. */
	int ite_k = 0;                 /* define loop index for parameter splitting iteration. */
	int ite_rg = 0;                /* define loop index for parameter regularization. */

	int ite_T_total = 1;           /* define the total iteration for temperature splitting. */
	int ite_k_total = 14;          /* define the total iteration for parameter splitting. */
	int ite_rg_total = 1;          /* define the total iteration for parameter regularization. */
	int ite_loop = 0;              /* define loop index for optimization cycle. */

	int ite_T_comp_opt = 0;        /* define the real step index for splitting temperature by PDE. */
	int ite_T_comp_nonopt = 0;     /* define the real step index for splitting temperature by PDE. */
	int ite_output = 50;           /* define the interval for state output. */
	int ite_restart = 50;          /* define the interval for restart output. */

	/* Calculate the time step used for splitting. */
	Real dt = get_time_step_size.parallel_exec();

	/* Averaged parameter of the last cycle. */
	Real averaged_residual_T_last_local(10.0);
	Real averaged_residual_T_last_global(10.0);
	Real averaged_residual_k_last_local(10.0);
	Real averaged_residual_k_last_global(10.0);
	Real averaged_variation_last_local(10.0);
	Real averaged_variation_last_global(10.0);
	Real maximum_residual_T_last_global(10.0);
	Real maximum_residual_k_last_global(10.0);
	Real maximum_variation_last_global(10.0);

	/* Averaged parameter of the current cycle. */
	Real averaged_residual_T_current_local(0.0);
	Real averaged_residual_T_current_global(0.0);
	Real averaged_residual_k_current_local(0.0);
	Real averaged_residual_k_current_global(0.0);
	Real averaged_variation_current_local(0.0);
	Real averaged_variation_current_global(0.0);
	Real maximum_residual_T_current_global(10.0);
	Real maximum_residual_k_current_global(10.0);
	Real maximum_variation_current_global(10.0);

	/* Variation scale for last and current cycle. */
	Real scale_residual_T = 1.0;
	Real scale_residual_k = 1.0;
	Real scale_variation = 1.0;

	/* Initial parameter for limitation. */
	Real initial_averaged_residual_T = 0.0;
	Real initial_averaged_variation = 0.0;
	Real opt_averaged_temperature = 0.0;
	Real nonopt_averaged_temperature = 0.0;
	Real averaged_k_parameter = 0.0;
	Real initial_eta_regularization = 1;
	Real current_eta_regularization = 1;

	/* Parameters related to objective observed. */
	Real relative_difference = 1.0;
	Real last_averaged_temperature = 0.0;
	Real current_averaged_temperature = 0.0;

	/* Gradient descent parameter for objective function and regularization.*/
	Real decay_step_alpha = 200; /* The decay step for learning rate. */
	Real decay_step_RG = 100;    /* The decay step for regularization. */
	Real initial_learning_rate = 0.005;
	Real decay_rate_alpha = 0.999;
	Real initial_value_RG = dt;
	Real decay_rate_RG = 0.999;
	Real learning_rate_alpha = initial_learning_rate * pow(decay_rate_alpha, (ite_loop / decay_step_alpha));
	Real value_RG = initial_value_RG * pow(decay_rate_RG, (ite_loop / decay_step_RG));

	/************************************************************************/
	/*            splitting thermal diffusivity optimization                */
	/************************************************************************/
	/* Solve diffusion PDE equation by under optimized condition. */
	InteractionSplit<TemperatureSplittingByPDEWithBoundary<SolidParticles, Solid, SolidParticles, Solid, Real>>
		temperature_splitting_pde_complex(diffusion_body_complex, "Phi");
	
	/* Update the global PDE residual from the temperature. */
	InteractionSplit<UpdateTemperaturePDEResidual<TemperatureSplittingByPDEWithBoundary<SolidParticles,
		Solid, SolidParticles, Solid, Real>, ComplexRelation, Real>>
		update_temperature_pde_residual(diffusion_body_complex, "Phi");

	/* Impose objective function with gradient descent method. */
	SimpleDynamics<CopyOptimizedTemperature> copy_optimized_temperature(diffusion_body);
	SimpleDynamics<ImposeObjectiveFunction> impose_objective_function(diffusion_body);
	
	/* Splitting parameter(k) to revise PDE residual aroused by objective function. */
	InteractionSplit<ParameterSplittingByPDEWithBoundary<SolidParticles, Solid, SolidParticles, Solid, Real>>
		parameter_splitting_pde_complex(diffusion_body_complex, "ThermalDiffusivity");

	/* Update the PDE residual difference from the parameter. */
	InteractionSplit<UpdateParameterPDEResidual<ParameterSplittingByPDEWithBoundary<SolidParticles, 
		Solid, SolidParticles, Solid, Real>, ComplexRelation, Real>>
		update_parameter_pde_residual(diffusion_body_complex, "ThermalDiffusivity");

	/* Regularization of the parameter and update the global variation of whole domain. */
	InteractionSplit<RegularizationByDiffusionInner<SolidParticles, Solid, Real>>
		thermal_diffusivity_regularization(diffusion_body_inner, "ThermalDiffusivity",
			initial_eta_regularization, maximum_variation_current_global);
	InteractionSplit<UpdateRegularizationVariation<SolidParticles, Solid, Real>>
		update_regularization_global_variation(diffusion_body_inner, "ThermalDiffusivity");

	/* Impose the constraint of parameter, i.e., the summation of parameter should be fixed. */
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		total_averaged_thermal_diffusivity(diffusion_body, "ThermalDiffusivity");
	SimpleDynamics<ThermalDiffusivityConstrain<SolidParticles, Solid, Real>>
		thermal_diffusivity_constrain(diffusion_body, "ThermalDiffusivity");

	/* Calculate the averaged global/local residual from temperature/parameter and variation. */
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_temperature_local_residual(diffusion_body, "residual_T_local");
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_thermal_diffusivity_local_residual(diffusion_body, "residual_k_local");
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_regularization_local_variation(diffusion_body, "variation_local");
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_temperature_global_residual(diffusion_body, "residual_T_global");
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_thermal_diffusivity_global_residual(diffusion_body, "residual_k_global");
	ReduceAverage<ComputeAveragedErrorOrPositiveParameter<SolidParticles, Solid>>
		calculate_regularization_global_variation(diffusion_body, "variation_global");

	/* Update the maximum PDE residual and variation. */
	ReduceDynamics<ComputeMaximumError<SolidParticles, Solid>>
		calculate_maximum_error(diffusion_body, "residual_T_global");
	ReduceDynamics<ComputeMaximumError<SolidParticles, Solid>>
		calculate_maximum_error_change(diffusion_body, "residual_k_global");
	ReduceDynamics<ComputeMaximumError<SolidParticles, Solid>>
		calculate_maximum_variation(diffusion_body, "variation_global");

	/* Calculate the averaged temperature and real heat flux. */
	ReduceAverage<DiffusionReactionSpeciesSummation<SolidParticles, Solid>>
		calculate_averaged_opt_temperature(diffusion_body, "Phi");

	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary. 
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	setup_diffusion_condition.exec();
	setup_boundary_condition.exec();
	thermal_diffusivity_random_initialization.exec();
	
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
	//	Statistics for CPU time
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	std::string filefullpath_all_information = io_environment.output_folder_ + "/" + "all_information.dat";
	std::ofstream out_file_all_information(filefullpath_all_information.c_str(), std::ios::app);
		
	std::string filefullpath_error_plot = io_environment.output_folder_ + "/" + "error_plot.dat";
	std::ofstream out_file_error_plot(filefullpath_error_plot.c_str(), std::ios::app);
	
	std::string filefullpath_learning_rate = io_environment.output_folder_ + "/" + "learning_rate.dat";
	std::ofstream out_file_learning_rate(filefullpath_learning_rate.c_str(), std::ios::app);

	std::string filefullpath_value_RG = io_environment.output_folder_ + "/" + "value_RG.dat";
	std::ofstream out_file_value_RG(filefullpath_value_RG.c_str(), std::ios::app);

	std::string filefullpath_opt_temperature = io_environment.output_folder_ + "/" + "opt_temperature.dat";
	std::ofstream out_file_opt_temperature(filefullpath_opt_temperature.c_str(), std::ios::app);

	std::string filefullpath_nonopt_temperature = io_environment.output_folder_ + "/" + "nonopt_temperature.dat";
	std::ofstream out_file_nonopt_temperature(filefullpath_nonopt_temperature.c_str(), std::ios::app);

	std::string filefullpath_ite_T = io_environment.output_folder_ + "/" + "ite_T.dat";
	std::ofstream out_file_ite_T(filefullpath_ite_T.c_str(), std::ios::app);

	std::string filefullpath_ite_k = io_environment.output_folder_ + "/" + "ite_k.dat";
	std::ofstream out_file_ite_k(filefullpath_ite_k.c_str(), std::ios::app);

	std::string filefullpath_residual_comp = io_environment.output_folder_ + "/" + "residual_comp.dat";
	std::ofstream out_file_residual_comp(filefullpath_residual_comp.c_str(), std::ios::app);

	//----------------------------------------------------------------------
	//	Initial States update.
	//----------------------------------------------------------------------
	write_states.writeToFile(ite); //output the initial states.

	update_regularization_global_variation.parallel_exec(dt);
	initial_averaged_variation = calculate_regularization_global_variation.parallel_exec(dt);
	averaged_variation_current_global = initial_averaged_variation;
	maximum_variation_current_global = calculate_maximum_variation.parallel_exec(dt);
	maximum_variation_last_global = maximum_variation_current_global;

	/* Preliminary regularization to avoid extreme values during splitting. */
	thermal_diffusivity_regularization.variation_local_ = thermal_diffusivity_regularization.variation_global_;
	thermal_diffusivity_regularization.UpdateAveragedVariation(averaged_variation_current_global);
	thermal_diffusivity_regularization.UpdateMaximumVariation(maximum_variation_current_global);
	for (size_t i = 0; i != 10; ++i) { thermal_diffusivity_regularization.parallel_exec(value_RG / initial_eta_regularization); };
	update_regularization_global_variation.parallel_exec(dt);
	averaged_variation_current_global = calculate_regularization_global_variation.parallel_exec(dt);
	maximum_variation_current_global = averaged_variation_current_global;
	maximum_variation_last_global = averaged_variation_current_global;
	ite++; write_states.writeToFile(ite);

	/* Preliminary solve the PDE equation to avoid extreme values during splitting. */
	for (size_t i = 0; i != 10; ++i) { temperature_splitting_pde_complex.parallel_exec(dt); }
	update_temperature_pde_residual.parallel_exec(dt);
	initial_averaged_residual_T = calculate_temperature_global_residual.parallel_exec(dt);
	averaged_residual_T_current_global = initial_averaged_residual_T;
	averaged_residual_T_last_global = averaged_residual_T_current_global;
	maximum_residual_T_current_global = calculate_maximum_error.parallel_exec(dt);
	maximum_residual_T_last_global = maximum_residual_T_current_global;
	current_averaged_temperature = calculate_averaged_opt_temperature.parallel_exec();

	ite++; write_states.writeToFile(ite);
	out_file_nonopt_temperature << std::fixed << std::setprecision(12) << ite << "   " << current_averaged_temperature << "\n";
	out_file_opt_temperature << std::fixed << std::setprecision(12) << ite_T_comp_opt << "   " << current_averaged_temperature << "\n";
	
	out_file_error_plot << std::fixed << std::setprecision(12) << ite << "   " <<
		averaged_residual_T_current_global << "   " << maximum_residual_T_current_global << "   " <<
		averaged_residual_k_current_global << "   " << maximum_residual_k_current_global << "   " <<
		averaged_variation_current_global << "   " << maximum_variation_current_global << "\n";
	out_file_all_information << std::fixed << std::setprecision(12) << ite_loop << "   " << ite << "   " <<
		averaged_residual_T_current_local << "   " << averaged_residual_T_current_global << "   " << averaged_residual_k_current_local << "   " <<
		averaged_residual_k_current_global << "   " << averaged_variation_current_local << "   " << averaged_variation_current_global << "\n";

	while ((relative_difference > 0.00001 || averaged_residual_T_current_global > 0.000005 || learning_rate_alpha > 0.00001) && ite_loop < 10)
	{
		std::cout << "This is the beginning of the " << ite_loop << " iteration loop." << std::endl;

		//----------------------------------------------------------------------
		//	Impose Objective Function.
		//----------------------------------------------------------------------
		out_file_all_information << "This is the step of imposing objective function." << "\n";
		ite++;

		/* Reserve the global PDE residual to provide the reference for parameter splitting based on PDE. */
		temperature_splitting_pde_complex.residual_T_local_ = temperature_splitting_pde_complex.residual_T_global_;

		/* Impose objective function and BC constraint error may increase. */
		impose_objective_function.parallel_exec(learning_rate_alpha);
		out_file_learning_rate << std::fixed << std::setprecision(12) << ite_loop << "   " << learning_rate_alpha << "\n";

		update_temperature_pde_residual.parallel_exec(dt);
		averaged_residual_T_current_global = calculate_temperature_global_residual.parallel_exec(dt);
		maximum_residual_T_current_global = calculate_maximum_error.parallel_exec(dt);

		update_parameter_pde_residual.parallel_exec(dt);
		averaged_residual_k_current_global = calculate_thermal_diffusivity_global_residual.parallel_exec();
		maximum_residual_k_current_global = calculate_maximum_error_change.parallel_exec(dt);

		out_file_error_plot << std::fixed << std::setprecision(12) << ite << "   " <<
			averaged_residual_T_current_global << "   " << maximum_residual_T_current_global << "   " <<
			averaged_residual_k_current_global << "   " << maximum_residual_k_current_global << "   " <<
			averaged_variation_current_global << "   " << maximum_variation_current_global << "\n";
		out_file_all_information << std::fixed << std::setprecision(12) << ite_loop << "   " << ite << "   " <<
			averaged_residual_T_current_local << "   " << averaged_residual_T_current_global << "   " << averaged_residual_k_current_local << "   " <<
			averaged_residual_k_current_global << "   " << averaged_variation_current_local << "   " << averaged_variation_current_global << "\n";
		if (ite_loop % ite_output == 0) { write_states.writeToFile(ite); }
		std::cout << "N=" << ite << " and the objective function has been imposed. " << "\n";

		//----------------------------------------------------------------------
		//	Parameter splitting.
		//----------------------------------------------------------------------
		/* Parameter splitting should return back the increased error by imposing objective function. */
		out_file_all_information << "This is the beginning of thermal diffusivity splitting." << "\n";

		while (ite_k < ite_k_total)
		{
			//----------------------------------------------------------------------
			//	Parameter splitting by PDE equation.
			//----------------------------------------------------------------------
			ite++; ite_k++;
			parameter_splitting_pde_complex.parallel_exec(dt);

			if ((ite_loop % ite_output == 0) && (ite_k % 1 == 0 || ite_k == ite_k_total))
			{
				update_temperature_pde_residual.parallel_exec(dt);
				averaged_residual_T_current_global = calculate_temperature_global_residual.parallel_exec(dt);
				maximum_residual_T_current_global = calculate_maximum_error.parallel_exec(dt);

				update_parameter_pde_residual.parallel_exec(dt);
				averaged_residual_k_current_local = calculate_thermal_diffusivity_local_residual.parallel_exec(dt);
				averaged_residual_k_current_global = calculate_thermal_diffusivity_global_residual.parallel_exec(dt);
				maximum_residual_k_current_global = calculate_maximum_error_change.parallel_exec(dt);

				update_regularization_global_variation.parallel_exec(dt);
				averaged_variation_current_local = calculate_regularization_local_variation.parallel_exec(dt);
				averaged_variation_current_global = calculate_regularization_global_variation.parallel_exec(dt);
				maximum_variation_current_global = calculate_maximum_variation.parallel_exec(dt);

				out_file_error_plot << std::fixed << std::setprecision(12) << ite << "   " <<
					averaged_residual_T_current_global << "   " << maximum_residual_T_current_global << "   " <<
					averaged_residual_k_current_global << "   " << maximum_residual_k_current_global << "   " <<
					averaged_variation_current_global << "   " << maximum_variation_current_global << "\n";
				out_file_all_information << std::fixed << std::setprecision(12) << ite_loop << "   " << ite << "   " <<
					averaged_residual_T_current_local << "   " << averaged_residual_T_current_global << "   " << averaged_residual_k_current_local << "   " <<
					averaged_residual_k_current_global << "   " << averaged_variation_current_local << "   " << averaged_variation_current_global << "\n";

				//write_states.writeToFile(ite);
			}

			//----------------------------------------------------------------------
			//	Constraint of the summation of parameter.
			//----------------------------------------------------------------------
			ite++;
			averaged_k_parameter = total_averaged_thermal_diffusivity.parallel_exec(dt);
			thermal_diffusivity_constrain.UpdateAveragedParameter(averaged_k_parameter);
			thermal_diffusivity_constrain.parallel_exec(dt);
			if ((ite_loop % ite_output == 0) && (ite_k % 1 == 0 || ite_k == ite_k_total))
			{
				update_temperature_pde_residual.parallel_exec(dt);
				averaged_residual_T_current_global = calculate_temperature_global_residual.parallel_exec(dt);
				maximum_residual_T_current_global = calculate_maximum_error.parallel_exec(dt);

				update_parameter_pde_residual.parallel_exec(dt);
				averaged_residual_k_current_local = calculate_thermal_diffusivity_local_residual.parallel_exec(dt);
				averaged_residual_k_current_global = calculate_thermal_diffusivity_global_residual.parallel_exec(dt);
				maximum_residual_k_current_global = calculate_maximum_error_change.parallel_exec(dt);

				update_regularization_global_variation.parallel_exec(dt);
				averaged_variation_current_local = calculate_regularization_local_variation.parallel_exec(dt);
				averaged_variation_current_global = calculate_regularization_global_variation.parallel_exec(dt);
				maximum_variation_current_global = calculate_maximum_variation.parallel_exec(dt);

				out_file_error_plot << std::fixed << std::setprecision(12) << ite << "   " <<
					averaged_residual_T_current_global << "   " << maximum_residual_T_current_global << "   " <<
					averaged_residual_k_current_global << "   " << maximum_residual_k_current_global << "   " <<
					averaged_variation_current_global << "   " << maximum_variation_current_global << "\n";
				out_file_all_information << std::fixed << std::setprecision(12) << ite_loop << "   " << ite << "   " <<
					averaged_residual_T_current_local << "   " << averaged_residual_T_current_global << "   " << averaged_residual_k_current_local << "   " <<
					averaged_residual_k_current_global << "   " << averaged_variation_current_local << "   " << averaged_variation_current_global << "\n";

				//write_states.writeToFile(ite);
			}

			//----------------------------------------------------------------------
			//	Regularization.
			//----------------------------------------------------------------------
			ite++; ite_rg++;
			thermal_diffusivity_regularization.UpdateCurrentEta(current_eta_regularization);
			thermal_diffusivity_regularization.UpdateMaximumVariation(maximum_variation_current_global);
			thermal_diffusivity_regularization.UpdateAveragedVariation(averaged_variation_current_global);
			thermal_diffusivity_regularization.parallel_exec(value_RG);

			if ((ite_loop % ite_output == 0) && (ite_k % 1 == 0 || ite_k == ite_k_total))
			{
				update_temperature_pde_residual.parallel_exec(dt);
				averaged_residual_T_current_global = calculate_temperature_global_residual.parallel_exec(dt);
				maximum_residual_T_current_global = calculate_maximum_error.parallel_exec(dt);

				update_parameter_pde_residual.parallel_exec(dt);
				averaged_residual_k_current_local = calculate_thermal_diffusivity_local_residual.parallel_exec(dt);
				averaged_residual_k_current_global = calculate_thermal_diffusivity_global_residual.parallel_exec(dt);
				maximum_residual_k_current_global = calculate_maximum_error_change.parallel_exec(dt);

				update_regularization_global_variation.parallel_exec(dt);
				averaged_variation_current_local = calculate_regularization_local_variation.parallel_exec(dt);
				averaged_variation_current_global = calculate_regularization_global_variation.parallel_exec(dt);
				maximum_variation_current_global = calculate_maximum_variation.parallel_exec(dt);

				out_file_error_plot << std::fixed << std::setprecision(12) << ite << "   " <<
					averaged_residual_T_current_global << "   " << maximum_residual_T_current_global << "   " <<
					averaged_residual_k_current_global << "   " << maximum_residual_k_current_global << "   " <<
					averaged_variation_current_global << "   " << maximum_variation_current_global << "\n";
				out_file_all_information << std::fixed << std::setprecision(12) << ite_loop << "   " << ite << "   " <<
					averaged_residual_T_current_local << "   " << averaged_residual_T_current_global << "   " << averaged_residual_k_current_local << "   " <<
					averaged_residual_k_current_global << "   " << averaged_variation_current_local << "   " << averaged_variation_current_global << "\n";

				//write_states.writeToFile(ite);
			}
		}
		copy_optimized_temperature.parallel_exec(); 
		ite_k = 0; ite_rg = 0; if (ite_loop % ite_output == 0) { write_states.writeToFile(ite); }
		std::cout << "N=" << ite << " and the k splitting is finished." << "\n";

		//----------------------------------------------------------------------
		//	Temperature splitting.
		//----------------------------------------------------------------------
		out_file_all_information << "This is the step of temperature splitting." << "\n";
		std::cout << "averaged_residual_T_last_global is " << averaged_residual_T_last_global << std::endl;
		while ((((averaged_residual_T_current_global > averaged_residual_T_last_global) || 
			     (maximum_residual_T_current_global > maximum_residual_T_last_global)) &&
			      averaged_residual_T_current_global > 0.000005) || ite_T < ite_T_total)
		{
			ite++; ite_T++; ite_T_comp_opt++;
			temperature_splitting_pde_complex.parallel_exec(dt);
			update_temperature_pde_residual.parallel_exec(dt);
			averaged_residual_T_current_local = calculate_temperature_local_residual.parallel_exec(dt);
			averaged_residual_T_current_global = calculate_temperature_global_residual.parallel_exec(dt);
			maximum_residual_T_current_global = calculate_maximum_error.parallel_exec(dt);

			if (ite_T == 1)
			{
				out_file_error_plot << std::fixed << std::setprecision(12) << ite << "   " <<
					averaged_residual_T_current_global << "   " << maximum_residual_T_current_global << "   " <<
					averaged_residual_k_current_global << "   " << maximum_residual_k_current_global << "   " <<
					averaged_variation_current_global << "   " << maximum_variation_current_global << "\n";
				out_file_all_information << std::fixed << std::setprecision(12) << ite_loop << "   " << ite << "   " <<
					averaged_residual_T_current_local << "   " << averaged_residual_T_current_global << "   " << averaged_residual_k_current_local << "   " <<
					averaged_residual_k_current_global << "   " << averaged_variation_current_local << "   " << averaged_variation_current_global << "\n";
				out_file_residual_comp << std::fixed << std::setprecision(12) << ite_T_comp_opt << "   " << averaged_residual_T_current_global << "\n";
			}
		}
		out_file_error_plot << std::fixed << std::setprecision(12) << ite << "   " <<
			averaged_residual_T_current_global << "   " << maximum_residual_T_current_global << "   " <<
			averaged_residual_k_current_global << "   " << maximum_residual_k_current_global << "   " <<
			averaged_variation_current_global << "   " << maximum_variation_current_global << "\n";
		out_file_all_information << std::fixed << std::setprecision(12) << ite_loop << "   " << ite << "   " <<
			averaged_residual_T_current_local << "   " << averaged_residual_T_current_global << "   " << averaged_residual_k_current_local << "   " <<
			averaged_residual_k_current_global << "   " << averaged_variation_current_local << "   " << averaged_variation_current_global << "\n";
		out_file_residual_comp << std::fixed << std::setprecision(12) << ite_T_comp_opt << "   " << averaged_residual_T_current_global << "\n";

		opt_averaged_temperature = calculate_averaged_opt_temperature.parallel_exec(dt);
		out_file_opt_temperature << std::fixed << std::setprecision(12) << ite_T_comp_opt << "   " << opt_averaged_temperature << "\n";
		out_file_nonopt_temperature << std::fixed << std::setprecision(12) << ite << "   " << opt_averaged_temperature << "\n";

		/* Decay learning rate by learning process. */
		if (nonopt_averaged_temperature > opt_averaged_temperature)
		{
			learning_rate_alpha = learning_rate_alpha;
			std::cout << "The learning rate is fixed!" << std::endl;
		}
		else if (nonopt_averaged_temperature < opt_averaged_temperature)
		{
			learning_rate_alpha = 0.95 * learning_rate_alpha;
			std::cout << "The learning rate is decreased by optimization process!" << std::endl;
		 }

		/* Decay by nature decay property. */
		/*learning_rate_alpha = initial_learning_rate * pow(0.9, ite_loop / 10);
		current_eta_regularization = initial_eta_regularization * pow(0.9, ite_loop / 10);*/

		nonopt_averaged_temperature = opt_averaged_temperature;

		averaged_residual_T_last_global = averaged_residual_T_current_global;
		maximum_residual_T_last_global = maximum_residual_T_current_global;

		std::cout << "averaged_residual_T_current_global is " << averaged_residual_T_current_global << std::endl;
		out_file_ite_T << std::fixed << std::setprecision(12) << ite_loop << "   " << ite_T << "\n";
		ite_T = 0; write_states.writeToFile(ite);
		std::cout << "N=" << ite << " and the temperature splitting is finished." << "\n";

		//----------------------------------------------------------------------
		//	Decision Making.
		//----------------------------------------------------------------------
		last_averaged_temperature = current_averaged_temperature;
		current_averaged_temperature = calculate_averaged_opt_temperature.parallel_exec();

		ite_loop++; std::cout << "This is the " << ite_loop << " iteration loop and the averaged temperature is " << opt_averaged_temperature
			<< " and the learning rate is " << learning_rate_alpha
			<< " and the regularization is " << current_eta_regularization << endl;
		relative_difference = abs(current_averaged_temperature - last_averaged_temperature) / last_averaged_temperature;
		out_file_value_RG << std::fixed << std::setprecision(12) << ite_loop << "   " << current_eta_regularization << "\n";

		if (ite_loop % ite_restart == 0) { restart_io.writeToFile(ite_loop); }
	}
	out_file_all_information.close();
	out_file_error_plot.close();
	out_file_learning_rate.close();
	out_file_value_RG.close();
	out_file_opt_temperature.close();
	out_file_nonopt_temperature.close();
	out_file_ite_T.close();
	out_file_residual_comp.close();

	tick_count t2 = tick_count::now();
	tick_count::interval_t tt;
	tt = t2 - t1;
	std::cout << "Total time for optimization: " << tt.seconds() << " seconds." << std::endl;
	return 0;
};
