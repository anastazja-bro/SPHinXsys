/**
 * @file 	3d_elastic_shell_with_relaxed_particles.cpp
 * @brief 	test deformation of elastic shell with relaxed particles
 * @details  
 * @author 	Anyong Zhang
 */
#include "sphinxsys.h" //SPHinXsys Library.
using namespace SPH;   // Namespace cite here.
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real resolution_ref = 2e-4;	//						/**< reference resolution. */
Real thickness = 3e-5; //				/**< thickness. */
Real level_set_refinement_ratio = 5;
Real width = 0.02;
Real length = 0.08;

BoundingBox system_domain_bounds(Vec3d(-0.001, -0.005, 0.0), Vec3d(0.02, 0.005, 3.0e-05));
//----------------------------------------------------------------------
//	Global paramters on material properties
//----------------------------------------------------------------------
Real rho0_s = 1.0e3;
Real Youngs_modulus = 1.e6;
Real poisson = 0.3;
Real physical_viscosity = 1.0e3;

Real gravity_g = 1;

Real end_time = 0.1;
Real increament = 0.2 * end_time;

std::string full_path_to_geometry = "./input/trapezoid_plane.stl";

//----------------------------------------------------------------------
class ImportedShellModel :public ComplexShape
{
public:
	explicit ImportedShellModel(const std::string& shape_name)
		:ComplexShape(shape_name)
	{
		add<TriangleMeshShapeSTL>(full_path_to_geometry, Vecd(0), 0.001);
	}
};

/** Define the boundary geometry. */
class BoundaryGeometry : public BodyPartByParticle
{
public:
	BoundaryGeometry(SPHBody& body, const std::string& body_part_name)
		: BodyPartByParticle(body, body_part_name)
	{
		TaggingParticleMethod tagging_particle_method = std::bind(&BoundaryGeometry::tagManually, this, _1);
		tagParticles(tagging_particle_method);
	};
	virtual ~BoundaryGeometry() {};

private:
	void tagManually(size_t index_i)
	{
		if (base_particles_->pos_[index_i][0] < 0)
		{
			body_part_particles_.push_back(index_i);
		}
	};
};

/**
 * define time dependent external force
 */
class TimeDependentExternalForce : public Gravity
{
public:
	explicit TimeDependentExternalForce(Vecd external_force)
		: Gravity(external_force) {}
	virtual Vecd InducedAcceleration(Vecd& position) override
	{
		Real current_time = GlobalStaticVariables::physical_time_;
		return current_time < increament ? current_time * global_acceleration_ / increament : global_acceleration_;
	}
};


//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	/** Setup the system. */
	SPHSystem system(system_domain_bounds, resolution_ref);
	InOutput in_output(system);

	system.run_particle_relaxation_ = false;
	system.reload_particles_ = !system.run_particle_relaxation_;

	/** Create a Cylinder body. */
	SolidBody plane_shell_body(system, makeShared<ImportedShellModel>("PlaneShell"));
	plane_shell_body.defineBodyLevelSetShape(level_set_refinement_ratio)->writeLevelSet(plane_shell_body);
	plane_shell_body.defineParticlesAndMaterial<ShellParticles, LinearElasticSolid>(rho0_s, Youngs_modulus, poisson);
	plane_shell_body.addBodyStateForRecording<Vecd>("NormalDirection");

	if (system.reload_particles_)
	{
		plane_shell_body.generateParticles<ParticleGeneratorReload>(in_output, plane_shell_body.getBodyName());
	}
	else
	{
		plane_shell_body.generateParticles<ThickSurfaceParticleGeneratorLattice>(thickness);
	}
	std::cout << "Total particlel number : " << plane_shell_body.base_particles_->pos_.size() << std::endl;

	if (system.run_particle_relaxation_)
	{
		BodyRelationInner body_inner(plane_shell_body);
		RandomizeParticlePosition random_body_particles(plane_shell_body);
		ReloadParticleIO write_particle_reload_files(in_output, { &plane_shell_body });
		relax_dynamics::ShellRelaxationStepInner relaxation_step_inner(body_inner, thickness,
			level_set_refinement_ratio);
		relax_dynamics::ShellNormalDirectionPrediction normal_prediction(body_inner, thickness, 0.25);

		random_body_particles.parallel_exec(0.25);
		relaxation_step_inner.mid_surface_bounding_.parallel_exec();
		plane_shell_body.updateCellLinkedList();

		int ite_p = 0;
		while (ite_p < 1000)
		{
			if (ite_p % 100 == 0)
			{
				std::cout << std::fixed << std::setprecision(9)
					<< "Relaxation steps for the inserted body N = "
					<< ite_p << std::endl;
			}
			relaxation_step_inner.parallel_exec();
			++ite_p;
		}
		normal_prediction.exec();
		write_particle_reload_files.writeToFile(0);
		std::cout << "The physics relaxation process of imported model finish! \n";

		return 0;
	}


	BodyRelationInner plane_shell_body_inner(plane_shell_body);
	/** Common particle dynamics. */
	TimeDependentExternalForce external_force(Vecd(0.0, 0.0, -gravity_g));
	TimeStepInitialization initialize_external_force(plane_shell_body, external_force);

	/**
	 * This section define all numerical methods will be used in this case.
	 */
	 /** Corrected configuration. */
	thin_structure_dynamics::ShellCorrectConfiguration corrected_configuration(plane_shell_body_inner);
	/** Time step size calculation. */
	thin_structure_dynamics::ShellAcousticTimeStepSize computing_time_step_size(plane_shell_body, 0.001);
	/** stress relaxation. */
	thin_structure_dynamics::ShellStressRelaxationFirstHalf
		stress_relaxation_first_half(plane_shell_body_inner);
	thin_structure_dynamics::ShellStressRelaxationSecondHalf
		stress_relaxation_second_half(plane_shell_body_inner);

	/** Constrain the Boundary. */
	BoundaryGeometry boundary_geometry(plane_shell_body, "BoundaryGeometry");
	thin_structure_dynamics::ConstrainShellBodyRegion
		fixed_free_rotate_shell_boundary(plane_shell_body, boundary_geometry);

	// Damping
	DampingWithRandomChoice<DampingPairwiseInner<Vec3d>>
		cylinder_position_damping(0.2, plane_shell_body_inner, "Velocity", physical_viscosity);
	DampingWithRandomChoice<DampingPairwiseInner<Vec3d>>
		cylinder_rotation_damping(0.2, plane_shell_body_inner, "AngularVelocity", physical_viscosity);

	/** Output */
	BodyStatesRecordingToVtp write_states(in_output, system.real_bodies_);

	/** Apply initial condition. */
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	corrected_configuration.parallel_exec();

	/**
	 * From here the time stepping begins.
	 * Set the starting time.
	 */
	GlobalStaticVariables::physical_time_ = 0.0;
	write_states.writeToFile(0);

	/** Setup physical parameters. */
	int ite = 0;
	//Real end_time = 1.0;
	Real output_period = end_time / 100.0;
	Real dt = 0.0;
	/** Statistics for computing time. */
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	size_t output_ite = 0;
	/**
	 * Main loop
	 */
	while (GlobalStaticVariables::physical_time_ < end_time)
	{
		Real integral_time = 0.0;
		while (integral_time < output_period)
		{
			if (ite % 100 == 0)
			{
				std::cout << "N=" << ite << " Time: "
					<< GlobalStaticVariables::physical_time_ << "	dt: "
					<< dt << "\n";
			}
			initialize_external_force.parallel_exec(dt);

			Real time = GlobalStaticVariables::physical_time_;

			stress_relaxation_first_half.parallel_exec(dt);
			fixed_free_rotate_shell_boundary.parallel_exec(dt);
			cylinder_position_damping.parallel_exec(dt);
			cylinder_rotation_damping.parallel_exec(dt);
			fixed_free_rotate_shell_boundary.parallel_exec(dt);

			stress_relaxation_second_half.parallel_exec(dt);

			ite++;
			dt = computing_time_step_size.parallel_exec();
			integral_time += dt;
			GlobalStaticVariables::physical_time_ += dt;
		}
		tick_count t2 = tick_count::now();
		++output_ite;
		write_states.writeToFile(output_ite);
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() / 60. << " min." << std::endl;

	return 0;
}
