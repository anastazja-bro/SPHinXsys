/* ---------------------------------------------------------------------------*
*            SPHinXsys: 2D oscillation beam example-one body version           *
* ----------------------------------------------------------------------------*
* This is the one of the basic test cases, also the first case for            *
* understanding SPH method for solid simulation.                              *
* In this case, the constraint of the beam is implemented with                *
* internal constrained subregion.                                             *
* ----------------------------------------------------------------------------*/
#include "sphinxsys.h"
using namespace SPH;
//------------------------------------------------------------------------------
//global parameters for the case
//------------------------------------------------------------------------------
Real PL = 0.2;	//beam length
Real PH = 0.02; //for thick plate; =0.01 for thin plate
Real SL = 0.06; //depth of the insert
//reference particle spacing
Real resolution_ref = PH / 5.0;
Real BW = resolution_ref * 4; //boundary width, at least three particles
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-SL - BW, -PL / 2.0),
								 Vec2d(PL + 3.0 * BW, PL / 2.0));
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_s = 1.0e3;		 //reference density
Real Youngs_modulus = 2.0e6; //reference Youngs modulus
Real poisson = 0.3975;		 //Poisson ratio
//----------------------------------------------------------------------
//	Parameters for initial condition on velocity
//----------------------------------------------------------------------
Real kl = 1.875;
Real M = sin(kl) + sinh(kl);
Real N = cos(kl) + cosh(kl);
Real Q = 2.0 * (cos(kl) * sinh(kl) - sin(kl) * cosh(kl));
Real vf = 0.05;
Real R = PL / (0.5 * Pi);
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
// a beam base shape
std::vector<Vecd> beam_base_shape{
	Vecd(-SL - BW, -PH / 2 - BW), Vecd(-SL - BW, PH / 2 + BW), Vecd(0.0, PH / 2 + BW),
	Vecd(0.0, -PH / 2 - BW), Vecd(-SL - BW, -PH / 2 - BW)};
// a beam shape
std::vector<Vecd> beam_shape{
	Vecd(-SL, -PH / 2), Vecd(-SL, PH / 2), Vecd(PL, PH / 2), Vecd(PL, -PH / 2), Vecd(-SL, -PH / 2)};
//Beam observer location
StdVec<Vecd> observation_location = {Vecd(PL, 0.0)};
//Densed Points
StdVec<Vecd> densed_points{Vecd(0.0, 0.01),Vecd(0.0, -0.01)};
//Densed Lines
StdVec<std::pair<Vecd, Vecd>> densed_lines{ std::make_pair(Vecd(0.0, 0.01), Vecd(0.2, 0.01)),
	std::make_pair(Vecd(0.0, -0.01), Vecd(0.2, -0.01)) };
//----------------------------------------------------------------------
//	Define the beam body
//----------------------------------------------------------------------
class Beam : public MultiPolygonShape
{
public:
	explicit Beam(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(beam_base_shape, ShapeBooleanOps::add);
		multi_polygon_.addAPolygon(beam_shape, ShapeBooleanOps::add);
	}
};
//----------------------------------------------------------------------
//	application dependent initial condition 
//----------------------------------------------------------------------
//class BeamInitialCondition
//	: public solid_dynamics::ElasticDynamicsInitialCondition
//{
//public:
//	explicit BeamInitialCondition(SPHBody& sph_body)
//		: solid_dynamics::ElasticDynamicsInitialCondition(sph_body){};
//
//	void Update(size_t index_i, Real dt) 
//	{
//		/** initial velocity profile */
//		Real x = pos_[index_i][0] / PL;
//		if (x > 0.0)
//		{
//			vel_[index_i][1] = vf * material_->ReferenceSoundSpeed() *
//								 (M * (cos(kl * x) - cosh(kl * x)) - N * (sin(kl * x) - sinh(kl * x))) / Q;
//		}
//	};
//};

class BeamInitialCondition
	: public solid_dynamics::ElasticDynamicsInitialCondition
{
public:
	explicit BeamInitialCondition(SPHBody& sph_body)
		: solid_dynamics::ElasticDynamicsInitialCondition(sph_body) {};

	void update(size_t index_i, Real dt)
	{
		/** initial velocity profile */
		Real x = pos_[index_i][0] / PL;
		if (x > 0.0)
		{
			vel_[index_i][1] = vf * material_->ReferenceSoundSpeed() *
				(M * (cos(kl * x) - cosh(kl * x)) - N * (sin(kl * x) - sinh(kl * x))) / Q;
		}
	};
};

//----------------------------------------------------------------------
//	define the beam base which will be constrained.
//----------------------------------------------------------------------
MultiPolygon createBeamConstrainShape()
{
	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(beam_base_shape, ShapeBooleanOps::add);
	multi_polygon.addAPolygon(beam_shape, ShapeBooleanOps::sub);
	return multi_polygon;
};
//------------------------------------------------------------------------------
//the main program
//------------------------------------------------------------------------------
int main(int ac, char *av[])
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	SPHSystem system(system_domain_bounds, resolution_ref);
	/** Tag for run particle relaxation for the initial body fitted distribution. */
	system.run_particle_relaxation_ = true;
	/** Tag for computation start with relaxed body fitted particles distribution. */
	system.reload_particles_ = false;
	//handle command line arguments
#ifdef BOOST_AVAILABLE
	system.handleCommandlineOptions(ac, av);
#endif
	/** output environment. */
	IOEnvironment io_environment(system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	SolidBody beam_body(system, makeShared<Beam>("BeamBody"));
	//beam_body.defineAdaptation<ParticleSpacingByBodyShape>(1.15, 1.0, 3);
	beam_body.defineAdaptation<ParticleSpacingByPositions>(1.15, 1.0, 2, 3, densed_points);
	beam_body.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(beam_body);
	beam_body.defineParticlesAndMaterial<ElasticSolidParticles, LinearElasticSolid>(rho0_s, Youngs_modulus, poisson);
	
	//(!system.run_particle_relaxation_ && system.reload_particles_)
	//	? beam_body.generateParticles<ParticleGeneratorReload>(io_environment, beam_body.getName())
	//	//: beam_body.generateParticles<ParticleGeneratorMultiResolution>();
	//	: beam_body.generateParticles<ParticleGeneratorMultiResolutionByPosition>();
	//beam_body.addBodyStateForRecording<Real>("SmoothingLengthRatio");
	
	if (!system.run_particle_relaxation_ && system.reload_particles_)
	{
		beam_body.generateParticles<ParticleGeneratorReload>(io_environment, beam_body.getName());
		//beam_body.sph_adaptation_->registerSmoothingLengthRatio(beam_body.base_particles_);
	}
	else {
		beam_body.generateParticles<ParticleGeneratorMultiResolutionByPosition>();
		beam_body.addBodyStateForRecording<Real>("SmoothingLengthRatio");
		beam_body.base_particles_->addVariableToReload<Real>("SmoothingLengthRatio");
	}
	
	//----------------------------------------------------------------------
	//	Define simple file input and outputs functions.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp beam_body_recording_to_vtp(io_environment, { &beam_body });
	MeshRecordingToPlt cell_linked_list_recording(io_environment, beam_body, beam_body.cell_linked_list_);

	//----------------------------------------------------------------------
	//	Define observer.
	//----------------------------------------------------------------------
	//ObserverBody beam_observer(system, "BeamObserver");
	//beam_observer.sph_adaptation_->resetAdaptationRatios(1.15, 2.0);
	//beam_observer.generateParticles<ObserverParticleGenerator>(observation_location);
	
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInnerVariableSmoothingLength beam_body_inner(beam_body);
	//BodyRelationContact beam_observer_contact(beam_observer, {&beam_body});
	

	//-----------------------------------------------------------------------------
	// Run particle relaxation for body-fitted distribution if chosen.
	//-----------------------------------------------------------------------------
	if (system.run_particle_relaxation_)
	{
		
		//----------------------------------------------------------------------
		//	Methods used for particle relaxation.
		//----------------------------------------------------------------------
		/** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_beam_particles(beam_body);
		relax_dynamics::RelaxationStepInner beam_relaxation_step_inner(beam_body_inner);
		SimpleDynamics<relax_dynamics::UpdateSmoothingLengthRatioByPositions> update_smoothing_length_ratio(beam_body);
		//SimpleDynamics<relax_dynamics::UpdateSmoothingLengthRatioByBodyShape> update_smoothing_length_ratio(beam_body);
		//SimpleDynamics<relax_dynamics::UpdateSmoothingLengthRatioByLines> update_smoothing_length_ratio(beam_body);

		/** Write the particle reload files. */
		ReloadParticleIO write_particle_reload_files(io_environment, system.real_bodies_);
		random_beam_particles.parallel_exec(0.25);
		beam_relaxation_step_inner.surface_bounding_.parallel_exec();
		update_smoothing_length_ratio.parallel_exec();
		beam_body.updateCellLinkedList();
		beam_body_recording_to_vtp.writeToFile(0.0);

		int ite_p = 0;
		while (ite_p < 2000)
		{
			update_smoothing_length_ratio.parallel_exec();
			beam_relaxation_step_inner.parallel_exec();
			ite_p += 1;
			if (ite_p % 100 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the beam N = " << ite_p << "\n";
				beam_body_recording_to_vtp.writeToFile(ite_p);
			}
		}
		std::cout << "The physics relaxation process of beam finish !" << std::endl;
		write_particle_reload_files.writeToFile(0);
		return 0;
	}

	//-----------------------------------------------------------------------------
	//this section define all numerical methods will be used in this case
	//-----------------------------------------------------------------------------
	/** initial condition */
	SimpleDynamics<BeamInitialCondition> beam_initial_velocity(beam_body);
	//BeamInitialCondition beam_initial_velocity(beam_body);
	//corrected strong configuration
	InteractionDynamics<solid_dynamics::CorrectConfiguration> beam_corrected_configuration(beam_body_inner);
	//time step size calculation
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> computing_time_step_size(beam_body);
	//stress relaxation for the beam
	Dynamics1Level<solid_dynamics::StressRelaxationFirstHalf> stress_relaxation_first_half(beam_body_inner);
	Dynamics1Level<solid_dynamics::StressRelaxationSecondHalf> stress_relaxation_second_half(beam_body_inner);
	// clamping a solid body part. This is softer than a driect constraint
	//solid_dynamics::ClampConstrainSolidBodyRegion clamp_constrain_beam_base(beam_body_inner, beam_base);
	BodyRegionByParticle beam_base(beam_body, makeShared<MultiPolygonShape>(createBeamConstrainShape()));
	SimpleDynamics<solid_dynamics::FixConstraint, BodyRegionByParticle> constraint_beam_base(beam_base);
	
	//BodyStatesRecordingToVtp write_beam_states(in_output, system.real_bodies_);
	//RegressionTestEnsembleAveraged<ObservedQuantityRecording<Vecd>>
		//write_beam_tip_displacement("Position", in_output, beam_observer_contact);
	
	//----------------------------------------------------------------------
	//	Setup computing and initial conditions.
	//----------------------------------------------------------------------
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	beam_initial_velocity.exec();
	beam_corrected_configuration.parallel_exec();
	//----------------------------------------------------------------------
	//	Setup computing time-step controls.
	//----------------------------------------------------------------------
	int ite = 0;
	Real T0 = 1.0;
	Real End_Time = T0;
	//time step size for output file
	Real output_interval = 0.01 * T0;
	Real Dt = 0.1 * output_interval; /**< Time period for data observing */
	Real dt = 0.0;			//default acoustic time step sizes

	//statistics for computing time
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//-----------------------------------------------------------------------------
	//from here the time stepping begines
	//-----------------------------------------------------------------------------
	//write_beam_states.writeToFile(0);
	//write_beam_tip_displacement.writeToFile(0);

	//computation loop starts
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real integration_time = 0.0;
		//integrate time (loop) until the next output time
		while (integration_time < output_interval)
		{

			Real relaxation_time = 0.0;
			while (relaxation_time < Dt)
			{
				stress_relaxation_first_half.parallel_exec(dt);
				//clamp_constrain_beam_base.parallel_exec();
				constraint_beam_base.parallel_exec();
				stress_relaxation_second_half.parallel_exec(dt);

				ite++;
				dt = computing_time_step_size.parallel_exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;

				if (ite % 100 == 0)
				{
					std::cout << "N=" << ite << " Time: "
							  << GlobalStaticVariables::physical_time_ << "	dt: "
							  << dt << "\n";
				}
			}
		}

		//write_beam_tip_displacement.writeToFile(ite);

		tick_count t2 = tick_count::now();
		//write_beam_states.writeToFile();
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	//write_beam_tip_displacement.newResultTest();

	return 0;
}
