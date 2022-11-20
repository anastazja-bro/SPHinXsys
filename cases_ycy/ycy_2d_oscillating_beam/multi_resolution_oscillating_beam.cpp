/* ---------------------------------------------------------------------------*
*            SPHinXsys: 2D oscillation beam example-one body version           *
* ----------------------------------------------------------------------------*
* This is the one of the basic test cases, also the first case for            *
* understanding SPH method for solid simulation.                              *
* In this case, the constraint of the beam is implemented with                *
* internal constrained subregion.                                             *
* ----------------------------------------------------------------------------*/
/**
  * @brief 	SPHinXsys Library.
  */
#include "sphinxsys.h"
/**
 * @brief Namespace cite here.
 */
using namespace SPH;

//------------------------------------------------------------------------------
//global parameters for the case
//------------------------------------------------------------------------------

//for geometry
Real PL = 0.2; 	//beam length
Real PH = 0.02; //for thick plate; =0.01 for thin plate
Real SL = 0.06; //depth of the insert
//particle spacing, at least three particles
Real resolution_ref = PH / 5.0;
Real BW = resolution_ref * 4; 	//boundary width
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-SL - BW, -PL / 2.0),
	Vec2d(PL + 3.0 * BW, PL / 2.0));

//for material properties of the beam
Real rho0_s = 1.0e3; 			//reference density
Real Youngs_modulus = 2.0e6;	//reference Youngs modulus
Real poisson = 0.3975; 			//Poisson ratio

//for initial condition on velocity
Real kl = 1.875;
Real M = sin(kl) + sinh(kl);
Real N = cos(kl) + cosh(kl);
Real Q = 2.0 * (cos(kl)*sinh(kl) - sin(kl)*cosh(kl));
Real vf = 0.05;
Real R = PL / (0.5 * Pi);

/**
* @brief define geometry and initial conditions of SPH bodies
*/
/**
* @brief create a beam base shape
*/
std::vector<Vecd> createBeamBaseShape()
{
	//geometry
	std::vector<Vecd> beam_base_shape;
	beam_base_shape.push_back(Vecd(-SL - BW, -PH / 2 - BW));
	beam_base_shape.push_back(Vecd(-SL - BW, PH / 2 + BW));
	beam_base_shape.push_back(Vecd(0.0, PH / 2 + BW));
	beam_base_shape.push_back(Vecd(0.0, -PH / 2 - BW));
	beam_base_shape.push_back(Vecd(-SL - BW, -PH / 2 - BW));
	
	return beam_base_shape;
}
/**
* @brief create a beam shape
*/
std::vector<Vecd> createBeamShape()
{
	std::vector<Vecd> beam_shape;
	beam_shape.push_back(Vecd(-SL, -PH / 2));
	beam_shape.push_back(Vecd(-SL, PH / 2));
	beam_shape.push_back(Vecd(PL, PH / 2));
	beam_shape.push_back(Vecd(PL, -PH / 2));
	beam_shape.push_back(Vecd(-SL, -PH / 2));

	return beam_shape;
}
/**
* @brief create the points need to be densed
*/
std::vector<Vecd> creatDensedPoints()
{
	std::vector<Vecd> densed_points;
	densed_points.push_back(Vecd(0.0, 0.01));
	densed_points.push_back(Vecd(0.0, -0.01));
	//densed_points.push_back(Vecd(-SL-BW, 0.00));
	return densed_points;
}
/**
* @brief create the lines need to be densed
*/
StdVec<std::pair<Vecd, Vecd>> creatDensedLines()
{
	StdVec<std::pair<Vecd, Vecd>> densed_lines;
	densed_lines.push_back(std::make_pair(Vecd(0.0, 0.01),Vecd(0.2, 0.01)));
	densed_lines.push_back(std::make_pair(Vecd(0.0, -0.01), Vecd(0.2, -0.01)));
	//densed_points.push_back(Vecd(-SL-BW, 0.00));
	return densed_lines;
}
/**
* @brief define the beam body
*/
class Beam : public SolidBody
{
public:
	Beam(SPHSystem &system, const std::string &body_name)
		: SolidBody(system, body_name,
			//makeShared<ParticleSpacingByBodyShape>(1.15, 1.0, 3),
			//makeShared<ParticleSpacingByPositions>(1.15, 1.0, 2, 3, creatDensedPoints()),
			makeShared<ParticleSpacingByLines>(1.15, 1.0, 2, 3, creatDensedLines()),
			//makeShared<ParticleGeneratorMultiResolution>())
			//makeShared<ParticleGeneratorMultiResolutionByPosition>())
			makeShared<ParticleGeneratorMultiResolutionByLines>())
	{
		/** Geometry definition. */
		MultiPolygon multi_polygon;
		multi_polygon.addAPolygon(createBeamBaseShape(), ShapeBooleanOps::add);
		multi_polygon.addAPolygon(createBeamShape(), ShapeBooleanOps::add);
		MultiPolygonShape multi_polygon_shape(multi_polygon);
		body_shape_.add<LevelSetShape>(this, multi_polygon_shape, false);
	}
};

/**
 * application dependent initial condition 
 */
class BeamInitialCondition
	: public solid_dynamics::ElasticDynamicsInitialCondition
{
public:
	explicit BeamInitialCondition(SolidBody &beam)
		: solid_dynamics::ElasticDynamicsInitialCondition(beam) {};
protected:
	void Update(size_t index_i, Real dt) override {
		/** initial velocity profile */
		Real x = pos_n_[index_i][0] / PL;
		if (x > 0.0) {
			vel_n_[index_i][1] 
				= vf * material_->ReferenceSoundSpeed()*(M*(cos(kl*x) - cosh(kl*x)) - N * (sin(kl*x) - sinh(kl*x))) / Q;
		}
	};
};
/**
* @brief define the beam base which will be constrained.
* NOTE: this class can only be instanced after body particles
* have been generated
*/
class BeamBase : public BodyPartByParticle
{
public:
	BeamBase(SolidBody &solid_body, const std::string &constrained_region_name)
		: BodyPartByParticle(solid_body, constrained_region_name)
	{
		/* Geometry definition */
		MultiPolygon multi_polygon;
		multi_polygon.addAPolygon(createBeamBaseShape(), ShapeBooleanOps::add);
		multi_polygon.addAPolygon(createBeamShape(), ShapeBooleanOps::sub);
		body_part_shape_ = shape_ptr_keeper_.createPtr<MultiPolygonShape>(multi_polygon);
		//tag the particles within the body part
		tagBodyPart();
	}
};

//define an observer body
class BeamObserver : public FictitiousBody
{
public:
	BeamObserver(SPHSystem &system, const std::string &body_name)
		: FictitiousBody(system, body_name, makeShared<ParticleAdaptation>(1.15, 2.0))
	{
		body_input_points_volumes_.push_back(std::make_pair(Vecd(PL, 0.0), 0.0));
	}
};
//------------------------------------------------------------------------------
//the main program
//------------------------------------------------------------------------------

int main(int ac, char *av[])
{
	//build up context -- a SPHSystem
	SPHSystem system(system_domain_bounds, resolution_ref);
	system.run_particle_relaxation_ = false;
	system.reload_particles_ = true;
	system.restart_step_ = 0;
#ifdef BOOST_AVAILABLE
	system.handleCommandlineOptions(ac, av);
#endif

	//-----------------------------------------------------------------------------
	//outputs
	//-----------------------------------------------------------------------------
	In_Output in_output(system);

	//the oscillating beam
	Beam beam_body(system, "BeamBody");
	if (!system.run_particle_relaxation_ && system.reload_particles_)
	{
		beam_body.replaceParticleGenerator<ParticleGeneratorReload>(in_output, beam_body.getBodyName());
	}
	//creat particles for the elastic body
	ElasticSolidParticles beam_particles(beam_body, makeShared<LinearElasticSolid>(rho0_s, Youngs_modulus, poisson));
	beam_particles.addAVariableToWrite<indexScalar, Real>("Volume");
	BeamObserver beam_observer(system, "BeamObserver");
	//create observer particles
	BaseParticles observer_particles(beam_observer);
	if (system.run_particle_relaxation_)
	{
		BodyRelationInnerVariableSmoothingLength beam_inner(beam_body);
		BodyStatesRecordingToPlt beam_recording_to_vtp(in_output, system.real_bodies_);
		RandomizePartilePosition random_beam_particles(beam_body);
		relax_dynamics::RelaxationStepInner beam_relaxation_step_inner(beam_inner);
		//relax_dynamics::UpdateSmoothingLengthRatioByBodyShape update_smoothing_length_ratio(beam_body);
		//relax_dynamics::UpdateSmoothingLengthRatioByPositions update_smoothing_length_ratio(beam_body);
		relax_dynamics::UpdateSmoothingLengthRatioByLines update_smoothing_length_ratio(beam_body);
		ReloadParticleIO write_particle_reload_files(in_output, system.real_bodies_);
		random_beam_particles.parallel_exec(0.25);
		beam_relaxation_step_inner.surface_bounding_.parallel_exec();
		update_smoothing_length_ratio.parallel_exec();
		beam_body.updateCellLinkedList();
		beam_recording_to_vtp.writeToFile(0.0);

		int ite_p = 0;
		while (ite_p < 2000)
		{
			update_smoothing_length_ratio.parallel_exec();
			beam_relaxation_step_inner.parallel_exec();
			ite_p += 1;
			if (ite_p % 100 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the beam N = " << ite_p << "\n";
				beam_recording_to_vtp.writeToFile(ite_p);
			}
		}
		std::cout << "The physics relaxation process of beam finish !" << std::endl;
		write_particle_reload_files.writeToFile(0);
		return 0;
	}
	/** topology */
	BodyRelationInnerVariableSmoothingLength beam_body_inner(beam_body);

	std::string output_folder_ = "./output";
	std::string filefullpath = output_folder_ + "/" + "check_particle_infomation_" + ".dat";
	std::ofstream out_file_(filefullpath.c_str(), std::ios::app);
	out_file_ << "\n";
	out_file_ << "position_x, " << "position_y, " << "index_i, " << "volume, " << "smoothing_length_ratio, " << endl;
	for (size_t i = 0; i != beam_particles.total_real_particles_; i++) {
		out_file_  << beam_particles.pos_n_[i][0] << "  " << beam_particles.pos_n_[i][1] << "  " << i << "  "
			<< beam_particles.Vol_[i] <<" "<< beam_body .particle_adaptation_->SmoothingLengthRatio(i)<< endl;
	}
	

	//BodyRelationContact beam_observer_contact(beam_observer, { &beam_body });

	//-----------------------------------------------------------------------------
	//this section define all numerical methods will be used in this case
	//-----------------------------------------------------------------------------
	/** initial condition */
	BeamInitialCondition beam_initial_velocity(beam_body);
	//corrected strong configuration	
	solid_dynamics::CorrectConfiguration
		beam_corrected_configuration_in_strong_form(beam_body_inner);

	//time step size calculation
	solid_dynamics::AcousticTimeStepSize computing_time_step_size(beam_body);

	//stress relaxation for the beam
	solid_dynamics::KirchhoffStressRelaxationFirstHalf
		stress_relaxation_first_half(beam_body_inner);
	solid_dynamics::StressRelaxationSecondHalf
		stress_relaxation_second_half(beam_body_inner);

	// clamping a solid body part. This is softer than a driect constraint
	BeamBase beam_base(beam_body, "BeamBase");
	solid_dynamics::ClampConstrainSolidBodyRegion clamp_constrain_beam_base(beam_body_inner, beam_base);

	
	BodyStatesRecordingToPlt write_beam_states(in_output, system.real_bodies_);
	/*RegressionTestEnsembleAveraged<ObservedQuantityRecording<indexVector, Vecd>>
		write_beam_tip_displacement("Position", in_output, beam_observer_contact);*/

	/**
	 * @brief Setup geometry and initial conditions
	 */
	system.initializeSystemCellLinkedLists();
	system.initializeSystemConfigurations();
	beam_initial_velocity.exec();
	beam_corrected_configuration_in_strong_form.parallel_exec();

	//-----------------------------------------------------------------------------
	//from here the time stepping begines
	//-----------------------------------------------------------------------------
	//starting time zero
	GlobalStaticVariables::physical_time_ = 0.0;
	write_beam_states.writeToFile(0);
	//write_beam_tip_displacement.writeToFile(0);

	int ite = 0;
	Real T0 = 2.0;
	Real End_Time = T0;
	//time step size for output file
	Real D_Time = 0.01*T0;
	Real Dt = 0.1*D_Time;			/**< Time period for data observing */
	Real dt = 0.0; 					//default acoustic time step sizes

	//statistics for computing time
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;

	//computation loop starts 
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real integration_time = 0.0;
		//integrate time (loop) until the next output time
		while (integration_time < D_Time) {

			Real relaxation_time = 0.0;
			while (relaxation_time < Dt) {
				stress_relaxation_first_half.exec(dt);
				clamp_constrain_beam_base.exec();
				stress_relaxation_second_half.exec(dt);

				ite++;
				dt = computing_time_step_size.exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;

				if (ite % 100 == 0) {
					std::cout << "N=" << ite << " Time: "
						<< GlobalStaticVariables::physical_time_ << "	dt: "
						<< dt << "\n";
				}
			}
		}

		//write_beam_tip_displacement.writeToFile(ite);

		tick_count t2 = tick_count::now();
		write_beam_states.writeToFile();
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
