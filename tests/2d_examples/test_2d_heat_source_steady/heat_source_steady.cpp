/**
 * @file 	heat_source_steady.cpp
 * @brief 	This is the first test to demonstrate SPHInXsys as an optimization tool.
 * @details Consider a 2d block thermal domain with two constant temperature regions at the lower
 * 			and upper boundaries. The radiation-like source is distributed in the entire block domain.
 * 			The optimization target is to achieve lowest average temperature by modifying the distribution of
 * 			thermal diffusion rate in the domain with an extra conservation constraint that
 * 			the integral of the thermal diffusion rate in the entire domain is constant.
 * @author 	Bo Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" // using SPHinXsys library
using namespace SPH;   // Namespace cite here
//----------------------------------------------------------------------
//	Global geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;					 // inner domain length
Real H = 1.0;					 // inner domain height
Real resolution_ref = H / 100.0; // reference resolution for discretization
Real BW = resolution_ref * 2.0;	 // boundary width
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Global parameters for physics state variables.
//----------------------------------------------------------------------
std::string variable_name = "Phi";
std::string residue_name = "LaplacianResidue";
Real lower_temperature = 300.0;
Real upper_temperature = 350.0;
Real reference_temperature = upper_temperature - lower_temperature;
Real heat_source = 100.0;
Real target_strength = -200.0;
Real learning_strength_ref = 1.0;
//----------------------------------------------------------------------
//	Global parameters for material properties or coefficient variables.
//----------------------------------------------------------------------
std::string coefficient_name = "ThermalDiffusionRate";
std::string reference_coefficient = "ReferenceDiffusionCoefficient";
Real diffusion_coff = 1.0;
//----------------------------------------------------------------------
//	Geometric regions used in the system.
//----------------------------------------------------------------------
Vec2d block_halfsize = Vec2d(0.5 * L, 0.5 * H);					 // local center at origin
Vec2d block_translation = block_halfsize;						 // translation to global coordinates
Vec2d constraint_halfsize = Vec2d(0.05 * L, 0.5 * BW);			 // constraint block half size
Vec2d top_constraint_translation = Vec2d(0.5 * L, L + 0.5 * BW); // top constraint
Vec2d bottom_constraint_translation = Vec2d(0.5 * L, -0.5 * BW); // bottom constraint
class IsothermalBoundaries : public ComplexShape
{
public:
	explicit IsothermalBoundaries(const std::string &shape_name)
		: ComplexShape(shape_name)
	{
		add<TransformShape<GeometricShapeBox>>(Transform2d(top_constraint_translation), constraint_halfsize);
		add<TransformShape<GeometricShapeBox>>(Transform2d(bottom_constraint_translation), constraint_halfsize);
	}
};
//----------------------------------------------------------------------
//	Initial condition.
//----------------------------------------------------------------------
class DiffusionBodyInitialCondition : public ValueAssignment<Real>
{
public:
	explicit DiffusionBodyInitialCondition(SPHBody &diffusion_body)
		: ValueAssignment<Real>(diffusion_body, variable_name),
		  pos_(particles_->pos_){};
	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = 375.0 + 25.0 * (((double)rand() / (RAND_MAX)) - 0.5) * 2.0;
	};

protected:
	StdLargeVec<Vecd> &pos_;
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
		variable_[index_i] = pos_[index_i][1] > 0.5 ? lower_temperature : upper_temperature;
	}

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Initial coefficient distribution.
//----------------------------------------------------------------------
class DiffusionCoefficientDistribution : public ValueAssignment<Real>
{
public:
	explicit DiffusionCoefficientDistribution(SPHBody &diffusion_body)
		: ValueAssignment<Real>(diffusion_body, coefficient_name),
		  pos_(particles_->pos_){};
	void update(size_t index_i, Real dt)
	{
		variable_[index_i] = diffusion_coff;
	};

protected:
	StdLargeVec<Vecd> &pos_;
};
//----------------------------------------------------------------------
//	Coefficient reference for imposing coefficient evolution.
//----------------------------------------------------------------------
class ReferenceDiffusionCoefficient : public ValueAssignment<Real>
{
public:
	ReferenceDiffusionCoefficient(SPHBody &diffusion_body, const std::string &coefficient_name_ref)
		: ValueAssignment<Real>(diffusion_body, coefficient_name),
		  variable_ref(*particles_->template getVariableByName<Real>(coefficient_name_ref)){};
	void update(size_t index_i, Real dt)
	{
		variable_ref[index_i] = variable_[index_i];
	};

protected:
	StdLargeVec<Real> &variable_ref;
};
//----------------------------------------------------------------------
//	Equation residue to measure the solution convergence properties.
//----------------------------------------------------------------------
class ThermalEquationResidue
	: public OperatorWithBoundary<LaplacianInner<Real, CoefficientByParticle<Real>>,
								  LaplacianFromWall<Real, CoefficientByParticle<Real>>>

{
	Real source_;
	StdLargeVec<Real> &residue_;

public:
	ThermalEquationResidue(ComplexRelation &complex_relation,
						   const std::string &in_name, const std::string &out_name,
						   const std::string &eta_name, Real source)
		: OperatorWithBoundary<LaplacianInner<Real, CoefficientByParticle<Real>>,
							   LaplacianFromWall<Real, CoefficientByParticle<Real>>>(
			  complex_relation, in_name, out_name, eta_name),
		  residue_(base_operator_.OutVariable()), source_(source){};
	void interaction(size_t index_i, Real dt)
	{
		OperatorWithBoundary<
			LaplacianInner<Real, CoefficientByParticle<Real>>,
			LaplacianFromWall<Real, CoefficientByParticle<Real>>>::interaction(index_i, dt);
		residue_[index_i] += source_;
	};
};
//----------------------------------------------------------------------
//	Source term for impose optimization target.
//----------------------------------------------------------------------
class ImposingTargetSource : public LocalDynamics, public GeneralDataDelegateSimple
{
public:
	ImposingTargetSource(SPHBody &sph_body, const std::string &variable_name, const Real &source_strength)
		: LocalDynamics(sph_body), GeneralDataDelegateSimple(sph_body),
		  variable_(*particles_->getVariableByName<Real>(variable_name)),
		  source_strength_(source_strength){};
	virtual ~ImposingTargetSource(){};
	void setSourceStrength(Real source_strength) { source_strength_ = source_strength; };
	void update(size_t index_i, Real dt)
	{
		Real increment = source_strength_ * dt;
		Real theta = increment < 0.0 ? SMIN((0.01 + Eps - variable_[index_i]) / increment, 1.0) : 1.0;
		variable_[index_i] += increment * theta;
	};

protected:
	StdLargeVec<Real> &variable_;
	Real source_strength_;
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
								 Transform2d(block_translation), block_halfsize, "DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<SolidParticles, Solid>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();
	//----------------------------------------------------------------------
	//	add extra discrete variables (not defined in the library)
	//----------------------------------------------------------------------
	StdLargeVec<Real> body_temperature;
	diffusion_body.addBodyState<Real>(body_temperature, variable_name);
	diffusion_body.addBodyStateForRecording<Real>(variable_name);
	diffusion_body.addBodyStateToRestart<Real>(variable_name);
	StdLargeVec<Real> diffusion_coefficient;
	diffusion_body.addBodyState<Real>(diffusion_coefficient, coefficient_name);
	diffusion_body.addBodyStateForRecording<Real>(coefficient_name);
	diffusion_body.addBodyStateToRestart<Real>(coefficient_name);
	StdLargeVec<Real> laplacian_residue;
	diffusion_body.addBodyState<Real>(laplacian_residue, residue_name);
	diffusion_body.addBodyStateForRecording<Real>(residue_name);

	SolidBody isothermal_boundaries(sph_system, makeShared<IsothermalBoundaries>("IsothermalBoundaries"));
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
	SimpleDynamics<DiffusionBodyInitialCondition> diffusion_initial_condition(diffusion_body);
	SimpleDynamics<IsothermalBoundariesConstraints> boundary_constraint(isothermal_boundaries);
	SimpleDynamics<DiffusionCoefficientDistribution> coefficient_distribution(diffusion_body);
	SimpleDynamics<ConstraintTotalScalarAmount> constrain_total_coefficient(diffusion_body, coefficient_name);
	SimpleDynamics<ImposingSourceTerm<Real>> thermal_source(diffusion_body, variable_name, heat_source);
	SimpleDynamics<ImposingTargetSource> target_source(diffusion_body, coefficient_name, target_strength);
	InteractionDynamics<ThermalEquationResidue>
		thermal_equation_residue(diffusion_body_complex, variable_name, residue_name, coefficient_name, heat_source);
	ReduceDynamics<MaximumNorm<Real>> maximum_equation_residue(diffusion_body, residue_name);
	ReduceDynamics<QuantityMoment<Real>> total_coefficient(diffusion_body, coefficient_name);
	ReduceAverage<QuantitySummation<Real>> average_temperature(diffusion_body, variable_name);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	RestartIO restart_io(io_environment, sph_system.real_bodies_);
	/************************************************************************/
	/*            splitting thermal diffusivity optimization                */
	/************************************************************************/
	InteractionSplit<DampingSplittingWithWallCoefficientByParticle<Real>>
		implicit_heat_transfer_solver(diffusion_body_complex, variable_name, coefficient_name);
	InteractionWithUpdate<CoefficientEvolutionWithWallExplicit>
		coefficient_evolution_with_wall(diffusion_body_complex, variable_name, coefficient_name);
	SimpleDynamics<ReferenceDiffusionCoefficient>
		update_reference_coefficient(diffusion_body, "ReferenceDiffusionCoefficient");
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	diffusion_initial_condition.parallel_exec();
	boundary_constraint.parallel_exec();
	coefficient_distribution.parallel_exec();
	constrain_total_coefficient.setupInitialScalarAmount();
	thermal_equation_residue.parallel_exec();
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	int ite = 0;
	Real End_Time = 5.0;
	Real Observe_time = 0.01 * End_Time;
	Real dt = 1.0e-4;
	Real dt_coeff = SMIN(dt, 0.25 * resolution_ref * resolution_ref / reference_temperature);
	int k_ite = 10; // default number of iteration for imposing target
	Real allowed_equation_residue = 2.0e5;
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_states.writeToFile(ite);
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real relaxation_time = 0.0;
		Real equation_residue_max = Infinity; // initial value
		while (relaxation_time < Observe_time)
		{
			thermal_source.parallel_exec(dt);
			implicit_heat_transfer_solver.parallel_exec(dt);
			thermal_equation_residue.parallel_exec();
			Real residue_max_before_target = maximum_equation_residue.parallel_exec();

			update_reference_coefficient.parallel_exec();
			for (size_t k = 0; k != k_ite; ++k)
			{
				target_source.parallel_exec(dt_coeff);
				coefficient_evolution_with_wall.parallel_exec(dt_coeff);
				constrain_total_coefficient.parallel_exec();
			}

			thermal_equation_residue.parallel_exec();
			Real residue_max_after_target = maximum_equation_residue.parallel_exec();
			if (residue_max_after_target > equation_residue_max && residue_max_after_target > allowed_equation_residue)
			{
				k_ite = 0; // do not impose target next iteration step
			}
			else
			{
				k_ite = 10;
				equation_residue_max = residue_max_after_target;
			}

			ite++;
			relaxation_time += dt;
			GlobalStaticVariables::physical_time_ += dt;

			if (ite % 100 == 0)
			{
				std::cout << "N= " << ite << " Time: " << GlobalStaticVariables::physical_time_ << "	dt: " << dt << "\n";
				std::cout << "Total coefficient is " << total_coefficient.parallel_exec() << "\n";
				std::cout << "Average temperature is " << average_temperature.parallel_exec() << "\n";
				std::cout << "Thermal equation maximum residue is " << equation_residue_max << "\n";
			}
		}

		write_states.writeToFile();
	}

	std::cout << "The computation has finished, but the solution is still not steady yet." << std::endl;
	return 0;
}
