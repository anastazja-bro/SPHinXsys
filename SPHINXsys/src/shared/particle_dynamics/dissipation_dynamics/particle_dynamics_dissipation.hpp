#ifndef PARTICLE_DYNAMICS_DISSIPATION_HPP
#define PARTICLE_DYNAMICS_DISSIPATION_HPP

#include "particle_dynamics_dissipation.h"

namespace SPH
{
	//=================================================================================================//
	template <typename VariableType>
	DampingBySplittingInner<VariableType>::
		DampingBySplittingInner(BaseInnerRelation &inner_relation,
								const std::string &variable_name, Real eta)
		: LocalDynamics(inner_relation.sph_body_),
		  DissipationDataInner(inner_relation), eta_(eta),
		  Vol_(particles_->Vol_), mass_(particles_->mass_),
		  variable_(*particles_->getVariableByName<VariableType>(variable_name)) {}
	//=================================================================================================//
	template <typename VariableType>
	ErrorAndParameters<VariableType>
	DampingBySplittingInner<VariableType>::computeErrorAndParameters(size_t index_i, Real dt)
	{
		Real Vol_i = Vol_[index_i];
		Real mass_i = mass_[index_i];
		VariableType &variable_i = variable_[index_i];
		ErrorAndParameters<VariableType> error_and_parameters;
		Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];
			// linear projection
			VariableType variable_derivative = (variable_i - variable_[index_j]);
			Real parameter_b = 2.0 * eta_ * inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt / inner_neighborhood.r_ij_[n];

			error_and_parameters.error_ -= variable_derivative * parameter_b;
			error_and_parameters.a_ += parameter_b;
			error_and_parameters.c_ += parameter_b * parameter_b;
		}
		error_and_parameters.a_ -= mass_i;
		return error_and_parameters;
	}
	//=================================================================================================//
	template <typename VariableType>
	void DampingBySplittingInner<VariableType>::
		updateStates(size_t index_i, Real dt, const ErrorAndParameters<VariableType> &error_and_parameters)
	{
		Real parameter_l = error_and_parameters.a_ * error_and_parameters.a_ + error_and_parameters.c_;
		VariableType parameter_k = error_and_parameters.error_ / (parameter_l + TinyReal);
		variable_[index_i] += parameter_k * error_and_parameters.a_;

		Real Vol_i = Vol_[index_i];
		VariableType &variable_i = variable_[index_i];
		Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];

			Real parameter_b = 2.0 * eta_ * inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt / inner_neighborhood.r_ij_[n];

			// predicted quantity at particle j
			VariableType variable_j = variable_[index_j] - parameter_k * parameter_b;
			VariableType variable_derivative = (variable_i - variable_j);

			// exchange in conservation form
			variable_[index_j] -= variable_derivative * parameter_b / mass_[index_j];
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	void DampingBySplittingInner<VariableType>::interaction(size_t index_i, Real dt)
	{
		ErrorAndParameters<VariableType> error_and_parameters = computeErrorAndParameters(index_i, dt);
		updateStates(index_i, dt, error_and_parameters);
	}
	//=================================================================================================//
	template <typename VariableType>
	DampingBySplittingComplex<VariableType>::
		DampingBySplittingComplex(ComplexRelation &complex_relation,
								  const std::string &variable_name, Real eta)
		: DampingBySplittingInner<VariableType>(complex_relation.getInnerRelation(), variable_name, eta),
		  DissipationDataContact(complex_relation.getContactRelation())
	{
		for (size_t k = 0; k != contact_particles_.size(); ++k)
		{
			contact_Vol_.push_back(&(contact_particles_[k]->Vol_));
			contact_mass_.push_back(&(contact_particles_[k]->mass_));
			contact_variable_.push_back(contact_particles_[k]->template getVariableByName<VariableType>(variable_name));
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	ErrorAndParameters<VariableType>
	DampingBySplittingComplex<VariableType>::computeErrorAndParameters(size_t index_i, Real dt)
	{
		ErrorAndParameters<VariableType> error_and_parameters =
			DampingBySplittingInner<VariableType>::computeErrorAndParameters(index_i, dt);

		VariableType &variable_i = this->variable_[index_i];
		Real Vol_i = this->Vol_[index_i];
		/** Contact interaction. */
		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			StdLargeVec<Real> &Vol_k = *(this->contact_Vol_[k]);
			StdLargeVec<Real> &mass_k = *(this->contact_mass_[k]);
			StdLargeVec<VariableType> &variable_k = *(this->contact_variable_[k]);
			Neighborhood &contact_neighborhood = (*this->contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];

				// linear projection
				VariableType variable_derivative = (variable_i - variable_k[index_j]);
				Real parameter_b = 2.0 * this->eta_ * contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt / contact_neighborhood.r_ij_[n];

				error_and_parameters.error_ -= variable_derivative * parameter_b;
				error_and_parameters.a_ += parameter_b;
				error_and_parameters.c_ += parameter_b * parameter_b;
			}
			return error_and_parameters;
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	void DampingBySplittingComplex<VariableType>::
		updateStates(size_t index_i, Real dt, const ErrorAndParameters<VariableType> &error_and_parameters)
	{
		DampingBySplittingInner<VariableType>::updateStates(index_i, dt, error_and_parameters);

		Real parameter_l = error_and_parameters.a_ * error_and_parameters.a_ + error_and_parameters.c_;
		VariableType parameter_k = error_and_parameters.error_ / (parameter_l + TinyReal);
		VariableType &variable_i = this->variable_[index_i];
		Real Vol_i = this->Vol_[index_i];
		/** Contact interaction. */
		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			StdLargeVec<Real> &Vol_k = *(this->contact_Vol_[k]);
			StdLargeVec<Real> &mass_k = *(this->contact_mass_[k]);
			StdLargeVec<VariableType> &variable_k = *(this->contact_variable_[k]);
			Neighborhood &contact_neighborhood = (*this->contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];

				// linear projection
				Real parameter_b = 2.0 * this->eta_ * contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt / contact_neighborhood.r_ij_[n];

				// predicted quantity at particle j
				VariableType variable_j = this->variable_k[index_j] - parameter_k * parameter_b;
				VariableType variable_derivative = (variable_i - variable_j);

				// exchange in conservation form
				this->variable_k[index_j] -= variable_derivative * parameter_b / mass_k[index_j];
			}
		}
	}
	//=================================================================================================//
	template <typename VariableType,
			  template <typename BaseVariableType> class BaseDampingBySplittingType>
	DampingBySplittingWithWall<VariableType, BaseDampingBySplittingType>::
		DampingBySplittingWithWall(ComplexRelation &complex_wall_relation,
								   const std::string &variable_name, Real eta)
		: BaseDampingBySplittingType<VariableType>(complex_wall_relation.getInnerRelation(), variable_name, eta),
		  DissipationDataWithWall(complex_wall_relation.getContactRelation())
	{
		for (size_t k = 0; k != DissipationDataWithWall::contact_particles_.size(); ++k)
		{
			wall_Vol_.push_back(&(contact_particles_[k]->Vol_));
			wall_variable_.push_back(contact_particles_[k]->template getVariableByName<VariableType>(variable_name));
		}
	}
	//=================================================================================================//
	template <typename VariableType,
			  template <typename BaseVariableType> class BaseDampingBySplittingType>
	ErrorAndParameters<VariableType>
	DampingBySplittingWithWall<VariableType, BaseDampingBySplittingType>::
		computeErrorAndParameters(size_t index_i, Real dt)
	{
		ErrorAndParameters<VariableType> error_and_parameters =
			BaseDampingBySplittingType<VariableType>::computeErrorAndParameters(index_i, dt);

		const VariableType &variable_i = this->variable_[index_i];
		Real Vol_i = this->Vol_[index_i];
		/** Contact interaction. */
		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			StdLargeVec<VariableType> &variable_k = *(this->wall_variable_[k]);
			Neighborhood &contact_neighborhood = (*DissipationDataWithWall::contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];

				// linear projection
				VariableType variable_derivative = (variable_i - variable_k[index_j]);
				Real parameter_b = 2.0 * this->eta_ * contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt / contact_neighborhood.r_ij_[n];

				error_and_parameters.error_ -= variable_derivative * parameter_b;
				error_and_parameters.a_ += parameter_b;
				error_and_parameters.c_ += parameter_b * parameter_b;
			}
		}
		return error_and_parameters;
	}
	//=================================================================================================//
	template <typename VariableType>
	BaseDampingPairwiseInner<VariableType>::
		BaseDampingPairwiseInner(BaseInnerRelation &inner_relation, const std::string &variable_name)
		: LocalDynamics(inner_relation.sph_body_), DissipationDataInner(inner_relation),
		  Vol_(particles_->Vol_), mass_(particles_->mass_),
		  variable_(*particles_->getVariableByName<VariableType>(variable_name)) {}
	//=================================================================================================//
	template <typename VariableType>
	template <typename InterParticleCoefficient>
	void BaseDampingPairwiseInner<VariableType>::
		dampPairwiseInner(size_t index_i, Real dt, const InterParticleCoefficient &coefficient)
	{
		Real Vol_i = Vol_[index_i];
		Real mass_i = mass_[index_i];
		VariableType &variable_i = variable_[index_i];
		Real dt2 = dt * 0.5;
		const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		// forward sweep
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];
			Real mass_j = mass_[index_j];

			VariableType variable_derivative = (variable_i - variable_[index_j]);
			Real parameter_b = 2.0 * coefficient(index_i, index_j) *
							   inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / inner_neighborhood.r_ij_[n];

			VariableType increment = parameter_b * variable_derivative / (mass_i * mass_j - parameter_b * (mass_i + mass_j));
			variable_i += increment * mass_j;
			variable_[index_j] -= increment * mass_i;
		}

		// backward sweep
		for (size_t n = inner_neighborhood.current_size_; n != 0; --n)
		{
			size_t index_j = inner_neighborhood.j_[n - 1];
			Real mass_j = mass_[index_j];

			VariableType variable_derivative = (variable_i - variable_[index_j]);
			Real parameter_b = 2.0 * coefficient(index_i, index_j) *
							   inner_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / inner_neighborhood.r_ij_[n - 1];

			VariableType increment = parameter_b * variable_derivative / (mass_i * mass_j - parameter_b * (mass_i + mass_j));
			variable_i += increment * mass_j;
			variable_[index_j] -= increment * mass_i;
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	DampingPairwiseInner<VariableType>::
		DampingPairwiseInner(BaseInnerRelation &inner_relation, const std::string &variable_name, Real eta)
		: BaseDampingPairwiseInner<VariableType>(inner_relation, variable_name), eta_(eta) {}
	//=================================================================================================//
	template <typename VariableType>
	void DampingPairwiseInner<VariableType>::interaction(size_t index_i, Real dt)
	{
		this->dampPairwiseInner(index_i, dt,
								[&](size_t i, size_t j)
								{ return eta_; });
	}
	//=================================================================================================//
	template <typename VariableType>
	DampingPairwiseInnerVariableCoefficient<VariableType>::
		DampingPairwiseInnerVariableCoefficient(BaseInnerRelation &inner_relation,
												const std::string &variable_name, const std::string &coefficient_name)
		: BaseDampingPairwiseInner<VariableType>(inner_relation, variable_name),
		  eta_(*this->particles_->template getVariableByName<VariableType>(coefficient_name)) {}
	//=================================================================================================//
	template <typename VariableType>
	void DampingPairwiseInnerVariableCoefficient<VariableType>::interaction(size_t index_i, Real dt)
	{
		this->dampPairwiseInner(index_i, dt,
								[&](size_t i, size_t j)
								{ return 0.5 * (eta_[i] + eta_[j]); });
	}
	//=================================================================================================//
	template <typename VariableType>
	BaseDampingPairwiseComplex<VariableType>::
		BaseDampingPairwiseComplex(BaseInnerRelation &inner_relation,
								   BaseContactRelation &contact_relation, const std::string &variable_name)
		: BaseDampingPairwiseInner<VariableType>(inner_relation, variable_name),
		  DissipationDataContact(contact_relation)
	{
		for (size_t k = 0; k != contact_particles_.size(); ++k)
		{
			contact_mass_.push_back(&(contact_particles_[k]->mass_));
			contact_variable_.push_back(contact_particles_[k]->template getVariableByName<VariableType>(variable_name));
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	BaseDampingPairwiseComplex<VariableType>::
		BaseDampingPairwiseComplex(ComplexRelation &complex_relation, const std::string &variable_name)
		: BaseDampingPairwiseComplex(complex_relation.getInnerRelation(),
									 complex_relation.getContactRelation(), variable_name) {}
	//=================================================================================================//
	template <typename VariableType>
	template <typename InterParticleCoefficient>
	void BaseDampingPairwiseComplex<VariableType>::
		dampPairwiseContact(size_t index_i, Real dt, const InterParticleCoefficient &coefficient)
	{
		Real Vol_i = this->Vol_[index_i];
		Real mass_i = this->mass_[index_i];
		VariableType &variable_i = this->variable_[index_i];
		Real dt2 = dt * 0.5;
		/** Contact interaction. */
		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			StdLargeVec<Real> &mass_k = *(this->contact_mass_[k]);
			StdLargeVec<VariableType> &variable_k = *(this->contact_variable_[k]);
			Neighborhood &contact_neighborhood = (*this->contact_configuration_[k])[index_i];
			// forward sweep
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];
				Real mass_j = mass_k[index_j];

				VariableType variable_derivative = (variable_i - variable_k[index_j]);
				Real parameter_b = 2.0 * coefficient(index_i, index_j) *
								   contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / contact_neighborhood.r_ij_[n];

				VariableType increment = parameter_b * variable_derivative / (mass_i * mass_j - parameter_b * (mass_i + mass_j));
				variable_i += increment * mass_j;
				variable_k[index_j] -= increment * mass_i;
			}
			// backward sweep
			for (size_t n = contact_neighborhood.current_size_; n != 0; --n)
			{
				size_t index_j = contact_neighborhood.j_[n - 1];
				Real mass_j = mass_k[index_j];

				VariableType variable_derivative = (variable_i - variable_k[index_j]);
				Real parameter_b = 2.0 * coefficient(index_i, index_j) *
								   contact_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / contact_neighborhood.r_ij_[n - 1];

				VariableType increment = parameter_b * variable_derivative / (mass_i * mass_j - parameter_b * (mass_i + mass_j));
				variable_i += increment * mass_j;
				variable_k[index_j] -= increment * mass_i;
			}
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	BaseDampingPairwiseFromWall<VariableType>::
		BaseDampingPairwiseFromWall(BaseContactRelation &contact_relation, const std::string &variable_name)
		: LocalDynamics(contact_relation.sph_body_),
		  DataDelegateContact<BaseParticles, SolidParticles>(contact_relation),
		  Vol_(particles_->Vol_), mass_(particles_->mass_),
		  variable_(*particles_->getVariableByName<VariableType>(variable_name))
	{
		for (size_t k = 0; k != contact_particles_.size(); ++k)
		{
			wall_variable_.push_back(contact_particles_[k]->template getVariableByName<VariableType>(variable_name));
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	template <typename InterParticleCoefficient>
	void BaseDampingPairwiseFromWall<VariableType>::
		dampPairwiseFromWall(size_t index_i, Real dt, const InterParticleCoefficient &coefficient)
	{
		Real Vol_i = Vol_[index_i];
		Real mass_i = mass_[index_i];
		VariableType &variable_i = variable_[index_i];
		Real dt2 = dt * 0.5;
		// interaction with wall
		for (size_t k = 0; k < contact_configuration_.size(); ++k)
		{
			StdLargeVec<VariableType> &variable_k = *(wall_variable_[k]);
			Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
			// forward sweep
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];
				Real parameter_b = 2.0 * coefficient(index_i, index_j) *
								   contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / contact_neighborhood.r_ij_[n];

				// only update particle i
				variable_i += parameter_b * (variable_i - variable_k[index_j]) / (mass_i - parameter_b);
			}

			// backward sweep
			for (size_t n = contact_neighborhood.current_size_; n != 0; --n)
			{
				size_t index_j = contact_neighborhood.j_[n - 1];
				Real parameter_b = 2.0 * coefficient(index_i, index_j) *
								   contact_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / contact_neighborhood.r_ij_[n - 1];

				// only update particle i
				variable_i += parameter_b * (variable_i - variable_k[index_j]) / (mass_i - parameter_b);
			}
		}
	}
	//=================================================================================================//
	template <typename VariableType>
	DampingPairwiseFromWall<VariableType>::
		DampingPairwiseFromWall(BaseContactRelation &contact_relation, const std::string &variable_name, Real eta)
		: BaseDampingPairwiseFromWall<VariableType>(contact_relation, variable_name), eta_(eta) {}
	//=================================================================================================//
	template <typename VariableType>
	void DampingPairwiseFromWall<VariableType>::interaction(size_t index_i, Real dt)
	{
		this->dampPairwiseFromWall(index_i, dt,
								   [&](size_t i, size_t j)
								   { return eta_; });
	}
	//=================================================================================================//
	template <typename VariableType>
	DampingPairwiseFromWallVariableCoefficient<VariableType>::
		DampingPairwiseFromWallVariableCoefficient(
			BaseContactRelation &contact_relation,
			const std::string &variable_name, const std::string &coefficient_name)
		: BaseDampingPairwiseFromWall<VariableType>(contact_relation, variable_name),
		  eta_(*this->particles_->template getVariableByName<VariableType>(coefficient_name)) {}
	//=================================================================================================//
	template <typename VariableType>
	void DampingPairwiseFromWallVariableCoefficient<VariableType>::interaction(size_t index_i, Real dt)
	{
		this->dampPairwiseFromWall(index_i, dt,
								   [&](size_t i, size_t j)
								   { return eta_[i]; });
	}
	//=================================================================================================//
	template <class DampingAlgorithmType>
	template <typename... ConstructorArgs>
	DampingWithRandomChoice<DampingAlgorithmType>::
		DampingWithRandomChoice(Real random_ratio, ConstructorArgs &&...args)
		: DampingAlgorithmType(std::forward<ConstructorArgs>(args)...), random_ratio_(random_ratio)
	{
		this->eta_ /= random_ratio;
	}
	//=================================================================================================//
	template <class DampingAlgorithmType>
	bool DampingWithRandomChoice<DampingAlgorithmType>::RandomChoice()
	{
		return ((double)rand() / (RAND_MAX)) < random_ratio_ ? true : false;
	}
	//=================================================================================================//
	template <class DampingAlgorithmType>
	void DampingWithRandomChoice<DampingAlgorithmType>::exec(Real dt)
	{
		if (RandomChoice())
			DampingAlgorithmType::exec(dt);
	}
	//=================================================================================================//
	template <class DampingAlgorithmType>
	void DampingWithRandomChoice<DampingAlgorithmType>::parallel_exec(Real dt)
	{
		if (RandomChoice())
			DampingAlgorithmType::parallel_exec(dt);
	}
	//=================================================================================================//
	template <typename SteadySolutionCheckType>
	template <class BodyRelationType, typename... Args>
	DampingSteadyCheckVariableCoefficient<SteadySolutionCheckType>::
		DampingSteadyCheckVariableCoefficient(BodyRelationType &body_relation,
											  const std::string &coefficient_name, Args &&...args)
		: SteadySolutionCheckType(body_relation, std::forward<Args>(args)...),
		  eta_(*this->particles_->template getVariableByName<Real>(coefficient_name)) {}
	//=================================================================================================//
	template <typename SteadySolutionCheckType>
	bool DampingSteadyCheckVariableCoefficient<SteadySolutionCheckType>::
		reduce(size_t index_i, Real dt)
	{
		return this->checkCriterion(this->Residue(index_i,
												  [&](size_t i, size_t j)
												  { return 0.5 * (eta_[i] + eta_[j]); }),
									dt);
	}
	//=================================================================================================//
}
#endif // PARTICLE_DYNAMICS_DISSIPATION_HPP