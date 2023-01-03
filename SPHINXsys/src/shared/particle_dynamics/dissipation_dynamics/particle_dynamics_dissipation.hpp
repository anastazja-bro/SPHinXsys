#ifndef PARTICLE_DYNAMICS_DISSIPATION_HPP
#define PARTICLE_DYNAMICS_DISSIPATION_HPP

#include "particle_dynamics_dissipation.h"

namespace SPH
{
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	template <typename CoefficientArg>
	BaseDampingSplittingInner<DataType, CoefficientType>::
		BaseDampingSplittingInner(BaseInnerRelation &inner_relation,
								  const std::string &variable_name, const CoefficientArg &eta)
		: OperatorInner<DataType, DataType, CoefficientType>(
			  inner_relation, variable_name, variable_name, eta),
		  Vol_(this->particles_->Vol_), mass_(this->particles_->mass_),
		  variable_(this->in_variable_) {}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	ErrorAndParameters<DataType>
	BaseDampingSplittingInner<DataType, CoefficientType>::computeErrorAndParameters(size_t index_i, Real dt)
	{
		Real Vol_i = Vol_[index_i];
		Real mass_i = mass_[index_i];
		DataType &variable_i = variable_[index_i];
		ErrorAndParameters<DataType> error_and_parameters;
		Neighborhood &inner_neighborhood = this->inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];
			// linear projection
			DataType variable_derivative = (variable_i - variable_[index_j]);
			Real parameter_b = 2.0 * this->coefficient_(index_i, index_j) *
							   inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt / inner_neighborhood.r_ij_[n];

			error_and_parameters.error_ -= variable_derivative * parameter_b;
			error_and_parameters.a_ += parameter_b;
			error_and_parameters.c_ += parameter_b * parameter_b;
		}
		error_and_parameters.a_ -= mass_i;
		return error_and_parameters;
	}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	void BaseDampingSplittingInner<DataType, CoefficientType>::
		updateStates(size_t index_i, Real dt, const ErrorAndParameters<DataType> &error_and_parameters)
	{
		Real parameter_l = error_and_parameters.a_ * error_and_parameters.a_ + error_and_parameters.c_;
		DataType parameter_k = error_and_parameters.error_ / (parameter_l + TinyReal);
		variable_[index_i] += parameter_k * error_and_parameters.a_;

		Real Vol_i = Vol_[index_i];
		Neighborhood &inner_neighborhood = this->inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];

			Real parameter_b = 2.0 * this->coefficient_(index_i, index_j) *
							   inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt / inner_neighborhood.r_ij_[n];

			// predicted quantity at particle j
			DataType variable_j = variable_[index_j] - parameter_k * parameter_b;
			DataType variable_derivative = (variable_[index_i] - variable_j);

			// exchange in conservation form
			variable_[index_j] -= variable_derivative * parameter_b / mass_[index_j];
		}
	}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	void BaseDampingSplittingInner<DataType, CoefficientType>::interaction(size_t index_i, Real dt)
	{
		ErrorAndParameters<DataType> error_and_parameters = computeErrorAndParameters(index_i, dt);
		updateStates(index_i, dt, error_and_parameters);
	}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	template <typename CoefficientArg>
	BaseDampingSplittingWithWall<DataType, CoefficientType>::
		BaseDampingSplittingWithWall(ComplexRelation &complex_wall_relation,
									 const std::string &variable_name, const CoefficientArg &eta)
		: BaseDampingSplittingInner<DataType, CoefficientType>(
			  complex_wall_relation.getInnerRelation(), variable_name, eta),
		  DissipationDataWithWall(complex_wall_relation.getContactRelation())
	{
		for (size_t k = 0; k != DissipationDataWithWall::contact_particles_.size(); ++k)
		{
			wall_variable_.push_back(contact_particles_[k]->template getVariableByName<DataType>(variable_name));
		}
	}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	ErrorAndParameters<DataType>
	BaseDampingSplittingWithWall<DataType, CoefficientType>::
		computeErrorAndParameters(size_t index_i, Real dt)
	{
		ErrorAndParameters<DataType> error_and_parameters =
			BaseDampingSplittingInner<DataType, CoefficientType>::computeErrorAndParameters(index_i, dt);

		const DataType &variable_i = this->variable_[index_i];
		Real Vol_i = this->Vol_[index_i];
		/** Contact interaction. */
		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			StdLargeVec<DataType> &variable_k = *(this->wall_variable_[k]);
			Neighborhood &contact_neighborhood = (*DissipationDataWithWall::contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];

				// linear projection
				DataType variable_derivative = (variable_i - variable_k[index_j]);
				Real parameter_b = 2.0 * this->coefficient_(index_i) *
								   contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt / contact_neighborhood.r_ij_[n];

				error_and_parameters.error_ -= variable_derivative * parameter_b;
				error_and_parameters.a_ += parameter_b;
			}
		}
		return error_and_parameters;
	}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	template <typename CoefficientArg>
	BaseDampingPairwiseInner<DataType, CoefficientType>::
		BaseDampingPairwiseInner(BaseInnerRelation &inner_relation,
								 const std::string &variable_name, const CoefficientArg &eta)
		: OperatorInner<DataType, DataType, CoefficientType>(
			  inner_relation, variable_name, variable_name, eta),
		  Vol_(this->particles_->Vol_), mass_(this->particles_->mass_),
		  variable_(this->in_variable_) {}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	void BaseDampingPairwiseInner<DataType, CoefficientType>::
		interaction(size_t index_i, Real dt)
	{
		Real Vol_i = Vol_[index_i];
		Real mass_i = mass_[index_i];
		DataType &variable_i = variable_[index_i];
		Real dt2 = dt * 0.5;
		const Neighborhood &inner_neighborhood = this->inner_configuration_[index_i];
		// forward sweep
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];
			Real mass_j = mass_[index_j];

			DataType variable_derivative = (variable_i - variable_[index_j]);
			Real parameter_b = 2.0 * this->coefficient_(index_i, index_j) *
							   inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / inner_neighborhood.r_ij_[n];

			DataType increment = parameter_b * variable_derivative / (mass_i * mass_j - parameter_b * (mass_i + mass_j));
			variable_i += increment * mass_j;
			variable_[index_j] -= increment * mass_i;
		}

		// backward sweep
		for (size_t n = inner_neighborhood.current_size_; n != 0; --n)
		{
			size_t index_j = inner_neighborhood.j_[n - 1];
			Real mass_j = mass_[index_j];

			DataType variable_derivative = (variable_i - variable_[index_j]);
			Real parameter_b = 2.0 * this->coefficient_(index_i, index_j) *
							   inner_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / inner_neighborhood.r_ij_[n - 1];

			DataType increment = parameter_b * variable_derivative / (mass_i * mass_j - parameter_b * (mass_i + mass_j));
			variable_i += increment * mass_j;
			variable_[index_j] -= increment * mass_i;
		}
	}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	template <typename CoefficientArg>
	BaseDampingPairwiseFromWall<DataType, CoefficientType>::
		BaseDampingPairwiseFromWall(BaseContactRelation &contact_relation,
									const std::string &variable_name, const CoefficientArg &eta)
		: OperatorFromBoundary<DataType, DataType, CoefficientType>(
			  contact_relation, variable_name, variable_name, eta),
		  Vol_(this->particles_->Vol_), mass_(this->particles_->mass_),
		  variable_(this->in_variable_), wall_variable_(this->contact_in_variable_) {}
	//=================================================================================================//
	template <typename DataType, class CoefficientType>
	void BaseDampingPairwiseFromWall<DataType, CoefficientType>::interaction(size_t index_i, Real dt)
	{
		Real Vol_i = Vol_[index_i];
		Real mass_i = mass_[index_i];
		DataType &variable_i = variable_[index_i];
		Real dt2 = dt * 0.5;
		// interaction with wall
		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			StdLargeVec<DataType> &variable_k = *(wall_variable_[k]);
			Neighborhood &contact_neighborhood = (*this->contact_configuration_[k])[index_i];
			// forward sweep
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];
				Real parameter_b = 2.0 * this->coefficient_(index_i) *
								   contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / contact_neighborhood.r_ij_[n];

				// only update particle i
				variable_i += parameter_b * (variable_i - variable_k[index_j]) / (mass_i - parameter_b);
			}

			// backward sweep
			for (size_t n = contact_neighborhood.current_size_; n != 0; --n)
			{
				size_t index_j = contact_neighborhood.j_[n - 1];
				Real parameter_b = 2.0 * this->coefficient_(index_i) *
								   contact_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / contact_neighborhood.r_ij_[n - 1];

				// only update particle i
				variable_i += parameter_b * (variable_i - variable_k[index_j]) / (mass_i - parameter_b);
			}
		}
	}
	//=================================================================================================//
	template <class DampingAlgorithmType>
	template <typename... ConstructorArgs>
	DampingWithRandomChoice<DampingAlgorithmType>::
		DampingWithRandomChoice(Real random_ratio, ConstructorArgs &&...args)
		: DampingAlgorithmType(std::forward<ConstructorArgs>(args)...), random_ratio_(random_ratio)
	{
		DampingAlgorithmType::Coefficient().rescaleCoefficient(1.0 / random_ratio);
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
}
#endif // PARTICLE_DYNAMICS_DISSIPATION_HPP