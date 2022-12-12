/**
* @file 	diffusion_splitting_base.hpp
* @author	Bo Zhang and Xiangyu Hu
*/

#ifndef DIFFUSION_SPLITTING_BASE_HPP
#define DIFFUSION_SPLITTING_BASE_HPP

#include "diffusion_splitting_base.h"

namespace SPH
{
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	OptimizationBySplittingAlgorithmBase<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>::
		OptimizationBySplittingAlgorithmBase(BaseInnerRelation& inner_relation, const std::string& variable_name) :
		LocalDynamics(inner_relation.sph_body_),
		DiffusionReactionInnerData<BaseParticlesType, BaseMaterialType, NUM_SPECIES>(inner_relation),
		splitting_index_(this->particles_->splitting_index_), constrain_index_(this->particles_->constrain_index_),
		boundary_index_(this->particles_->boundary_index_), Vol_(this->particles_->Vol_), mass_(this->particles_->mass_),
		heat_flux_(this->particles_->heat_flux_), heat_source_(this->particles_->heat_source_),
		species_modified_(this->particles_->species_modified_), species_recovery_(this->particles_->species_recovery_),
		parameter_recovery_(this->particles_->parameter_recovery_), eta_(this->particles_->eta_regularization_),
		normal_distance_(this->particles_->normal_distance_), normal_vector_(this->particles_->normal_vector_),
		residual_T_local_(this->particles_->residual_T_local_), residual_T_global_(this->particles_->residual_T_global_),
		residual_T_constrain_(this->particles_->residual_T_constrain_), residual_k_local_(this->particles_->residual_k_local_),
		residual_k_global_(this->particles_->residual_k_global_), residual_k_constrain_(this->particles_->residual_k_constrain_),
		variation_local_(this->particles_->variation_local_), variation_global_(this->particles_->variation_global_),
		residual_sp_pde_(this->particles_->residual_sp_pde_), residual_sp_constrain_(this->particles_->residual_sp_constrain_),
		real_heat_flux_T_(this->particles_->real_heat_flux_T_), real_heat_flux_k_(this->particles_->real_heat_flux_k_),
		variable_(*this->particles_->template getVariableByName<VariableType>(variable_name)),
		species_n_(this->particles_->species_n_)
	{
		phi_ = this->particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
		species_diffusion_ = this->particles_->diffusion_reaction_material_.SpeciesDiffusion();
	}
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	RegularizationByDiffusionInner<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>::
		RegularizationByDiffusionInner(BaseInnerRelation& inner_relation, const std::string& variable_name, 
			                           Real initial_eta, Real variation) :
		OptimizationBySplittingAlgorithmBase<BaseParticlesType, BaseMaterialType, VariableType, 
		                                     NUM_SPECIES>(inner_relation, variable_name),
		initial_eta_(initial_eta), maximum_variation_(variation), averaged_variation_(variation) {}
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	ErrorAndParameters<VariableType> RegularizationByDiffusionInner<BaseParticlesType, BaseMaterialType, 
		VariableType, NUM_SPECIES>::computeVariationAndParameters(size_t index_i, Real dt)
	{
		Real Vol_i = this->Vol_[index_i];
		Real mass_i = this->mass_[index_i];
		VariableType& variable_i = this->variable_[index_i];
		ErrorAndParameters<VariableType> error_and_parameters;
		Neighborhood& inner_neighborhood = this->inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t& index_j = inner_neighborhood.j_[n];
			Real& dW_ijV_j_ = inner_neighborhood.dW_ijV_j_[n];
			Real& r_ij_ = inner_neighborhood.r_ij_[n];

			/* different option for choosing local regularization coefficient. */
			//this->eta_[index_i] = initial_eta_ * abs(this->variation_local_[index_i] + TinyReal) / averaged_variation_;
			//this->eta_[index_i] = initial_eta_ * abs(this->variation_local_[index_i] + TinyReal) / maximum_variation_;
			this->eta_[index_i] = initial_eta_; //uniform coefficient.

			VariableType variable_derivative = (variable_i - this->variable_[index_j]);
			Real parameter_b = 2.0 * this->eta_[index_i] * dW_ijV_j_ * Vol_i * dt / r_ij_;

			error_and_parameters.error_ -= variable_derivative * parameter_b;
			error_and_parameters.a_ += parameter_b;
			error_and_parameters.c_ += parameter_b * parameter_b;
		}
		error_and_parameters.a_ -= mass_i;
		return error_and_parameters;
	}
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	void RegularizationByDiffusionInner<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>::
		updateStatesByVariation(size_t index_i, Real dt, const ErrorAndParameters<VariableType>& error_and_parameters)
	{
		Real parameter_l = error_and_parameters.a_ * error_and_parameters.a_ + error_and_parameters.c_;
		VariableType parameter_k = error_and_parameters.error_ / (parameter_l + TinyReal);
		this->variable_[index_i] += parameter_k * error_and_parameters.a_;
		if (this->variable_[index_i] < 0.000001) { this->variable_[index_i] = 0.000001; }

		Real Vol_i = this->Vol_[index_i];
		VariableType& variable_i = this->variable_[index_i];
		Neighborhood& inner_neighborhood = this->inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];
			Real& dW_ijV_j_ = inner_neighborhood.dW_ijV_j_[n];
			Real& r_ij_ = inner_neighborhood.r_ij_[n];

			Real parameter_b = 2.0 * this->eta_[index_i] * dW_ijV_j_ * Vol_i * dt / r_ij_;

			//predicted quantity at particle j
			VariableType variable_j = this->variable_[index_j] - parameter_k * parameter_b;
			VariableType variable_derivative = (variable_i - variable_j);

			//exchange in conservation form
			this->variable_[index_j] -= variable_derivative * parameter_b / this->mass_[index_j];
			if (this->variable_[index_j] < 0.000001) { this->variable_[index_j] = 0.000001; }
		}
	}
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	void RegularizationByDiffusionInner<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>::
		interaction(size_t index_i, Real dt)
	{
		ErrorAndParameters<VariableType> error_and_parameters = computeVariationAndParameters(index_i, dt);
		updateStatesByVariation(index_i, dt, error_and_parameters);
		this->variation_local_[index_i] = error_and_parameters.error_ / dt / this->eta_[index_i];
	}
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	UpdateRegularizationVariation<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>::
		UpdateRegularizationVariation(BaseInnerRelation& inner_relation, const std::string& variable_name) :
		OptimizationBySplittingAlgorithmBase<BaseParticlesType, BaseMaterialType,
		                                     VariableType, NUM_SPECIES>(inner_relation, variable_name) {};
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	ErrorAndParameters<VariableType> UpdateRegularizationVariation<BaseParticlesType, BaseMaterialType, 
		VariableType, NUM_SPECIES>::computeVariationAndParameters(size_t index_i, Real dt)
	{
		Real Vol_i = this->Vol_[index_i];
		Real mass_i = this->mass_[index_i];
		VariableType &variable_i = this->variable_[index_i];
		ErrorAndParameters<VariableType> error_and_parameters;

		Neighborhood &inner_neighborhood = this->inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];
			Real &dW_ijV_j_ = inner_neighborhood.dW_ijV_j_[n];
			Real &r_ij_ = inner_neighborhood.r_ij_[n];

			VariableType variable_derivative = (variable_i - this->variable_[index_j]);
			Real parameter_b = 2.0 * this->eta_[index_i] * dW_ijV_j_ * Vol_i * dt / r_ij_;
		                  	   
			error_and_parameters.error_ -= variable_derivative * parameter_b;
			error_and_parameters.a_ += parameter_b;
			error_and_parameters.c_ += parameter_b * parameter_b;
		}
		error_and_parameters.a_ -= mass_i;
		return error_and_parameters;
	}
	//=================================================================================================//
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES>
	void UpdateRegularizationVariation<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>::
		interaction(size_t index_i, Real dt)
	{
		ErrorAndParameters<VariableType> error_and_parameters = this->computeVariationAndParameters(index_i, dt);
		this->variation_global_[index_i] = error_and_parameters.error_ / dt / this->eta_[index_i];
	}
	//=================================================================================================//
};

#endif DIFFUSION_SPLITTING_BASE_H