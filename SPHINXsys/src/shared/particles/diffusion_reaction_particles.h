/* -------------------------------------------------------------------------*
 *								SPHinXsys									*
 * -------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle*
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for	*
 * physical accurate simulation and aims to model coupled industrial dynamic*
 * systems including fluid, solid, multi-body dynamics and beyond with SPH	*
 * (smoothed particle hydrodynamics), a meshless computational method using	*
 * particle discretization.													*
 *																			*
 * SPHinXsys is partially funded by German Research Foundation				*
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1,			*
 *  HU1527/12-1 and HU1527/12-4												*
 *                                                                          *
 * Portions copyright (c) 2017-2022 Technical University of Munich and		*
 * the authors' affiliations.												*
 *                                                                          *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may  *
 * not use this file except in compliance with the License. You may obtain a*
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.       *
 *                                                                          *
 * ------------------------------------------------------------------------*/
/**
 * @file 	diffusion_reaction_particles.h
 * @brief 	This is the derived class of diffusion reaction particles.
 * @author	Chi ZHang and Xiangyu Hu
 */

#ifndef DIFFUSION_REACTION_PARTICLES_H
#define DIFFUSION_REACTION_PARTICLES_H

#include "base_particles.h"
#include "base_body.h"
#include "base_material.h"
#include "diffusion_reaction.h"

namespace SPH
{

	/**
	 * @class DiffusionReactionParticles
	 * @brief A group of particles with diffusion or/and reactions particle data.
	 */
	template <class BaseParticlesType, class BaseMaterialType = BaseMaterial, int NUM_SPECIES = 1>
	class DiffusionReactionParticles : public BaseParticlesType
	{
	protected:
		size_t number_of_species_;			 /**< Total number of diffusion and reaction species . */
		size_t number_of_diffusion_species_; /**< Total number of diffusion species . */
		std::map<std::string, size_t> species_indexes_map_;

	public:
		StdVec<StdLargeVec<Real>> species_n_;	 /**< array of diffusion/reaction scalars */
		StdVec<StdLargeVec<Real>> diffusion_dt_; /**< array of the time derivative of diffusion species */
		DiffusionReaction<BaseMaterialType, NUM_SPECIES> &diffusion_reaction_material_;

		/* added by Bo for thermal optimization */
		StdLargeVec<Real> species_modified_;         /**< species modified by objective function for parameter splitting */
		StdLargeVec<Real> species_recovery_;         /**< species recovery after one step of parameter splitting */
		StdLargeVec<Real> parameter_recovery_;       /**< backup of the parameter for gradient descent failure */
		StdLargeVec<Real> eta_regularization_;       /**< regularization coefficient for each particle */
		StdLargeVec<Real> heat_flux_; 				 /**< heat flux value for Neumann boundary condition */
		StdLargeVec<Real> real_heat_flux_T_;         /**< real heat flux value calculated from temperature*/
		StdLargeVec<Real> real_heat_flux_k_;         /**< real heat flux value calculated from parameter */
		StdLargeVec<Real> heat_source_;              /**< heat source used for inner heat intensity */
		StdLargeVec<Real> normal_distance_;          /**< normal distance to the boundary (only for boundary particles) */
		StdLargeVec<Vecd> normal_vector_;  			 /**< unit normal vector for Neumann BC (may be same as n_ of base particle) */
		StdLargeVec<Real> residual_T_local_;         /**< array of the local PDE residual calculated from temperature */
		StdLargeVec<Real> residual_T_global_;        /**< array of the global PDE residual calculated from temperature */
		StdLargeVec<Real> residual_k_local_;         /**< array of the local PDE residual calculated from parameter */
		StdLargeVec<Real> residual_k_global_;		 /**< array of the global PDE residual calcualted from parameter */
		StdLargeVec<Real> residual_sp_pde_;          /**< array of the local PDE residual after one step parameter splitting */
		StdLargeVec<Real> residual_T_constrain_;     /**< array of the BC residual calculated from temperature */
		StdLargeVec<Real> residual_k_constrain_;     /**< array of the BC residual calculated from parameter */
		StdLargeVec<Real> residual_sp_constrain_;    /**< array of the BC residual after one step parameter splitting */
		StdLargeVec<Real> variation_local_;          /**< array of the local variation in regularization */
		StdLargeVec<Real> variation_global_;         /**< array of the global variation in regularization */
		StdLargeVec<int> splitting_index_;           /**< array of the index indicates the valid of PDE　parameter splitting */
		StdLargeVec<int> constrain_index_;           /**< array of the index indicates the valid of BC parameter splitting */
		StdLargeVec<int> boundary_index_;            /**< array of the index indicates the boundary particles */

		DiffusionReactionParticles(SPHBody &sph_body,
								   DiffusionReaction<BaseMaterialType, NUM_SPECIES> *diffusion_reaction_material)
			: BaseParticlesType(sph_body, diffusion_reaction_material),
			  number_of_species_(diffusion_reaction_material->NumberOfSpecies()),
			  number_of_diffusion_species_(diffusion_reaction_material->NumberOfSpeciesDiffusion()),
			  species_indexes_map_(diffusion_reaction_material->SpeciesIndexMap()),
			  diffusion_reaction_material_(*diffusion_reaction_material)
		{
			species_n_.resize(number_of_species_);
			diffusion_dt_.resize(number_of_diffusion_species_);
		};
		virtual ~DiffusionReactionParticles(){};

		std::map<std::string, size_t> SpeciesIndexMap() { return species_indexes_map_; };

		virtual void initializeOtherVariables() override
		{
			BaseParticlesType::initializeOtherVariables();

			std::map<std::string, size_t>::iterator itr;
			for (itr = species_indexes_map_.begin(); itr != species_indexes_map_.end(); ++itr)
			{
				// Register a specie.
				this->registerVariable(species_n_[itr->second], itr->first);
				/** the scalars will be sorted if particle sorting is called, Note that we call a template function from a template class. */
				this->template registerSortableVariable<Real>(itr->first);
				/** add species to basic output particle data. */
				this->template addVariableToWrite<Real>(itr->first);
				/** add species to output restart particle data. */
				this->template addVariableToRestart<Real>(itr->first);
			}

			for (size_t m = 0; m < number_of_diffusion_species_; ++m)
			{
				constexpr int type_index = DataTypeIndex<Real>::value;
				/** 
				 * register reactive change rate terms without giving variable name
				 */
				std::get<type_index>(this->all_particle_data_).push_back(&diffusion_dt_[m]);
				diffusion_dt_[m].resize(this->real_particles_bound_, Real(0));
			}

			/** added by Bo for thermal optimization */
			this->registerVariable(species_modified_, "SpeciesModified");
			this->template addVariableToWrite<Real>("SpeciesModified");
			this->registerVariable(species_recovery_, "SpeciesRecovery");
			this->template addVariableToWrite<Real>("SpeciesRecovery");
			this->registerVariable(parameter_recovery_, "ParameterRecovery");
			this->template addVariableToWrite<Real>("ParameterRecovery");
			this->registerVariable(eta_regularization_, "EtaRegularization", [&](size_t i) -> Real { return 1; });
			this->template addVariableToWrite<Real>("EtaRegularization");
			this->registerVariable(heat_flux_, "HeatFlux");
			this->template addVariableToWrite<Real>("HeatFlux");
			this->registerVariable(heat_source_, "HeatSource");
			this->template addVariableToWrite<Real>("HeatSource");
			this->registerVariable(normal_distance_, "NormalDistance");
			this->template addVariableToWrite<Real>("NormalDistance");
			this->registerVariable(normal_vector_, "UnitNormalVector");
			this->template addVariableToWrite<Vecd>("UnitNormalVector");

			this->registerVariable(residual_T_local_, "residual_T_local");
			this->template addVariableToWrite<Real>("residual_T_local");
			this->registerVariable(residual_T_global_, "residual_T_global");
			this->template addVariableToWrite<Real>("residual_T_global");
			this->registerVariable(residual_k_local_, "residual_k_local");
			this->template addVariableToWrite<Real>("residual_k_local");
			this->registerVariable(residual_k_global_, "residual_k_global");
			this->template addVariableToWrite<Real>("residual_k_global");
			this->registerVariable(residual_sp_pde_, "residual_sp_pde");
			this->template addVariableToWrite<Real>("residual_sp_pde");
			this->registerVariable(residual_T_constrain_, "residual_T_constrain");
			this->template addVariableToWrite<Real>("residual_T_constrain");
			this->registerVariable(residual_k_constrain_, "residual_k_constrain");
			this->template addVariableToWrite<Real>("residual_k_constrain");
			this->registerVariable(residual_sp_constrain_, "residual_sp_constrain");
			this->template addVariableToWrite<Real>("residual_sp_constrain");
			this->registerVariable(variation_local_, "variation_local");
			this->template addVariableToWrite<Real>("variation_local");
			this->registerVariable(variation_global_, "variation_global");
			this->template addVariableToWrite<Real>("variation_global");
			this->registerVariable(real_heat_flux_T_, "RealHeatFluxFromT");
			this->template addVariableToWrite<Real>("RealHeatFluxFromT");
			this->registerVariable(real_heat_flux_k_, "RealHeatFluxFromk");
			this->template addVariableToWrite<Real>("RealHeatFluxFromk");
			this->registerVariable(splitting_index_, "splitting_index");
			this->template addVariableToWrite<int>("splitting_index");
			this->registerVariable(constrain_index_, "constrain_index");
			this->template addVariableToWrite<int>("constrain_index");
			this->registerVariable(boundary_index_, "boundary_index");
			this->template addVariableToWrite<int>("boundary_index");
		};

		virtual DiffusionReactionParticles<BaseParticlesType, BaseMaterialType, NUM_SPECIES> *ThisObjectPtr() override { return this; };
	};
}
#endif // DIFFUSION_REACTION_PARTICLES_H