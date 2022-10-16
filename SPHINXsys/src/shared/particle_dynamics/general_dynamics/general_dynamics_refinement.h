/* -------------------------------------------------------------------------*
 *								SPHinXsys									*
 * --------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle	*
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for	*
 * physical accurate simulation and aims to model coupled industrial dynamic *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH	*
 * (smoothed particle hydrodynamics), a meshless computational method using	*
 * particle discretization.													*
 *																			*
 * SPHinXsys is partially funded by German Research Foundation				*
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1				*
 * and HU1527/12-1.															*
 *                                                                           *
 * Portions copyright (c) 2017-2020 Technical University of Munich and		*
 * the authors' affiliations.												*
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * --------------------------------------------------------------------------*/
/**
 * @file 	general_dynamics_confinement.h
 * @brief 	This is the particle dynamics applicable for all type bodies
 * @author	Yijie Sun and Xiangyu Hu
 */

#pragma once

#include "general_dynamics.h"

namespace SPH
{
	class LevelSetComplexShape;
	class ParticleSplitAndMerge;

	/**
	 * @class ComputeDensityErrorInner
	 * @brief compute error of particle splitting and merging
	 */
	class ComputeDensityErrorInner : public GeneralDataDelegateInner
	{
	public:
		ComputeDensityErrorInner(BaseInnerRelation &inner_relation)
			: GeneralDataDelegateInner(inner_relation),
			  h_ratio_(*particles_->getVariableByName<Real>("SmoothingLengthRatio"))
		{
			particle_adaptation_ = DynamicCast<ParticleSplitAndMerge>(this, inner_relation.sph_body_.sph_adaptation_);
			density_error_.resize(particles_->real_particles_bound_);
			particles_->addVariableToWrite<Real>("Density");
		};
		virtual ~ComputeDensityErrorInner(){};

		Vecd getPositionFromDensityError(StdVec<size_t> original_indices, StdVec<Vecd> new_positions,
										 StdVec<size_t> new_indices_, Real min_distance, Real max_distance);
		virtual void initializeDensityError();

		StdLargeVec<Real> density_error_;
		StdLargeVec<bool> tag_split_;

	protected:
		ParticleSplitAndMerge *particle_adaptation_;
		StdLargeVec<Real> &h_ratio_;
		Vecd E_cof_ = Vecd(0.0);
		Real sigma_E_ = 0.0;
		Real E_cof_sigma_ = 0.0;
		StdVec<Vecd> grad_new_indices_;
		StdVec<Vecd> dW_new_indices_;
		StdVec<Real> sign_new_indices_;

		virtual Vecd computeKernelGradient(size_t index_rho);
		virtual Real computeNewGeneratedParticleDensity(size_t index_rho, Vecd position);
		virtual Vecd getPosition(StdVec<size_t> original_indices, StdVec<Vecd> new_positions, StdVec<size_t> new_indices);
		virtual void densityErrorOfNewGeneratedParticles(StdVec<size_t> new_indices, StdVec<Vecd> new_positions);
		virtual void densityErrorOfNeighborParticles(StdVec<size_t> new_indices, StdVec<size_t> original_indices, StdVec<Vecd> new_positions);
		virtual Real computeKernelWeightBetweenParticles(Real h_ratio, Vecd displacement, Real Vol_ratio = 1.0);
		virtual Vecd computeKernelWeightGradientBetweenParticles(Real h_ratio_min, Vecd displacement, Real Vol);
		virtual void computeDensityErrorOnNeighborParticles(Neighborhood &neighborhood, size_t index_rho,
															StdVec<size_t> original_indices, StdVec<Vecd> new_positions);
		virtual Vecd positionLimitation(Vecd displacement, Real min_distance, Real max_distance);
	};

	/**
	 * @class ComputeDensityErrorWithWall
	 * @brief compute error of particle splitting and merging
	 */
	class ComputeDensityErrorWithWall : public ComputeDensityErrorInner, public GeneralDataDelegateContact
	{
	public:
		ComputeDensityErrorWithWall(ComplexRelation &complex_relation)
			: ComputeDensityErrorInner(complex_relation.inner_relation_),
			  GeneralDataDelegateContact(complex_relation.contact_relation_)
		{
			for (size_t k = 0; k != contact_bodies_.size(); ++k)
			{
				contact_Vol_.push_back(&(contact_bodies_[k]->getBaseParticles().Vol_));
			}
		};
		virtual ~ComputeDensityErrorWithWall(){};

	protected:
		StdVec<StdLargeVec<Real> *> contact_Vol_;

		virtual Vecd computeKernelGradient(size_t index_rho) override;
		virtual Real computeNewGeneratedParticleDensity(size_t index_rho, Vecd position) override;
		virtual void densityErrorOfNeighborParticles(StdVec<size_t> new_indices, StdVec<size_t> original_indices,
													 StdVec<Vecd> new_positions) override;
	};
	/**
	 * @class ParticleRefinementWithPrescribedArea
	 * @brief particle split in prescribed area.
	 */
	class ParticleSplitWithPrescribedArea : public LocalDynamics, public GeneralDataDelegateSimple
	{
	public:
		ParticleSplitWithPrescribedArea(SPHBody &body, BodyRegionByCell &refinement_area, size_t body_buffer_width);
		virtual ~ParticleSplitWithPrescribedArea(){};

		StdVec<size_t> new_indices_;
		void interaction(size_t index_i, Real dt = 0.0);
		void update(size_t index_i, Real dt = 0.0);

	protected:
		SPHBody *body_;
		BodyRegionByCell *refinement_area_;
		StdLargeVec<Vecd> &pos_;
		StdLargeVec<Real> &Vol_;
		StdLargeVec<Real> &mass_;
		ParticleSplitAndMerge *particle_adaptation_;
		StdLargeVec<Real> &h_ratio_; /**< the ratio between reference smoothing length to variable smoothing length */
		StdLargeVec<Real> &rho_;

		StdVec<Vecd> split_position_;
		StdVec<Vecu> split_index_;
		size_t particle_number_change = 0;

		virtual void setupDynamics(Real dt) override;
		virtual void splittingModel(size_t index_i);
		virtual bool splitCriteria(Vecd position, Real volume);
		virtual Vecd getSplittingPosition(StdVec<size_t> new_indices_);
		virtual void updateNewlySplittingParticle(size_t index_i, size_t index_j, Vecd pos_split);
	};

	/**
	 * @class SplitWithMinimumDensityErrorInner
	 * @brief split particles with minimum density error.
	 */
	class SplitWithMinimumDensityErrorInner : public ParticleSplitWithPrescribedArea
	{
	public:
		SplitWithMinimumDensityErrorInner(BaseInnerRelation &inner_relation, BodyRegionByCell &refinement_area, size_t body_buffer_width)
			: ParticleSplitWithPrescribedArea(inner_relation.sph_body_, refinement_area, body_buffer_width),
			  compute_density_error(inner_relation)
		{
			particles_->registerVariable(particle_adaptation_->total_split_error_, "SplitDensityError", 0.0);
			// base_particles->addVariableToWrite<Real>("SplitDensityError");
		};
		virtual ~SplitWithMinimumDensityErrorInner(){};

		void update(size_t index_i, Real dt = 0.0);

	protected:
		ComputeDensityErrorInner compute_density_error;

		virtual Vecd getSplittingPosition(StdVec<size_t> new_indices_) override;
		virtual void setupDynamics(Real dt) override;
	};

	/**
	 * @class SplitWithMinimumDensityErrorWithWall
	 * @brief split particles with minimum density error.
	 */
	class SplitWithMinimumDensityErrorWithWall : public SplitWithMinimumDensityErrorInner
	{
	public:
		SplitWithMinimumDensityErrorWithWall(ComplexRelation &complex_relation, BodyRegionByCell &refinement_area, size_t body_buffer_width)
			: SplitWithMinimumDensityErrorInner(complex_relation.inner_relation_, refinement_area, body_buffer_width),
			  compute_density_error(complex_relation){};
		virtual ~SplitWithMinimumDensityErrorWithWall(){};

	protected:
		ComputeDensityErrorWithWall compute_density_error;
	};

	/**
	 * @class ParticleMergeWithPrescribedArea
	 * @brief merging particle for a body in prescribed area.
	 */
	class ParticleMergeWithPrescribedArea : public LocalDynamics, public GeneralDataDelegateInner
	{
	public:
		ParticleMergeWithPrescribedArea(BaseInnerRelation &inner_relation, BodyRegionByCell &refinement_area);
		virtual ~ParticleMergeWithPrescribedArea(){};

		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		BodyRegionByCell *refinement_area_;
		ParticleData &all_particle_data_;
		Real rho0_inv_;
		StdLargeVec<Vecd> &pos_;
		StdLargeVec<Real> &Vol_;
		StdLargeVec<Real> &mass_;
		int dimension_;
		StdLargeVec<Real> &h_ratio_; /**< the ratio between reference smoothing length to variable smoothing length */
		ParticleSplitAndMerge *particle_adaptation_;
		StdLargeVec<Real> &rho_;
		StdLargeVec<Vecd> &vel_n_;
		StdLargeVec<bool> tag_merged_;

		virtual void setupDynamics(Real dt) override;
		virtual void mergingModel(const StdVec<size_t> &merge_indices);
		virtual bool mergeCriteria(size_t index_i, StdVec<size_t> &merge_indices);
		bool findMergeParticles(size_t index_i, StdVec<size_t> &merge_indices, Real search_size, Real search_distance);
		virtual void updateMergedParticleInformation(size_t merged_index, const StdVec<size_t> &merge_indices);

		template <typename VariableType>
		struct mergeParticleDataValue
		{
			void operator()(ParticleData &particle_data, size_t merged_index, const StdVec<size_t> &merge_indices, StdVec<Real> merge_mass)
			{
				Real total_mass = 0.0;
				for (size_t k = 0; k != merge_indices.size(); ++k)
					total_mass += merge_mass[k];

				constexpr int type_index = DataTypeIndex<VariableType>::value;
				for (size_t i = 0; i != std::get<type_index>(particle_data).size(); ++i)
				{
					VariableType particle_data_temp = VariableType(0);
					for (size_t k = 0; k != merge_indices.size(); ++k)
						particle_data_temp += merge_mass[k] * (*std::get<type_index>(particle_data)[i])[merge_indices[k]];

					(*std::get<type_index>(particle_data)[i])[merged_index] = particle_data_temp / (total_mass + TinyReal);
				}
			};
		};
		DataAssembleOperation<mergeParticleDataValue> merge_particle_value_;
	};

	/**
	 * @class ParticleMergeWithPrescribedArea
	 * @brief merging particles with minimum density error.
	 */
	class MergeWithMinimumDensityErrorInner : public ParticleMergeWithPrescribedArea
	{
	public:
		MergeWithMinimumDensityErrorInner(BaseInnerRelation &inner_relation, BodyRegionByCell &refinement_area)
			: ParticleMergeWithPrescribedArea(inner_relation, refinement_area),
			  compute_density_error(inner_relation){};
		virtual ~MergeWithMinimumDensityErrorInner(){};

		void interaction(size_t index_i, Real dt = 0.0)
		{
			ParticleMergeWithPrescribedArea::interaction(index_i, dt);
		};

	protected:
		ComputeDensityErrorInner compute_density_error;
		StdLargeVec<Real> merge_error_;
		Real rotation = 0.0;
		size_t merge_change_number = 0;

		virtual void setupDynamics(Real dt) override;
		virtual void mergingModel(const StdVec<size_t> &merge_indices) override;
		virtual bool mergeCriteria(size_t index_i, StdVec<size_t> &merge_indices) override;
		virtual Vecd getMergingPosition(StdVec<size_t> new_indices, const StdVec<size_t> &merge_indices);
		virtual Real angularMomentumConservation(size_t index_center, const StdVec<size_t> &merge_indices);
		virtual void kineticEnergyConservation(const StdVec<size_t> &merge_indices);
		virtual void updateNewlyMergingParticle(size_t index_center, StdVec<size_t> new_indexs, Vecd pos_split);
	};

	/**
	 * @class ParticleMergeWithPrescribedArea
	 * @brief merging particles with minimum density error.
	 */
	class MergeWithMinimumDensityErrorWithWall : public MergeWithMinimumDensityErrorInner
	{
	public:
		MergeWithMinimumDensityErrorWithWall(ComplexRelation &complex_relation, BodyRegionByCell &refinement_area)
			: MergeWithMinimumDensityErrorInner(complex_relation.inner_relation_, refinement_area),
			  compute_density_error(complex_relation){};
		virtual ~MergeWithMinimumDensityErrorWithWall(){};

		void interaction(size_t index_i, Real dt = 0.0)
		{
			MergeWithMinimumDensityErrorInner::interaction(index_i, dt);
		};

	protected:
		ComputeDensityErrorWithWall compute_density_error;
	};
}