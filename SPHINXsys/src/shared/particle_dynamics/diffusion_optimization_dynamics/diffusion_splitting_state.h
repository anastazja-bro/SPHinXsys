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
* @file 	diffusion_splitting_state.h
* @brief 	This is the method for splitting state variable optimization problem.
* @author   Bo Zhang and Xiangyu Hu
*/

#ifndef DIFFUSION_SPLITTING_STATE_H
#define DIFFUSION_SPLITTING_STATE_H

#include "diffusion_splitting_base.h"
#include "diffusion_splitting_parameter.h"

namespace SPH
{
	/**
	 * @class TemperatureSplittingByPDEInner
	 * @brief The temperature on each particle will be modified inner to satisfy the PDE equation.
	 */
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES = 1>
	class TemperatureSplittingByPDEInner 
		: public OptimizationBySplittingAlgorithmBase<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>
	{
	public:
		TemperatureSplittingByPDEInner(BaseInnerRelation& inner_relation, const std::string& variable_name);
		virtual ~TemperatureSplittingByPDEInner() {};

	protected:
		virtual ErrorAndParameters<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
		virtual void updateStatesByError(size_t index_i, Real dt, const ErrorAndParameters<VariableType>& error_and_parameters);
		virtual void updateWithSourceTerm(size_t index_i, Real dt) { this->variable_[index_i] += this->heat_source_[index_i] * dt; }
		virtual void interaction(size_t index_i, Real dt = 0.0) override;
	};

	/**
	 * @class TemperatureSplitingByPDEWithBoundary
	 * @brief The temperature on each particle will be modified with boundary to satisfy the PDE equation.
	 */
	template <class BaseParticlesType, class BaseMaterialType, class ContactBaseParticlesType, 
		      class ContactBaseMaterialType, typename VariableType, int NUM_SPECIES = 1>
    class TemperatureSplittingByPDEWithBoundary 
		: public TemperatureSplittingByPDEInner<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>,
		  public DiffusionReactionContactData<BaseParticlesType, BaseMaterialType, ContactBaseParticlesType, 
		                                      ContactBaseMaterialType, NUM_SPECIES>
	{
	public:
		TemperatureSplittingByPDEWithBoundary(ComplexRelation& complex_relation, const std::string& variable_name);
		virtual ~TemperatureSplittingByPDEWithBoundary() {};

	protected:
		StdVec<StdLargeVec<VariableType>*> boundary_variable_;
		StdVec<StdLargeVec<Real>*> boundary_normal_distance_;
		StdVec<StdLargeVec<Vecd>*> boundary_normal_vector_;
		virtual ErrorAndParameters<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0) override;
		virtual void updateStatesByError(size_t index_i, Real dt, const ErrorAndParameters<VariableType>& error_and_parameters) override;
	};

	/**
	 * @class UpdateTemperaturePDEResidual
	 * @brief Update the global residual from temperature after one splitting loop finished.
	 */
	template <typename TemperatureSplittingType, typename BaseBodyRelationType, typename VariableType>
	class UpdateTemperaturePDEResidual : public TemperatureSplittingType
	{
	public:
		UpdateTemperaturePDEResidual(BaseBodyRelationType& body_relation, const std::string& variable_name);
		virtual ~UpdateTemperaturePDEResidual() {};

	protected:
		virtual void interaction(size_t index_i, Real dt = 0.0) override;
	};
	
	/**
	 * @class TemperatureSplittingByBCInner
	 * @brief The temperature on each particle will be modified inner to satisfy the BC constraint.
	 *        Used for the boundary particle is within the thermal domain.
	 */
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES = 1>
	class TemperatureSplittingByBCInner
		: public OptimizationBySplittingAlgorithmBase< BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>
	{
	public:
		TemperatureSplittingByBCInner(BaseInnerRelation& inner_relation, const std::string& variable_name);
		virtual ~TemperatureSplittingByBCInner() {};

	protected:
		virtual ErrorAndParameters<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
		virtual void updateStatesByError(size_t index_i, Real dt, const ErrorAndParameters<VariableType>& error_and_parameters);
		virtual void interaction(size_t index_i, Real dt = 0.0) override;
	};

	/**
	 * @class TemperatureSplitingByPDEWithBoundary
	 * @brief The temperature on each particle will be modified with boundary to satisfy the BC constraint.
	 *        Used for the boundary particle is in another body, i.e., dummy particle.
	 */
	template <class BaseParticlesType, class BaseMaterialType, class ContactBaseParticlesType,
		      class ContactBaseMaterialType, typename VariableType, int NUM_SPECIES = 1>
	class TemperatureSplittingByBCWithBoundary
		: public TemperatureSplittingByBCInner<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>,
		  public DiffusionReactionContactData<BaseParticlesType, BaseMaterialType, ContactBaseParticlesType, 
		                                      ContactBaseMaterialType, NUM_SPECIES>
	{
	public:
		TemperatureSplittingByBCWithBoundary(ComplexRelation& complex_relation, const std::string& variable_name);
		virtual ~TemperatureSplittingByBCWithBoundary() {};

	protected:
		StdVec<StdLargeVec<Vecd>*> boundary_normal_vector_;
		StdVec<StdLargeVec<Real>*> boundary_normal_distance_, boundary_diffusivity_;
		StdVec<StdLargeVec<VariableType>*> boundary_variable_;

		virtual ErrorAndParameters<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0) override;
		virtual void updateStatesByError(size_t index_i, Real dt, const ErrorAndParameters<VariableType>& error_and_parameters) override;
		virtual void interaction(size_t index_i, Real dt = 0.0) override;
	};

	/**
	 * @class UpdateTemperaturePDEResidual
	 * @brief Update the global residual from temperature after one splitting loop finished.
	 */
	template <typename TemperatureConstrainType, typename BaseBodyRelationType, typename VariableType>
	class UpdateTemperatureBCResidual : public TemperatureConstrainType
	{
	public:
		UpdateTemperatureBCResidual(BaseBodyRelationType& body_relation, const std::string& variable_name);
		virtual ~UpdateTemperatureBCResidual() {};

	protected:
		virtual void interaction(size_t index_i, Real dt = 0.0) override;
	};

	/**
	 * @class UpdateBoundaryParticleTemperature
	 * @brief Calculate the temperature on boundary particle for Neumann BC.
	 *        Used for the boundary particle is within the thermal domain.
	 */
	template <class BaseParticlesType, class BaseMaterialType, typename VariableType, int NUM_SPECIES = 1>
	class UpdateBoundaryParticleTemperature
		: public LocalDynamics,
		  public DiffusionReactionInnerData<BaseParticlesType, BaseMaterialType, NUM_SPECIES>
	{
	public:
		UpdateBoundaryParticleTemperature(BaseInnerRelation& inner_relation, const std::string& variable_name);
		virtual ~UpdateBoundaryParticleTemperature() {};

	protected:
		size_t phi_;
		Real sigma0_;
		StdLargeVec<Vecd>& normal_vector_;
		StdVec<BaseDiffusion*> species_diffusion_;
		StdLargeVec<Real>& variable_, & heat_flux_;
		virtual void interaction(size_t index_i, Real dt = 0.0);
	};
	 
	/**
	 * @class UpdateBoundaryParticleTemperatureDummy
	 * @brief Calculate the temperature on dummy boundary particles for Neumann BC.
	 *        Used for the boundary particle is in another body, i.e., dummy particle.
	 */
	template<class BaseParticlesType, class BaseMaterialType, class ContactBaseParticlesType, 
		     class ContactBaseMaterialType, class VariableType, int NUM_SPECIES = 1>
	class UpdateBoundaryParticleTemperatureDummy
		: public UpdateBoundaryParticleTemperature<BaseParticlesType, BaseMaterialType, VariableType, NUM_SPECIES>,
		  public DiffusionReactionContactData<BaseParticlesType, BaseMaterialType, ContactBaseParticlesType, 
		                                      ContactBaseMaterialType, NUM_SPECIES>
	{
	public:
		UpdateBoundaryParticleTemperatureDummy(ComplexRelation& body_complex_relation, const std::string& variable_name);
		virtual ~UpdateBoundaryParticleTemperatureDummy() {};

	protected:
		StdVec<StdLargeVec<Vecd>*> boundary_normal_vector_;
		StdVec<StdLargeVec<Real>*> boundary_variable_, boundary_diffusivity_;
		virtual void interaction(size_t index_i, Real dt = 0.0) override;
	};
}

#endif DIFFUSION_SPLITTING_STATE_H