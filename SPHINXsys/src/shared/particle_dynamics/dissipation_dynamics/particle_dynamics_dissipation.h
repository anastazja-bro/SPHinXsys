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
 * @file 	particle_dynamics_dissipation.h
 * @brief 	Here are the classes for damping the magnitude of
 * 			any variables.
 * 			Note that, currently, these classes works only in single resolution.
 * @author	Bo Zhang, Chi ZHang and Xiangyu Hu
 */

#ifndef PARTICLE_DYNAMICS_DISSIPATION_H
#define PARTICLE_DYNAMICS_DISSIPATION_H

#include "all_particle_dynamics.h"
#include "solid_particles.h"

namespace SPH
{
	typedef DataDelegateInner<BaseParticles> DissipationDataInner;
	typedef DataDelegateContact<BaseParticles, BaseParticles, DataDelegateEmptyBase>
		DissipationDataContact;
	typedef DataDelegateContact<BaseParticles, SolidParticles, DataDelegateEmptyBase>
		DissipationDataWithWall;

	template <typename VariableType>
	struct ErrorAndParameters
	{
		VariableType error_;
		Real a_, c_;
		ErrorAndParameters() : error_(ZeroData<VariableType>::value), a_(0), c_(0){};
	};

	/**
	 * @class DampingBySplittingAlgorithm
	 * @brief A quantity damping by splitting scheme
	 * this method modifies the quantity directly.
	 * Note that, if periodic boundary condition is applied,
	 * the parallelized version of the method requires the one using ghost particles
	 * because the splitting partition only works in this case.
	 */
	template <typename VariableType>
	class DampingBySplittingInner : public LocalDynamics, public DissipationDataInner
	{
	protected:
	public:
		DampingBySplittingInner(BaseInnerRelation &inner_relation, const std::string &variable_name, Real eta);
		virtual ~DampingBySplittingInner(){};
		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		Real eta_; /**< damping coefficient */
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<VariableType> &variable_;

		virtual ErrorAndParameters<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
		virtual void updateStates(size_t index_i, Real dt, const ErrorAndParameters<VariableType> &error_and_parameters);
	};

	template <typename VariableType>
	class DampingBySplittingComplex : public DampingBySplittingInner<VariableType>, public DissipationDataContact
	{
	public:
		DampingBySplittingComplex(ComplexRelation &complex_relation, const std::string &variable_name, Real eta);
		virtual ~DampingBySplittingComplex(){};

	protected:
		virtual ErrorAndParameters<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0) override;
		virtual void updateStates(size_t index_i, Real dt, const ErrorAndParameters<VariableType> &error_and_parameters) override;

	private:
		StdVec<StdLargeVec<Real> *> contact_Vol_, contact_mass_;
		StdVec<StdLargeVec<VariableType> *> contact_variable_;
	};

	template <typename VariableType,
			  template <typename BaseVariableType>
			  class BaseDampingBySplittingType>
	class DampingBySplittingWithWall : public BaseDampingBySplittingType<VariableType>, public DissipationDataWithWall
	{
	public:
		DampingBySplittingWithWall(ComplexRelation &complex_wall_relation, const std::string &variable_name, Real eta);
		virtual ~DampingBySplittingWithWall(){};

	protected:
		virtual ErrorAndParameters<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0) override;

	private:
		StdVec<StdLargeVec<Real> *> wall_Vol_;
		StdVec<StdLargeVec<VariableType> *> wall_variable_;
	};

	/**
	 * @class BaseDampingPairwiseInner
	 * @brief Base class for a quantity damping by a pairwise splitting scheme
	 * this method modifies the quantity directly
	 * Note that, if periodic boundary condition is applied for a simulation,
	 * the parallelized version of the method requires the one using ghost particles
	 * because the splitting partition only works in this case.
	 */
	template <typename VariableType>
	class BaseDampingPairwiseInner : public LocalDynamics, public DissipationDataInner
	{
	public:
		BaseDampingPairwiseInner(BaseInnerRelation &inner_relation, const std::string &variable_name);
		virtual ~BaseDampingPairwiseInner(){};

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<VariableType> &variable_;

		template <typename InterParticleCoefficient>
		void dampPairwiseInner(size_t index_i, Real dt, const InterParticleCoefficient &coefficient);
	};

	/**
	 * @class DampingPairwiseInner
	 * @brief Damping with constant coefficient.
	 */
	template <typename VariableType>
	class DampingPairwiseInner : public BaseDampingPairwiseInner<VariableType>
	{
	public:
		DampingPairwiseInner(BaseInnerRelation &inner_relation, const std::string &variable_name, Real eta);
		virtual ~DampingPairwiseInner(){};
		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		Real eta_; /**< damping coefficient */
	};

	/**
	 * @class DampingPairwiseInnerVariableCoefficient
	 * @brief Damping with constant coefficient.
	 */
	template <typename VariableType>
	class DampingPairwiseInnerVariableCoefficient : public BaseDampingPairwiseInner<VariableType>
	{
	public:
		DampingPairwiseInnerVariableCoefficient(
			BaseInnerRelation &inner_relation,
			const std::string &variable_name, const std::string &coefficient_name);
		virtual ~DampingPairwiseInnerVariableCoefficient(){};
		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		StdLargeVec<Real> &eta_; /**< variable damping coefficient */
	};

	/**
	 * @class DampingCoefficientEvolution
	 * @brief Only works for scalar variable and coefficient. 
	 * TODO: to be generalized for different data type
	 */
	class DampingCoefficientEvolution : public LocalDynamics, public DissipationDataInner
	{
	public:
		DampingCoefficientEvolution(BaseInnerRelation &inner_relation, 
		const std::string &variable_name, const std::string &coefficient_name);
		virtual ~DampingCoefficientEvolution(){};
		void interaction(size_t index_i, Real dt);

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<Real> &variable_;
		StdLargeVec<Real> &eta_; /**< variable damping coefficient */
	};

	/**
	 * @class BaseDampingPairwiseComplex
	 * @brief Damping between contact bodies with constant coefficient.
	 * TODO: not tested yet.
	 */
	template <typename VariableType>
	class BaseDampingPairwiseComplex : public BaseDampingPairwiseInner<VariableType>, public DissipationDataContact
	{
	public:
		BaseDampingPairwiseComplex(BaseInnerRelation &inner_relation,
								   BaseContactRelation &contact_relation, const std::string &variable_name);
		BaseDampingPairwiseComplex(ComplexRelation &complex_relation, const std::string &variable_name);
		virtual ~BaseDampingPairwiseComplex(){};

	private:
		StdVec<StdLargeVec<Real> *> contact_mass_;
		StdVec<StdLargeVec<VariableType> *> contact_variable_;

		template <typename InterParticleCoefficient>
		void dampPairwiseContact(size_t index_i, Real dt, const InterParticleCoefficient &coefficient);
	};

	/**
	 * @class BaseDampingPairwiseFromWall
	 * @brief Damping to wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <typename VariableType>
	class BaseDampingPairwiseFromWall : public LocalDynamics,
										public DataDelegateContact<BaseParticles, SolidParticles>
	{
	public:
		BaseDampingPairwiseFromWall(BaseContactRelation &contact_relation, const std::string &variable_name);
		virtual ~BaseDampingPairwiseFromWall(){};

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<VariableType> &variable_;
		StdVec<StdLargeVec<VariableType> *> wall_variable_;

		template <typename InterParticleCoefficient>
		void dampPairwiseFromWall(size_t index_i, Real dt, const InterParticleCoefficient &coefficient);
	};

	/**
	 * @class DampingPairwiseFromWall
	 * @brief Damping to wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <typename VariableType>
	class DampingPairwiseFromWall : public BaseDampingPairwiseFromWall<VariableType>
	{
	public:
		DampingPairwiseFromWall(BaseContactRelation &contact_relation, const std::string &variable_name, Real eta);
		virtual ~DampingPairwiseFromWall(){};
		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		Real eta_; /**< damping coefficient */
	};

	/**
	 * @class DampingPairwiseFromWall
	 * @brief Damping to wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <typename VariableType>
	class DampingPairwiseFromWallVariableCoefficient : public BaseDampingPairwiseFromWall<VariableType>
	{
	public:
		DampingPairwiseFromWallVariableCoefficient(
			BaseContactRelation &contact_relation,
			const std::string &variable_name, const std::string &coefficient_name);
		virtual ~DampingPairwiseFromWallVariableCoefficient(){};
		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		StdLargeVec<Real> &eta_; /**< variable damping coefficient */
	};

	/**
	 * @class DampingComplex
	 * @brief Damping with wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <class BaseDampingInnerType, class BaseDampingContactType>
	class DampingComplex : public LocalDynamics
	{
		BaseDampingInnerType inner_interaction_;
		BaseDampingContactType contact_interaction_;

	public:
		template <typename... Args>
		DampingComplex(BaseInnerRelation &inner_relation,
					   BaseContactRelation &contact_relation, Args &&...args)
			: LocalDynamics(inner_relation.sph_body_),
			  inner_interaction_(inner_relation, std::forward<Args>(args)...),
			  contact_interaction_(contact_relation, std::forward<Args>(args)...){};
		template <typename... Args>
		DampingComplex(ComplexRelation &complex_wall_relation, Args &&...args)
			: DampingComplex(complex_wall_relation.getInnerRelation(),
							 complex_wall_relation.getContactRelation(),
							 std::forward<Args>(args)...){};
		virtual ~DampingComplex(){};
		void interaction(size_t index_i, Real dt = 0.0)
		{
			contact_interaction_.interaction(index_i, dt);
			inner_interaction_.interaction(index_i, dt);
		};
	};

	/**
	 * @class DampingWithRandomChoice
	 * @brief A random choice method for obtaining static equilibrium state
	 * Note that, if periodic boundary condition is applied,
	 * the parallelized version of the method requires the one using ghost particles
	 * because the splitting partition only works in this case.
	 */
	template <class DampingAlgorithmType>
	class DampingWithRandomChoice : public DampingAlgorithmType
	{
	protected:
		Real random_ratio_;
		bool RandomChoice();

	public:
		template <typename... ConstructorArgs>
		DampingWithRandomChoice(Real random_ratio, ConstructorArgs &&...args);
		virtual ~DampingWithRandomChoice(){};

		virtual void exec(Real dt = 0.0) override;
		virtual void parallel_exec(Real dt = 0.0) override;
	};

	template <typename VariableType>
	struct ErrorAndParametersConserved
	{
		VariableType error_;
		VariableType wall_flux_;
		Real a_, c_, d_, e_;
		ErrorAndParametersConserved() : error_(ZeroData<VariableType>::value),
			                            wall_flux_(ZeroData<VariableType>::value),
			                            a_(0), c_(0), d_(0), e_(0) {};
	};

	/**
	 * @class DampingBySplittingAlgorithm
	 * @brief A quantity damping by splitting scheme
	 * this method modifies the quantity directly.
	 * Note that, if periodic boundary condition is applied,
	 * the parallelized version of the method requires the one using ghost particles
	 * because the splitting partition only works in this case.
	 */
	template <typename VariableType>
	class DampingByConservedSplittingInner : public LocalDynamics, public DissipationDataInner
	{
	protected:
	public:
		DampingByConservedSplittingInner(BaseInnerRelation& inner_relation, const std::string& variable_name, Real eta);
		virtual ~DampingByConservedSplittingInner() {};
		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		Real eta_; /**< damping coefficient */
		StdLargeVec<Real>& Vol_, & mass_;
		StdLargeVec<VariableType>& variable_;

		virtual ErrorAndParametersConserved<VariableType> computeErrorAndParameters(size_t index_i, Real dt = 0.0);
		virtual void updateStates(size_t index_i, Real dt, const ErrorAndParametersConserved<VariableType>& error_and_parameters);
	};
}
#endif // PARTICLE_DYNAMICS_DISSIPATION_H