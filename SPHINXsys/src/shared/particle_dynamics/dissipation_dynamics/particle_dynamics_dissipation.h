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
 * @brief 	Here are the classes for damping the magnitude of any discrete variable.
 * 			Note that, currently, these classes works only in single resolution.
 * @author	Chi ZHang and Xiangyu Hu
 */

#ifndef PARTICLE_DYNAMICS_DISSIPATION_H
#define PARTICLE_DYNAMICS_DISSIPATION_H

#include "all_particle_dynamics.h"
#include "general_operators.h"
#include "solid_particles.h"

namespace SPH
{
	typedef DataDelegateInner<BaseParticles> DissipationDataInner;
	typedef DataDelegateContact<BaseParticles, BaseParticles, DataDelegateEmptyBase>
		DissipationDataContact;
	typedef DataDelegateContact<BaseParticles, SolidParticles, DataDelegateEmptyBase>
		DissipationDataWithWall;

	template <typename DataType>
	struct ErrorAndParameters
	{
		DataType error_;
		Real a_, c_;
		ErrorAndParameters() : error_(ZeroData<DataType>::value), a_(0), c_(0){};
	};

	/**
	 * @class BaseDampingSplittingInner
	 * @brief A quantity damping by splitting scheme
	 * this method modifies the quantity directly.
	 * Note that, if periodic boundary condition is applied,
	 * the parallelized version of the method requires the one using ghost particles
	 * because the splitting partition only works in this case.
	 */
	template <typename DataType, class CoefficientType>
	class BaseDampingSplittingInner : public OperatorInner<DataType, DataType, CoefficientType>
	{
	public:
		template <typename CoefficientArg>
		BaseDampingSplittingInner(BaseInnerRelation &inner_relation,
								  const std::string &variable_name, const CoefficientArg &eta);
		virtual ~BaseDampingSplittingInner(){};
		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<DataType> &variable_;

		virtual ErrorAndParameters<DataType> computeErrorAndParameters(size_t index_i, Real dt);
		virtual void updateStates(size_t index_i, Real dt, const ErrorAndParameters<DataType> &error_and_parameters);
	};

	template <typename DataType>
	using DampingSplittingInner = BaseDampingSplittingInner<DataType, ConstantCoefficient<Real>>;
	template <typename DataType>
	using DampingSplittingInnerCoefficientByParticle = BaseDampingSplittingInner<DataType, CoefficientByParticle<Real>>;

	/**
	 * @class BaseDampingSplittingWithWall
	 * @brief Note that the values on the wall are constrained. 
	 */
	template <typename DataType, class CoefficientType>
	class BaseDampingSplittingWithWall : public BaseDampingSplittingInner<DataType, CoefficientType>, public DissipationDataWithWall
	{
	public:
		template <typename CoefficientArg>
		BaseDampingSplittingWithWall(ComplexRelation &complex_wall_relation,
									 const std::string &variable_name, const CoefficientArg &eta);
		virtual ~BaseDampingSplittingWithWall(){};

	protected:
		virtual ErrorAndParameters<DataType> computeErrorAndParameters(size_t index_i, Real dt = 0.0) override;

	private:
		StdVec<StdLargeVec<DataType> *> wall_variable_;
	};

	template <typename DataType>
	using DampingSplittingWithWall = BaseDampingSplittingWithWall<DataType, ConstantCoefficient<Real>>;
	template <typename DataType>
	using DampingSplittingWithWallCoefficientByParticle = BaseDampingSplittingWithWall<DataType, CoefficientByParticle<Real>>;

	/**
	 * @class BaseDampingPairwiseInner
	 * @brief Base class for a quantity damping by a pairwise splitting scheme
	 * this method modifies the quantity directly
	 * Note that, if periodic boundary condition is applied for a simulation,
	 * the parallelized version of the method requires the one using ghost particles
	 * because the splitting partition only works in this case.
	 */
	template <typename DataType, class CoefficientType>
	class BaseDampingPairwiseInner : public OperatorInner<DataType, DataType, CoefficientType>
	{
	public:
		template <typename CoefficientArg>
		BaseDampingPairwiseInner(BaseInnerRelation &inner_relation,
								 const std::string &variable_name, const CoefficientArg &eta);
		virtual ~BaseDampingPairwiseInner(){};

		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<DataType> &variable_;
	};

	template <typename DataType>
	using DampingPairwiseInner = BaseDampingPairwiseInner<DataType, ConstantCoefficient<Real>>;
	template <typename DataType>
	using DampingPairwiseInnerCoefficientByParticle = BaseDampingPairwiseInner<DataType, CoefficientByParticle<Real>>;

	/**
	 * @class BaseDampingPairwiseFromWall
	 * @brief Damping to wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <typename DataType, class CoefficientType>
	class BaseDampingPairwiseFromWall : public OperatorFromBoundary<DataType, DataType, CoefficientType>
	{
	public:
		template <typename CoefficientArg>
		BaseDampingPairwiseFromWall(BaseContactRelation &contact_relation,
									const std::string &variable_name, const CoefficientArg &eta);
		virtual ~BaseDampingPairwiseFromWall(){};
		void interaction(size_t index_i, Real dt);

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<DataType> &variable_;
		StdVec<StdLargeVec<DataType> *> &wall_variable_;
	};

	/**
	 * @class DampingPairwiseFromWall
	 * @brief Damping to wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <typename DataType>
	using DampingPairwiseFromWall = BaseDampingPairwiseFromWall<DataType, ConstantCoefficient<Real>>;
	template <typename DataType>
	using DampingPairwiseFromWallCoefficientByParticle = BaseDampingPairwiseFromWall<DataType, CoefficientByParticle<Real>>;

	/**
	 * @class DampingPairwiseWithWall
	 * @brief Damping with wall with the priority for update operator to wall first.
	 */
	template <class BaseDampingType, class DampingFromWallType>
	class DampingPairwiseWithWall : public LocalDynamics
	{
	public:
		template <class BodyRelationType, typename... Args>
		DampingPairwiseWithWall(BodyRelationType &body_relation,
						BaseContactRelation &relation_to_boundary, Args &&...args)
			: LocalDynamics(body_relation.sph_body_),
			  base_operator_(body_relation, std::forward<Args>(args)...),
			  damping_from_wall_(relation_to_boundary, std::forward<Args>(args)...){};
		template <typename... Args>
		DampingPairwiseWithWall(ComplexRelation &complex_relation, Args &&...args)
			: DampingPairwiseWithWall(complex_relation.getInnerRelation(),
							  complex_relation.getContactRelation(), std::forward<Args>(args)...){};
		virtual ~DampingPairwiseWithWall(){};

		void interaction(size_t index_i, Real dt)
		{
			damping_from_wall_.interaction(index_i, dt);
			base_operator_.interaction(index_i, dt);
		};

	protected:
		BaseDampingType base_operator_;
		DampingFromWallType damping_from_wall_;
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
}
#endif // PARTICLE_DYNAMICS_DISSIPATION_H