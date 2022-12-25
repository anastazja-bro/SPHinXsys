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
	template <typename DataType, class CoefficientType>
	class BaseDampingPairwiseInner : public OperatorInner<DataType, DataType, CoefficientType>
	{
	public:
		template <typename Arg>
		BaseDampingPairwiseInner(BaseInnerRelation &inner_relation,
								 const std::string &variable_name, const Arg &eta);
		virtual ~BaseDampingPairwiseInner(){};

		void interaction(size_t index_i, Real dt = 0.0);

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<DataType> &variable_;
	};

	/**
	 * @class DampingPairwiseInner
	 * @brief Damping with constant coefficient.
	 */
	template <typename DataType>
	class DampingPairwiseInner
		: public BaseDampingPairwiseInner<DataType, ConstantCoefficient<Real>>
	{
	public:
		DampingPairwiseInner(BaseInnerRelation &inner_relation,
							 const std::string &variable_name, Real eta)
			: BaseDampingPairwiseInner<DataType, ConstantCoefficient<Real>>(
				  inner_relation, variable_name, eta){};
		virtual ~DampingPairwiseInner(){};
	};

	/**
	 * @class DampingPairwiseInnerCoefficientByParticle
	 * @brief Damping with variable coefficient.
	 */
	template <typename DataType>
	class DampingPairwiseInnerCoefficientByParticle
		: public BaseDampingPairwiseInner<DataType, CoefficientByParticle<Real>>
	{
	public:
		DampingPairwiseInnerCoefficientByParticle(
			BaseInnerRelation &inner_relation, const std::string &variable_name,
			const std::string &eta)
			: BaseDampingPairwiseInner<DataType, CoefficientByParticle<Real>>(
				  inner_relation, variable_name, eta){};
		virtual ~DampingPairwiseInnerCoefficientByParticle(){};
	};

	/**
	 * @class CoefficientEvolution
	 * @brief Only works for scalar variable and coefficient.
	 * TODO: to be generalized for different data type
	 */
	class CoefficientEvolution : public LocalDynamics, public DissipationDataInner
	{
	public:
		CoefficientEvolution(BaseInnerRelation &inner_relation,
									const std::string &variable_name, const std::string &eta, Real threshold);
		virtual ~CoefficientEvolution(){};
		void interaction(size_t index_i, Real dt);

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<Real> &variable_;
		StdLargeVec<Real> &eta_; /**< variable damping coefficient */
		Real threshold_;
	};

	/**
	 * @class CoefficientEvolutionExplicit
	 * @brief Only works for scalar variable and coefficient.
	 * TODO: to be generalized for different data type
	 */
	class CoefficientEvolutionExplicit : public LocalDynamics, public DissipationDataInner
	{
	public:
		CoefficientEvolutionExplicit(BaseInnerRelation &inner_relation,
											const std::string &variable_name, const std::string &eta, Real threshold);
		virtual ~CoefficientEvolutionExplicit(){};
		void interaction(size_t index_i, Real dt);
		void update(size_t index_i, Real dt);

	protected:
		StdLargeVec<Real> &rho_;
		StdLargeVec<Real> change_rate_;
		StdLargeVec<Real> &variable_;
		StdLargeVec<Real> &eta_; /**< variable damping coefficient */
		Real threshold_;
	};

	/**
	 * @class CoefficientEvolutionFromWallExplicit
	 * @brief Only works for scalar variable and coefficient.
	 * TODO: to be generalized for different data type
	 */
	class CoefficientEvolutionWithWallExplicit : public CoefficientEvolutionExplicit,
														public DissipationDataWithWall
	{
	public:
		CoefficientEvolutionWithWallExplicit(ComplexRelation &complex_relation,
											const std::string &variable_name, const std::string &eta, Real threshold);
		virtual ~CoefficientEvolutionWithWallExplicit(){};
		void interaction(size_t index_i, Real dt);

	protected:
		StdVec<StdLargeVec<Real> *> wall_variable_;
	};

	/**
	 * @class CoefficientEvolutionFromWall
	 * @brief Only works for scalar variable and coefficient.
	 * TODO: to be generalized for different data type
	 */
	class CoefficientEvolutionFromWall : public LocalDynamics,
												public DataDelegateContact<BaseParticles, SolidParticles>
	{
	public:
		CoefficientEvolutionFromWall(BaseContactRelation &contact_relation,
											const std::string &variable_name, const std::string &eta, Real threshold);
		virtual ~CoefficientEvolutionFromWall(){};
		void interaction(size_t index_i, Real dt);

	protected:
		StdLargeVec<Real> &Vol_, &mass_;
		StdLargeVec<Real> &variable_;
		StdLargeVec<Real> &eta_; /**< variable damping coefficient */
		StdVec<StdLargeVec<Real> *> wall_variable_;
		Real threshold_;
	};

	/**
	 * @class BaseDampingPairwiseFromWall
	 * @brief Damping to wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <typename DataType, class CoefficientType>
	class BaseDampingPairwiseFromWall : public OperatorFromBoundary<DataType, DataType, CoefficientType>
	{
	public:
		template <typename Arg>
		BaseDampingPairwiseFromWall(BaseContactRelation &contact_relation,
									const std::string &variable_name, const Arg &eta);
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
	class DampingPairwiseFromWall
		: public BaseDampingPairwiseFromWall<DataType, ConstantCoefficient<Real>>
	{
	public:
		DampingPairwiseFromWall(BaseContactRelation &contact_relation,
								const std::string &variable_name, Real eta)
			: BaseDampingPairwiseFromWall<DataType, ConstantCoefficient<Real>>(
				  contact_relation, variable_name, eta){};
		virtual ~DampingPairwiseFromWall(){};
	};

	/**
	 * @class DampingPairwiseFromWallCoefficientByParticle
	 * @brief Damping to wall by which the wall velocity is not updated
	 * and the mass of wall particle is not considered.
	 */
	template <typename DataType>
	class DampingPairwiseFromWallCoefficientByParticle
		: public BaseDampingPairwiseFromWall<DataType, CoefficientByParticle<Real>>
	{
	public:
		DampingPairwiseFromWallCoefficientByParticle(
			BaseContactRelation &contact_relation,
			const std::string &variable_name, const std::string &eta)
			: BaseDampingPairwiseFromWall<DataType, CoefficientByParticle<Real>>(
				  contact_relation, variable_name, eta){};
		virtual ~DampingPairwiseFromWallCoefficientByParticle(){};
	};

	/**
	 * @class DampingWithWall
	 * @brief Damping with wall with the priority for update operator to wall first.
	 */
	template <class BaseDampingType, class DampingFromWallType>
	class DampingWithWall : public LocalDynamics
	{
	public:
		template <class BodyRelationType, typename Arg>
		DampingWithWall(BodyRelationType &body_relation,
						BaseContactRelation &relation_to_boundary,
						const std::string &variable_name, const Arg &eta)
			: LocalDynamics(body_relation.sph_body_),
			  base_operator_(body_relation, variable_name, eta),
			  damping_from_wall_(relation_to_boundary, variable_name, eta){};
		template <typename Arg, typename... ExtraCoefficientArg>
		DampingWithWall(ComplexRelation &complex_relation,
						const std::string &variable_name, const Arg &eta)
			: DampingWithWall(complex_relation.getInnerRelation(),
							  complex_relation.getContactRelation(),
							  variable_name, eta){};
		virtual ~DampingWithWall(){};

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