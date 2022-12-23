/* -----------------------------------------------------------------------------*
 *                               SPHinXsys                                      *
 * -----------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle    *
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for       *
 * physical accurate simulation and aims to model coupled industrial dynamic    *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH      *
 * (smoothed particle hydrodynamics), a meshless computational method using     *
 * particle discretization.                                                     *
 *                                                                              *
 * SPHinXsys is partially funded by German Research Foundation                  *
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1,               *
 * HU1527/12-1 and HU1527/12-4.                                                 *
 *                                                                              *
 * Portions copyright (c) 2017-2022 Technical University of Munich and          *
 * the authors' affiliations.                                                   *
 *                                                                              *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may      *
 * not use this file except in compliance with the License. You may obtain a    *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.           *
 *                                                                              *
 * -----------------------------------------------------------------------------*/
/**
 * @file 	general_operators.h
 * @brief 	This is the particle dynamics applicable for all type bodies
 * @author	Xiangyu Hu
 */

#ifndef GENERAL_OPERATORS_H
#define GENERAL_OPERATORS_H

#include "general_dynamics.h"

namespace SPH
{
    /**
     * @class BaseOperatorInner
     * @brief Base class for spatial operators with inner relation
     */
    template <typename InDataType, typename OutDataType>
    class BaseOperatorInner : public LocalDynamics, public GeneralDataDelegateInner
    {
    public:
        BaseOperatorInner(BaseInnerRelation &inner_relation,
                          const std::string &in_variable_name, const std::string &out_variable_name)
            : LocalDynamics(inner_relation.sph_body_), GeneralDataDelegateInner(inner_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_variable_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_variable_name)) {}
        virtual ~BaseOperatorInner(){};

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
    };

    /**
     * @class BaseOperatorContact
     * @brief Base class for spatial operators with contact relation
     */
    template <typename InDataType, typename OutDataType>
    class BaseOperatorContact : public LocalDynamics, public DataDelegateContact<BaseParticles, BaseParticles>
    {
    public:
        BaseOperatorContact(BaseContactRelation &contact_relation,
                            const std::string &in_variable_name, const std::string &out_variable_name)
            : LocalDynamics(contact_relation.sph_body_),
              DataDelegateContact<BaseParticles, BaseParticles>(contact_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_variable_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_variable_name))
        {
            for (size_t k = 0; k != contact_particles_.size(); ++k)
            {
                contact_in_variable_.push_back(contact_particles_[k]->template getVariableByName<InDataType>(in_variable_name));
            }
        }

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
        StdVec<StdLargeVec<InDataType> *> contact_in_variable_;
    };

    /**
     * @class OperatorConstantCoefficient
     * @brief Base class for computing Laplacian operators with constant coefficient
     * This can be used for computing dissipative terms
     */
    template <typename CoefficientType, class BaseOperatorType>
    class OperatorConstantCoefficient : public BaseOperatorType
    {
    public:
        template <class BodyRelationType, typename... Args>
        OperatorConstantCoefficient(BodyRelationType &body_relation,
                                    const CoefficientType &coefficient, Args &&...args)
            : BaseOperatorType(body_relation, std::forward<Args>(args)...), coefficient_(coefficient){};
        virtual ~OperatorConstantCoefficient(){};

        void interaction(size_t index_i, Real dt = 0.0)
        {
            this->loopNeighbors(index_i, [&](size_t i, size_t j)
                                { return coefficient_; });
        }

    protected:
        CoefficientType coefficient_;
    };

    /**
     * @class BaseOperatorVariableCoefficient
     * @brief
     */
    template <typename CoefficientType, class BaseOperatorType>
    class BaseOperatorVariableCoefficient : public BaseOperatorType
    {
    public:
        template <class BodyRelationType, typename... Args>
        BaseOperatorVariableCoefficient(BodyRelationType &body_relation,
                                        const std::string &coefficient_name, Args &&...args)
            : BaseOperatorType(body_relation, std::forward<Args>(args)...),
              coefficient_(*this->particles_->template getVariableByName<CoefficientType>(coefficient_name)){};
        virtual ~BaseOperatorVariableCoefficient(){};

    protected:
        StdLargeVec<CoefficientType> &coefficient_;
    };

    /**
     * @class OperatorAlgebraAverageCoefficient
     * @brief
     */
    template <typename CoefficientType, class BaseOperatorType>
    class OperatorAlgebraAverageCoefficient : public BaseOperatorVariableCoefficient<CoefficientType, BaseOperatorType>
    {
    public:
        template <class BodyRelationType, typename... Args>
        OperatorAlgebraAverageCoefficient(BodyRelationType &body_relation,
                                          const std::string &coefficient_name, Args &&...args)
            : BaseOperatorVariableCoefficient<CoefficientType, BaseOperatorType>(
                  body_relation, coefficient_name, std::forward<Args>(args)...){};
        virtual ~OperatorAlgebraAverageCoefficient(){};

        void interaction(size_t index_i, Real dt = 0.0)
        {
            this->loopNeighbors(index_i, [&](size_t i, size_t j)
                                { return 0.5 * (this->coefficient_[i] + this->coefficient_[j]); });
        }
    };

    /**
     * @class OperatorGeometryAverageCoefficient
     * @brief
     */
    template <typename CoefficientType, class BaseOperatorType>
    class OperatorGeometryAverageCoefficient : public BaseOperatorVariableCoefficient<CoefficientType, BaseOperatorType>
    {
    public:
        template <class BodyRelationType, typename... Args>
        OperatorGeometryAverageCoefficient(BodyRelationType &body_relation,
                                           const std::string &coefficient_name, Args &&...args)
            : BaseOperatorVariableCoefficient<CoefficientType, BaseOperatorType>(
                  body_relation, coefficient_name, std::forward<Args>(args)...){};
        virtual ~OperatorGeometryAverageCoefficient(){};

        void interaction(size_t index_i, Real dt = 0.0)
        {
            this->loopNeighbors(index_i, [&](size_t i, size_t j)
                                { return 2.0 * this->coefficient_[i] * this->coefficient_[j] /
                                         (this->coefficient_[i] + this->coefficient_[j]); });
        }
    };

    /**
     * @class OperatorOneSideCoefficient
     * @brief
     */
    template <typename CoefficientType, class BaseOperatorType>
    class OperatorOneSideCoefficient : public BaseOperatorVariableCoefficient<CoefficientType, BaseOperatorType>
    {
    public:
        template <class BodyRelationType, typename... Args>
        OperatorOneSideCoefficient(BodyRelationType &body_relation,
                                   const std::string &coefficient_name, Args &&...args)
            : BaseOperatorVariableCoefficient<CoefficientType, BaseOperatorType>(
                  body_relation, coefficient_name, std::forward<Args>(args)...){};
        virtual ~OperatorOneSideCoefficient(){};

        void interaction(size_t index_i, Real dt = 0.0)
        {
            this->loopNeighbors(index_i, [&](size_t i, size_t j)
                                { return this->coefficient_[i]; });
        }
    };
}
#endif // GENERAL_OPERATORS_H