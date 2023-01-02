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
 * @brief 	Generalization of the operators acting on discrete variables.
 * @author	Xiangyu Hu
 */

#ifndef GENERAL_OPERATORS_H
#define GENERAL_OPERATORS_H

#include "general_dynamics.h"

namespace SPH
{
    template <typename DataType>
    class ConstantCoefficient
    {
        const DataType eta_i_;
        const DataType eta_j_;

    public:
        template <class ParticlesType>
        ConstantCoefficient(ParticlesType *particles, const DataType &eta)
            : eta_i_(eta), eta_j_(eta){};
        template <class ParticlesType>
        ConstantCoefficient(ParticlesType *particles, const DataType &eta, const DataType &contact_eta)
            : eta_i_(eta), eta_j_(contact_eta){};

        inline DataType operator()(size_t index_i, size_t index_j)
        {
            return 0.5 * (eta_i_ + eta_j_);
        };

        inline DataType operator()(size_t index_i)
        {
            return eta_i_;
        };

    };

    template <typename DataType>
    class CoefficientByParticle
    {
        const StdLargeVec<DataType> &eta_i_;
        const StdLargeVec<DataType> &eta_j_;

    public:
        template <class ParticlesType>
        CoefficientByParticle(ParticlesType *particles, const std::string &name)
            : eta_i_(*particles->template getVariableByName<DataType>(name)),
              eta_j_(*particles->template getVariableByName<DataType>(name)){};
        template <class InnerParticlesType, class ContactParticleType>
        CoefficientByParticle(InnerParticlesType *inner_particles,
                              ContactParticleType *contact_particles, const std::string name)
            : eta_i_(*inner_particles->template getVariableByName<DataType>(name)),
              eta_j_(*contact_particles->template getVariableByName<DataType>(name)){};

        inline DataType operator()(size_t index_i, size_t index_j)
        {
            return 0.5 * (eta_i_[index_i] + eta_j_[index_j]);
        }

        inline DataType operator()(size_t index_i)
        {
            return eta_i_[index_i];
        }
    };

    /**
     * @class BaseOperator
     * @brief Base class for spatial operators
     */
    template <typename DataDelegationType, typename InDataType, typename OutDataType, class CoefficientType>
    class BaseOperator : public LocalDynamics, public DataDelegationType
    {
    public:
        template <typename BodyRelationType, typename CoefficientArg>
        BaseOperator(BodyRelationType &body_relation,
                     const std::string &in_name, const std::string &out_name, const CoefficientArg &eta)
            : LocalDynamics(body_relation.sph_body_), DataDelegationType(body_relation),
              in_variable_(*this->particles_->template getVariableByName<InDataType>(in_name)),
              out_variable_(*this->particles_->template getVariableByName<OutDataType>(out_name)),
              coefficient_(this->particles_, eta){};
        virtual ~BaseOperator(){};
        StdLargeVec<InDataType> &InVariable() { return in_variable_; }
        StdLargeVec<OutDataType> &OutVariable() { return out_variable_; }

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
        CoefficientType coefficient_;
    };

    /**
     * @class OperatorInner
     * @brief Base class for spatial operators with inner relation
     */
    template <typename... DataTypes>
    class OperatorInner : public BaseOperator<GeneralDataDelegateInner, DataTypes...>
    {
    public:
        template <typename... Args>
        OperatorInner(BaseInnerRelation &inner_relation, Args &&...args)
            : BaseOperator<GeneralDataDelegateInner, DataTypes...>(
                  inner_relation, std::forward<Args>(args)...){};
        virtual ~OperatorInner(){};
    };

    /**
     * @class OperatorContact
     * @brief Base class for spatial operators with contact relation
     */
    template <typename InDataType, typename OutDataType, class CoefficientType>
    class OperatorContact : public BaseOperator<GeneralDataDelegateContact, InDataType, OutDataType, CoefficientType>
    {
    public:
        template <typename CoefficientArg, typename ContactCoefficientArg>
        OperatorContact(BaseContactRelation &contact_relation,
                        const std::string &in_name, const std::string &out_name,
                        const CoefficientArg &eta, const ContactCoefficientArg &contact_eta)
            : BaseOperator<GeneralDataDelegateContact, InDataType, OutDataType, CoefficientType>(
                  contact_relation, in_name, out_name, eta, contact_eta)
        {
            auto particles = this->particles_;
            auto contact_particles = this->contact_particles_;
            for (size_t k = 0; k != contact_particles.size(); ++k)
            {
                contact_in_variable_.push_back(contact_particles[k]->template getVariableByName<InDataType>(in_name));
                contact_coefficient_.push_back(particles, contact_particles, eta, contact_eta);
            }
        }
        virtual ~OperatorContact(){};

    protected:
        StdVec<StdLargeVec<InDataType> *> contact_in_variable_;
        StdVec<CoefficientType> &contact_coefficient_;
    };

    /**
     * @class OperatorFromBoundary
     * @brief Base class for spatial operators with contact relation
     */
    template <typename InDataType, typename... OtherDataTypes>
    class OperatorFromBoundary : public BaseOperator<GeneralDataDelegateContact, InDataType, OtherDataTypes...>
    {
    public:
        template <typename... Args>
        OperatorFromBoundary(BaseContactRelation &contact_relation, const std::string &in_name, Args &&...args)
            : BaseOperator<GeneralDataDelegateContact, InDataType, OtherDataTypes...>(
                  contact_relation, in_name, std::forward<Args>(args)...)
        {
            auto contact_particles = this->contact_particles_;
            for (size_t k = 0; k != contact_particles.size(); ++k)
            {
                contact_in_variable_.push_back(contact_particles[k]->template getVariableByName<InDataType>(in_name));
            }
        }
        virtual ~OperatorFromBoundary(){};

    protected:
        StdVec<StdLargeVec<InDataType> *> contact_in_variable_;
    };

    /**
     * @class OperatorWithBoundary
     * @brief Base class for spatial operators with contact relation
     */
    template <class BaseOperatorType, class OperatorFromBoundaryType>
    class OperatorWithBoundary : public LocalDynamics
    {
    public:
        template <class BodyRelationType, typename CoefficientArg, typename... ContactCoefficientArg>
        OperatorWithBoundary(BodyRelationType &body_relation, BaseContactRelation &relation_to_boundary,
                             const std::string &in_name, const std::string &out_name,
                             const CoefficientArg &eta, ContactCoefficientArg &&...contact_eta)
            : LocalDynamics(body_relation.sph_body_),
              base_operator_(body_relation, in_name, out_name, eta,
                             std::forward<ContactCoefficientArg>(contact_eta)...),
              operator_from_boundary_(relation_to_boundary, in_name, out_name, eta){};
        template <typename... Args>
        OperatorWithBoundary(ComplexRelation &complex_relation, Args &&...args)
            : OperatorWithBoundary(complex_relation.getInnerRelation(),
                                   complex_relation.getContactRelation(),
                                   std::forward<Args>(args)...){};
        virtual ~OperatorWithBoundary(){};

        void interaction(size_t index_i, Real dt)
        {
            base_operator_.interaction(index_i, dt);
            operator_from_boundary_.interaction(index_i, dt);
        };

    protected:
        BaseOperatorType base_operator_;
        OperatorFromBoundaryType operator_from_boundary_;
    };
}
#endif // GENERAL_OPERATORS_H