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
    };

    /**
     * @class OperatorInner
     * @brief Base class for spatial operators with inner relation
     */
    template <typename InDataType, typename OutDataType, class CoefficientType>
    class OperatorInner : public LocalDynamics, public GeneralDataDelegateInner
    {
    public:
        template <typename Arg>
        OperatorInner(BaseInnerRelation &inner_relation,
                      const std::string &in_name, const std::string &out_name, const Arg &eta)
            : LocalDynamics(inner_relation.sph_body_), GeneralDataDelegateInner(inner_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_name)),
              coefficient_(particles_, eta){};
        virtual ~OperatorInner(){};

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
        CoefficientType coefficient_;
    };

    /**
     * @class OperatorContact
     * @brief Base class for spatial operators with contact relation
     */
    template <typename InDataType, typename OutDataType, class CoefficientType>
    class OperatorContact : public LocalDynamics, public DataDelegateContact<BaseParticles, BaseParticles>
    {
    public:
        template <typename Arg, typename ContactArg>
        OperatorContact(BaseContactRelation &contact_relation,
                        const std::string &in_name, const std::string &out_name,
                        const Arg &eta, const ContactArg &contact_eta)
            : LocalDynamics(contact_relation.sph_body_),
              DataDelegateContact<BaseParticles, BaseParticles>(contact_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_name))
        {
            for (size_t k = 0; k != contact_particles_.size(); ++k)
            {
                contact_in_variable_.push_back(contact_particles_[k]->template getVariableByName<InDataType>(in_name));
                contact_coefficient_.push_back(particles_, contact_particles_[k], eta, contact_eta);
            }
        }
        virtual ~OperatorContact(){};

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
        StdVec<StdLargeVec<InDataType> *> contact_in_variable_;
        StdVec<CoefficientType> &contact_coefficient_;
    };

    /**
     * @class OperatorFromBoundary
     * @brief Base class for spatial operators with contact relation
     */
    template <typename InDataType, typename OutDataType, class CoefficientType>
    class OperatorFromBoundary : public LocalDynamics, public DataDelegateContact<BaseParticles, BaseParticles>
    {
    public:
        template <typename Arg>
        OperatorFromBoundary(BaseContactRelation &contact_relation,
                             const std::string &in_name, const std::string &out_name, const Arg &eta)
            : LocalDynamics(contact_relation.sph_body_),
              DataDelegateContact<BaseParticles, BaseParticles>(contact_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_name)),
              coefficient_(particles_, eta)
        {
            for (size_t k = 0; k != contact_particles_.size(); ++k)
            {
                contact_in_variable_.push_back(contact_particles_[k]->template getVariableByName<InDataType>(in_name));
            }
        }
        virtual ~OperatorFromBoundary(){};

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
        StdVec<StdLargeVec<InDataType> *> contact_in_variable_;
        CoefficientType coefficient_;
    };

    /**
     * @class OperatorWithBoundary
     * @brief Base class for spatial operators with contact relation
     */
    template <class BaseOperatorType, class OperatorFromBoundaryType>
    class OperatorWithBoundary : public LocalDynamics
    {
    public:
        template <class BodyRelationType, typename Arg, typename... ContactArg>
        OperatorWithBoundary(BodyRelationType &body_relation, BaseContactRelation &relation_to_boundary,
                             const std::string &in_name, const std::string &out_name,
                             const Arg &eta, ContactArg &&...contact_eta)
            : LocalDynamics(body_relation.sph_body_),
              base_operator_(body_relation, in_name, out_name, eta,
                             std::forward<ContactArg>(contact_eta)...),
              operator_from_boundary_(relation_to_boundary, in_name, out_name, eta){};
        template <typename Arg, typename... ContactArg>
        OperatorWithBoundary(ComplexRelation &complex_relation,
                             const std::string &in_name, const std::string &out_name,
                             const Arg &eta, ContactArg &&...contact_eta)
            : OperatorWithBoundary(complex_relation.getInnerRelation(),
                                   complex_relation.getContactRelation(),
                                   in_name, out_name, eta,
                                   std::forward<ContactArg>(contact_eta)...){};
        virtual ~OperatorWithBoundary(){};

        void interaction(size_t index_i, Real dt)
        {
            this->base_operator_.interaction(index_i, dt);
            this->operator_from_boundary_.interaction(index_i, dt);
        };

    protected:
        BaseOperatorType base_operator_;
        OperatorFromBoundaryType operator_from_boundary_;
    };
}
#endif // GENERAL_OPERATORS_H