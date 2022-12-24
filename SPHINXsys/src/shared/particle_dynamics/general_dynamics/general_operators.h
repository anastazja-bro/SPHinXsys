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
        CoefficientByParticle(ParticlesType *particles, const std::string &coefficient_name)
            : eta_i_(*particles->template getVariableByName<DataType>(coefficient_name)),
              eta_j_(*particles->template getVariableByName<DataType>(coefficient_name)){};
        template <class InnerParticlesType, class ContactParticleType>
        CoefficientByParticle(InnerParticlesType *inner_particles, ContactParticleType *contact_particles,
                              const std::string coefficient_name)
            : eta_i_(*inner_particles->template getVariableByName<DataType>(coefficient_name)),
              eta_j_(*contact_particles->template getVariableByName<DataType>(coefficient_name)){};
        inline DataType operator()(size_t index_i, size_t index_j)
        {
            return 0.5 * (eta_i_[index_i] + eta_j_[index_j]);
        }
    };

    template <typename DataType>
    class ConstantSource
    {
        const DataType s_;

    public:
        template <class ParticlesType>
        ConstantSource(ParticlesType *particles, const DataType &s = ZeroData<DataType>::value) : s_(s){};
        inline DataType operator()(size_t index_i)
        {
            return s_;
        };
    };

    template <typename DataType>
    class SourceByParticle
    {
        const StdLargeVec<DataType> &s_;

    public:
        template <class ParticlesType>
        SourceByParticle(ParticlesType *particles, const std::string &source_name)
            : s_(*particles->template getVariableByName<DataType>(source_name)){};
        inline DataType operator()(size_t index_i)
        {
            return s_[index_i];
        }
    };

    /**
     * @class OperatorInner
     * @brief Base class for spatial operators with inner relation
     */
    template <typename InDataType, typename OutDataType,
              template <typename SourceDataType> class SourceType, class CoefficientType>
    class OperatorInner : public LocalDynamics, public GeneralDataDelegateInner
    {
    public:
        template <typename SourceArg, typename CoefficientArg>
        OperatorInner(BaseInnerRelation &inner_relation,
                      const std::string &in_variable_name, const std::string &out_variable_name,
                      const SourceArg &source_arg, const CoefficientArg &coefficient_arg)
            : LocalDynamics(inner_relation.sph_body_), GeneralDataDelegateInner(inner_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_variable_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_variable_name)),
              source_(particles_, source_arg), coefficient_(particles_, coefficient_arg){};
        virtual ~OperatorInner(){};

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
        CoefficientType coefficient_;
        SourceType<OutDataType> source_;
    };

    /**
     * @class OperatorContact
     * @brief Base class for spatial operators with contact relation
     */
    template <typename InDataType, typename OutDataType, class CoefficientType>
    class OperatorContact : public LocalDynamics, public DataDelegateContact<BaseParticles, BaseParticles>
    {
    public:
        template <typename CoefficientArg, typename ContactCoefficientArg>
        OperatorContact(BaseContactRelation &contact_relation,
                        const std::string &in_variable_name, const std::string &out_variable_name,
                        const CoefficientArg &coefficient_arg, const ContactCoefficientArg &contact_coefficient_arg)
            : LocalDynamics(contact_relation.sph_body_),
              DataDelegateContact<BaseParticles, BaseParticles>(contact_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_variable_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_variable_name))
        {
            for (size_t k = 0; k != contact_particles_.size(); ++k)
            {
                contact_in_variable_.push_back(contact_particles_[k]->template getVariableByName<InDataType>(in_variable_name));
                contact_coefficient_.push_back(particles_, contact_particles_[k], coefficient_arg, contact_coefficient_arg);
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
        template <typename CoefficientArg>
        OperatorFromBoundary(BaseContactRelation &contact_relation,
                             const std::string &in_variable_name, const std::string &out_variable_name,
                             const CoefficientArg &coefficient_arg)
            : LocalDynamics(contact_relation.sph_body_),
              DataDelegateContact<BaseParticles, BaseParticles>(contact_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_variable_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_variable_name)),
              coefficient_(particles_, coefficient_arg)
        {
            for (size_t k = 0; k != contact_particles_.size(); ++k)
            {
                contact_in_variable_.push_back(contact_particles_[k]->template getVariableByName<InDataType>(in_variable_name));
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
        template <class BodyRelationType, typename SourceArg, typename CoefficientArg, typename... ExtraCoefficientArg>
        OperatorWithBoundary(BodyRelationType &body_relation, BaseContactRelation &relation_to_boundary,
                             const std::string &in_variable_name, const std::string &out_variable_name,
                             const SourceArg &source_arg, const CoefficientArg &coefficient_arg,
                             ExtraCoefficientArg &&...extra_coefficient_args)
            : LocalDynamics(body_relation.sph_body_),
              base_operator_(body_relation, in_variable_name, out_variable_name, source_arg, coefficient_arg,
                             std::forward<ExtraCoefficientArg>(extra_coefficient_args)...),
              operator_near_boundary_(relation_to_boundary, in_variable_name, out_variable_name, coefficient_arg){};
        template <typename SourceArg, typename CoefficientArg, typename... ExtraCoefficientArg>
        OperatorWithBoundary(ComplexRelation &complex_relation,
                             const std::string &in_variable_name, const std::string &out_variable_name,
                             const SourceArg &source_arg, const CoefficientArg &coefficient_arg,
                             ExtraCoefficientArg &&...extra_coefficient_args)
            : OperatorWithBoundary(complex_relation.getInnerRelation(), complex_relation.getContactRelation(),
                                   in_variable_name, out_variable_name, source_arg, coefficient_arg,
                                   std::forward<ExtraCoefficientArg>(extra_coefficient_args)...){};
        virtual ~OperatorWithBoundary(){};

        void interaction(size_t index_i, Real dt)
        {
            base_operator_.interaction(index_i, dt);
            operator_near_boundary_.interaction(index_i, dt);
        };

    protected:
        BaseOperatorType base_operator_;
        OperatorFromBoundaryType operator_near_boundary_;
    };
}
#endif // GENERAL_OPERATORS_H