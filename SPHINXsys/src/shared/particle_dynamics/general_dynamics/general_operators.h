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
 * HU1527/12-1 and HU1527/12-4	.                                                 *
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
 * @author	Chi Zhang and Xiangyu Hu
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
     * @class BaseOperatorInner
     * @brief Base class for spatial operators with contact relation
     */
    template <typename InDataType, typename OutDataType>
    class BaseOperatorContact : public LocalDynamics, public GeneralDataDelegateInner
    {
    public:
        BaseOperatorContact(BaseContactRelation &contact_relation,
                            const std::string &in_variable_name, const std::string &out_variable_name)
            : LocalDynamics(contact_relation.sph_body_), GeneralDataDelegateContact(contact_relation),
              in_variable_(*particles_->template getVariableByName<InDataType>(in_variable_name)),
              out_variable_(*particles_->template getVariableByName<OutDataType>(out_variable_name))
        {
            for (size_t k = 0; k != contact_particles_.size(); ++k)
            {
                contact_variable_.push_back(contact_particles_[k]->template getVariableByName<DataType>(in_variable_name));
            }
        }

    protected:
        StdLargeVec<InDataType> &in_variable_;
        StdLargeVec<OutDataType> &out_variable_;
        StdVec<StdLargeVec<InDataType> *> contact_in_variable_;
    };

    /**
     * @class BaseOperatorConstantCoefficient
     * @brief Base class for computing Laplacian operators with constant coefficient
     * This can be used for computing dissipative terms
     */
    template <typename CoefficientType, class BaseOperatorType>
    class BaseOperatorConstantCoefficient : public BaseOperatorType
    {
    public:
        template <typename... Args>
        BaseOperatorConstantCoefficient(const CoefficientType &coefficient, Args &&...args)
            : BaseOperatorType(std::forward<ConstructorArgs>(args)...),
              coefficient_(coefficient){};
        virtual ~BaseOperatorConstantCoefficient(){};

    protected:
        CoefficientType coefficient_;
    };

    /**
     * @class BaseOperatorVariableCoefficient
     * @brief Base class for computing spatial operators with variable coefficient
     * This can be used for computing dissipative terms
     */
    template <typename CoefficientType, class BaseOperatorType>
    class BaseOperatorVariableCoefficient : public BaseOperatorType
    {
    public:
        template <typename... Args>
        BaseOperatorVariableCoefficient(const std::string &coefficient_name, Args &&...args)
            : BaseOperatorType(std::forward<ConstructorArgs>(args)...),
              coefficient_(*this->particles_->template getVariableByName<CoefficientType>(coefficient_name)){};
        virtual ~BaseOperatorVariableCoefficient(){};

    protected:
        StdLargeVec<CoefficientType> coefficient_;
    };

    /**
     * @class BaseLaplacianInner
     * @brief Base class for computing Laplacian operators with inner relation
     * This can be used for computing dissipative terms
     */
    template <typename DataType>
    class BaseLaplacianInner : public BaseOperatorInner<DataType, DataType>
    {
    public:
        BaseLaplacianInner(BaseInnerRelation &inner_relation,
                           const std::string &in_variable_name, const std::string &out_variable_name)
            : BaseOperatorInner<DataType, DataType>(inner_relation, in_variable_name, out_variable_name){};
        virtual ~BaseLaplacianInner(){};

    protected:
        template <typename CoefficientFunction>
        void inline loopNeighbors(size_t index_i, const CoefficientFunction &coefficient)
        {
            DataType sum = ZeroData<DataType>::value;
            const Neighborhood &neighborhood = this->inner_configuration_[index_i];
            for (size_t n = 0; n != neighborhood.current_size_; ++n)
            {
                size_t index_j = neighborhood.j_[n];
                sum += 2.0 * coefficient(index_i, index_j) *
                       (this->in_variable_[index_i] - this->in_variable_[index_j]) *
                       neighborhood.dW_ijV_j_[n] / neighborhood.r_ij_[n];
            }
            this->out_variable_[index_i] = sum;
        };
    };

    /**
     * @class BaseLaplacianContact
     * @brief Base class for computing Laplacian operators with contact relation
     * This can be used for computing dissipative terms
     */
    template <typename DataType>
    class BaseLaplacianContact : public BaseOperatorContact<DataType, DataType>
    {
    public:
        BaseLaplacianContact(BaseContactRelation &contact_relation,
                             const std::string &in_variable_name, const std::string &out_variable_name)
            : BaseOperatorContact<DataType, DataType>(contact_relation, in_variable_name, out_variable_name){};
        virtual ~BaseLaplacianContact(){};

    protected:
        template <typename CoefficientFunction>
        void inline loopNeighbors(size_t index_i, const CoefficientFunction &coefficient)
        {
            DataType sum = ZeroData<DataType>::value;
            for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
            {
                const Neighborhood &neighborhood = (*this->contact_configuration_[k])[index_i];
                StdLargeVec<DataType> &in_variable_k = *(this->contact_in_variable_[k]);
                for (size_t n = 0; n != neighborhood.current_size_; ++n)
                {
                    size_t index_j = neighborhood.j_[n];
                    sum += 2.0 * coefficient(index_i, index_j) *
                           (this->in_variable_[index_i] - in_variable_k[index_j]) *
                           neighborhood.dW_ijV_j_[n] / neighborhood.r_ij_[n];
                }
                this->out_variable_[index_i] = sum;
            }
        };
    };

    /**
     * @class LaplacianInner
     * @brief Base class for computing Laplacian operators with inner relation
     * This can be used for computing dissipative terms
     */
    template <typename DataType, typename CoefficientType>
    class LaplacianInner : public BaseLaplacian<InnerDataType>
    {
    public:
        LaplacianInner(BaseInnerRelation &inner_relation, const std::string &in_variable_name,
                       const std::string &out_variable_name, const CoefficientType &coefficient)
            : BaseOperatorInner<DataType, DataType>(inner_relation, in_variable_name, out_variable_name),
              {};
        virtual ~BaseLaplacianInner(){};

    protected:
        CoefficientType coefficient_;
    };

    /**
     * @class SteadySolutionCheckInner
     * @brief check whether a variable has reached a steady state
     */
    template <class DifferentialOperatorType>
    class SteadySolutionCheckInner : public LocalDynamicsReduce<bool, ReduceAND>,
                                     public GeneralDataDelegateInner
    {
    protected:
        using DataType = typename DifferentialOperatorType::VariableType;
        StdLargeVec<DataType> &variable_;
        DifferentialOperatorType operator_;
        DataType steady_reference_;
        const Real criterion_ = 1.0e-6;

        bool checkCriterion(const Real &residue, Real dt)
        {
            return residue * residue * dt * dt / steady_reference_ / steady_reference_ < criterion_;
        };

        template <typename IncrementDatatype>
        bool checkCriterion(const IncrementDatatype &residue, Real dt)
        {
            return residue.squaredNorm() * dt * dt / steady_reference_.squaredNorm() < criterion_;
        };

        template <class OperatorCoefficient>
        DataType Residue(size_t index_i, const OperatorCoefficient &coefficient)
        {
            return operator_(index_i, inner_configuration_[index_i], variable_, coefficient);
        };

    public:
        SteadySolutionCheckInner(BaseInnerRelation &inner_relation, const std::string &variable_name, const DataType &steady_reference)
            : LocalDynamicsReduce<bool, ReduceAND>(sph_body_, true),
              GeneralDataDelegateInner(inner_relation), steady_reference_(steady_reference),
              variable_(*particles_->getVariableByName<DataType>(variable_name)),
              operator_(variable_){};
        virtual ~SteadySolutionCheckInner(){};
    };

    template <class DifferentialOperatorType>
    class SteadySolutionCheckComplex : public SteadySolutionCheckInner<DifferentialOperatorType>,
                                       public GeneralDataDelegateContact
    {
    protected:
        using DataType = typename DifferentialOperatorType::VariableType;
        StdVec<StdLargeVec<DataType> *> contact_variable_;

        template <class OperatorCoefficient>
        DataType Residue(size_t index_i, const OperatorCoefficient &coefficient)
        {
            DataType residue = SteadySolutionCheckInner<DifferentialOperatorType>::Residue(index_i, coefficient);

            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                residue += this->operator_(index_i, (*contact_configuration_[k])[index_i], *(contact_variable_[k]), coefficient);
            }
            return residue;
        };

    public:
        SteadySolutionCheckComplex(ComplexRelation &complex_relation, const std::string &variable_name, const DataType &steady_reference)
            : SteadySolutionCheckInner<DifferentialOperatorType>(complex_relation.getInnerRelation(), variable_name, steady_reference),
              GeneralDataDelegateContact(complex_relation.getContactRelation())
        {
            for (size_t k = 0; k != contact_particles_.size(); ++k)
            {
                contact_variable_.push_back(contact_particles_[k]->template getVariableByName<DataType>(variable_name));
            }
        };
        virtual ~SteadySolutionCheckComplex(){};
    };

    class ConstraintTotalScalarAmount : public LocalDynamics, public GeneralDataDelegateSimple
    {
    public:
        ConstraintTotalScalarAmount(SPHBody &sph_body, const std::string &variable_name);
        virtual ~ConstraintTotalScalarAmount(){};
        void setupInitialScalarAmount();
        void setupDynamics(Real dt = 0.0) override;
        void update(size_t index_i, Real dt = 0.0);

    protected:
        StdLargeVec<Real> &variable_;
        ReduceDynamics<QuantityMoment<Real>> total_scalar_;
        bool is_initialized_;
        Real inital_total_;
        Real increment_;
    };
}
#endif // GENERAL_OPERATORS_H