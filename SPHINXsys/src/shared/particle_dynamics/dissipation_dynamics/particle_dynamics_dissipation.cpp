/**
 * @file 	particle_dynamics_dissipation.cpp
 * @author	Xiangyu Hu
 */

#include "particle_dynamics_dissipation.h"

namespace SPH
{
    //=================================================================================================//
    CoefficientSplittingInner::
        CoefficientSplittingInner(BaseInnerRelation &inner_relation,
                                  const std::string &variable_name,
                                  const std::string &coefficient_name,
                                  Real source)
        : LocalDynamics(inner_relation.sph_body_),
          DissipationDataInner(inner_relation),
          Vol_(particles_->Vol_), mass_(particles_->mass_), source_(source),
          variable_(*particles_->getVariableByName<Real>(variable_name)),
          eta_(*particles_->getVariableByName<Real>(coefficient_name)) {}
    //=================================================================================================//
    std::pair<ErrorAndParameters<Real>, ErrorAndParameters<Real>>
    CoefficientSplittingInner::computeErrorAndParameters(size_t index_i, Real dt)
    {
        Real Vol_i = Vol_[index_i];
        Real mass_i = mass_[index_i];
        Real variable_i = variable_[index_i];
        Real eta_i = eta_[index_i];
        ErrorAndParameters<Real> error_and_parameters_plus;
        ErrorAndParameters<Real> error_and_parameters_minus;

        error_and_parameters_plus.error_ = -source_ * dt;
        error_and_parameters_minus.error_ = source_ * dt;
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];
            Real parameter_b = 2.0 * inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt / inner_neighborhood.r_ij_[n];
            Real variable_diff = (variable_i - variable_[index_j]);
            Real variable_diff_abs = ABS(variable_diff);
            Real coefficient_ave = 0.5 * (eta_i + eta_[index_j]);
            Real coefficient_diff = 0.5 * (eta_i - eta_[index_j]);

            error_and_parameters_plus.error_ -= parameter_b * (coefficient_diff * variable_diff_abs + coefficient_ave * variable_diff);
            error_and_parameters_plus.a_ += 0.5 * parameter_b * (variable_diff_abs + variable_diff);
            error_and_parameters_plus.c_ += 0.25 * parameter_b * parameter_b * (variable_diff_abs - variable_diff) * (variable_diff_abs - variable_diff);

            error_and_parameters_minus.error_ -= parameter_b * (coefficient_diff * variable_diff_abs - coefficient_ave * variable_diff);
            error_and_parameters_minus.a_ += 0.5 * parameter_b * (variable_diff_abs - variable_diff);
            error_and_parameters_minus.c_ += 0.25 * parameter_b * parameter_b * (variable_diff_abs + variable_diff) * (variable_diff_abs + variable_diff);
        }
        error_and_parameters_plus.a_ -= mass_i;
        error_and_parameters_minus.a_ -= mass_i;
        return std::pair(error_and_parameters_plus, error_and_parameters_minus);
    }
    //=================================================================================================//
    void CoefficientSplittingInner::
        updateStates(size_t index_i, Real dt,
                     const std::pair<ErrorAndParameters<Real>, ErrorAndParameters<Real>> &error_and_parameters)
    {
        Real error_plus_norm = ABS(error_and_parameters.first.error_);
        Real parameter_l_plus = error_and_parameters.first.a_ * error_and_parameters.first.a_ + error_and_parameters.first.c_;
        Real parameter_k_plus = error_and_parameters.first.error_ / (parameter_l_plus + TinyReal);
        Real increment_i_plus = parameter_k_plus * error_and_parameters.first.a_;
        Real increment_i_plus_norm = ABS(increment_i_plus);

        Real error_minus_norm = ABS(error_and_parameters.second.error_);
        Real parameter_l_minus = error_and_parameters.second.a_ * error_and_parameters.second.a_ + error_and_parameters.second.c_;
        Real parameter_k_minus = error_and_parameters.second.error_ / (parameter_l_minus + TinyReal);
        Real increment_i_minus = parameter_k_minus * error_and_parameters.second.a_;
        Real increment_i_minus_norm = ABS(increment_i_minus);

        if (increment_i_plus_norm < error_plus_norm || increment_i_minus_norm < error_minus_norm)
        {
            Real &eta_i = eta_[index_i];
            Real final_parameter_k = parameter_k_plus;
            Real final_increment_i = increment_i_plus;
            Real sign = 1.0;
            if (increment_i_plus_norm > increment_i_minus_norm)
            {
                final_parameter_k = parameter_k_minus;
                final_increment_i = increment_i_minus;
                sign = -1.0;
            }
            eta_i += final_increment_i;

            Real Vol_i = Vol_[index_i];
            Real variable_i = variable_[index_i];
            Neighborhood &inner_neighborhood = inner_configuration_[index_i];
            for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
            {
                size_t index_j = inner_neighborhood.j_[n];

                Real parameter_b = 2.0 * inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt / inner_neighborhood.r_ij_[n];

                // predicted quantity at particle j
                Real variable_diff = sign * (variable_i - variable_[index_j]);
                Real variable_diff_abs = ABS(variable_diff);
                Real eta_j = eta_[index_j] + final_parameter_k * 0.5 * parameter_b * (variable_diff - variable_diff_abs);

                // exchange in conservation form
                Real coefficient_ave = 0.5 * (eta_i + eta_j);
                Real coefficient_diff = 0.5 * (eta_i - eta_j);
                eta_[index_j] -= parameter_b * (coefficient_ave * variable_diff + coefficient_diff * variable_diff_abs) / mass_[index_j];
            }
        }
    }
    //=================================================================================================//
    void CoefficientSplittingInner::interaction(size_t index_i, Real dt)
    {
        std::pair<ErrorAndParameters<Real>, ErrorAndParameters<Real>>
            error_and_parameters = computeErrorAndParameters(index_i, dt);
        updateStates(index_i, dt, error_and_parameters);
    }
    //=================================================================================================//
    CoefficientSplittingWithWall::
        CoefficientSplittingWithWall(ComplexRelation &complex_wall_relation,
                                     const std::string &variable_name,
                                     const std::string &coefficient_name,
                                     Real source)
        : CoefficientSplittingInner(complex_wall_relation.getInnerRelation(), variable_name, coefficient_name, source),
          DissipationDataWithWall(complex_wall_relation.getContactRelation())
    {
        for (size_t k = 0; k != contact_particles_.size(); ++k)
        {
            wall_variable_.push_back(contact_particles_[k]->template getVariableByName<Real>(variable_name));
        }
    }
    //=================================================================================================//
    std::pair<ErrorAndParameters<Real>, ErrorAndParameters<Real>>
    CoefficientSplittingWithWall::
        computeErrorAndParameters(size_t index_i, Real dt)
    {
        std::pair<ErrorAndParameters<Real>, ErrorAndParameters<Real>> error_and_parameters =
            CoefficientSplittingInner::computeErrorAndParameters(index_i, dt);

        Real Vol_i = Vol_[index_i];
        Real variable_i = variable_[index_i];
        Real eta_i = eta_[index_i];

        ErrorAndParameters<Real> error_and_parameters_plus = error_and_parameters.first;
        ErrorAndParameters<Real> error_and_parameters_minus = error_and_parameters.second;
        for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
        {
            StdLargeVec<Real> &variable_k = *(this->wall_variable_[k]);
            Neighborhood &contact_neighborhood = (*DissipationDataWithWall::contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                size_t index_j = contact_neighborhood.j_[n];
                Real parameter_b = 2.0 * contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt / contact_neighborhood.r_ij_[n];
                Real variable_diff = (variable_i - variable_k[index_j]);

                error_and_parameters_plus.error_ -= parameter_b * eta_i * variable_diff;
                error_and_parameters_plus.a_ += 0.5 * parameter_b * variable_diff;

                error_and_parameters_minus.error_ += parameter_b * eta_i * variable_diff;
                error_and_parameters_minus.a_ -= 0.5 * parameter_b * variable_diff;
            }
        }
        return std::pair(error_and_parameters_plus, error_and_parameters_minus);
    }
    //=================================================================================================//
    CoefficientEvolution::
        CoefficientEvolution(BaseInnerRelation &inner_relation, const std::string &variable_name,
                             const std::string &coefficient_name, Real threshold)
        : LocalDynamics(inner_relation.sph_body_), DissipationDataInner(inner_relation),
          Vol_(particles_->Vol_), mass_(particles_->mass_),
          variable_(*particles_->getVariableByName<Real>(variable_name)),
          eta_(*this->particles_->template getVariableByName<Real>(coefficient_name)),
          threshold_(threshold) {}
    //=================================================================================================//
    void CoefficientEvolution::interaction(size_t index_i, Real dt)
    {
        Real Vol_i = Vol_[index_i];
        Real mass_i = mass_[index_i];
        const Real &variable_i = variable_[index_i];
        Real &eta_i = eta_[index_i];

        if (eta_i > threshold_)
        {
            Real dt2 = dt * 0.5;
            const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
            // forward sweep
            for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
            {
                Real parameter_b = 2.0 * inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / inner_neighborhood.r_ij_[n];
                size_t index_j = inner_neighborhood.j_[n];
                Real mass_j = mass_[index_j];

                Real variable_diff = (variable_i - variable_[index_j]);
                Real variable_diff_abs = ABS(variable_diff);
                Real coefficient_ave = 0.5 * (eta_i + eta_[index_j]);
                Real coefficient_diff = 0.5 * (eta_i - eta_[index_j]);
                Real increment = parameter_b * (coefficient_ave * variable_diff + coefficient_diff * variable_diff_abs) /
                                 (mass_i * mass_j - parameter_b * ((mass_j - mass_i) * coefficient_diff + (mass_j + mass_i) * variable_diff_abs));

                if (eta_[index_j] > threshold_)
                {
                    Real theta_i = SMIN((threshold_ - eta_i) / increment / mass_j, 1.0);
                    Real theta_j = SMIN((eta_[index_j] - threshold_) / increment / mass_i, 1.0);
                    increment *= SMIN(theta_i, theta_j);
                    eta_i += increment * mass_j;
                    eta_[index_j] -= increment * mass_i;
                }
            }

            // backward sweep
            for (size_t n = inner_neighborhood.current_size_; n != 0; --n)
            {
                Real parameter_b = 2.0 * inner_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / inner_neighborhood.r_ij_[n - 1];
                size_t index_j = inner_neighborhood.j_[n - 1];
                Real mass_j = mass_[index_j];

                Real variable_diff = (variable_i - variable_[index_j]);
                Real variable_diff_abs = ABS(variable_diff);
                Real coefficient_ave = 0.5 * (eta_i + eta_[index_j]);
                Real coefficient_diff = 0.5 * (eta_i - eta_[index_j]);
                Real increment = parameter_b * (coefficient_ave * variable_diff + coefficient_diff * variable_diff_abs) /
                                 (mass_i * mass_j - parameter_b * ((mass_j - mass_i) * coefficient_diff + (mass_j + mass_i) * variable_diff_abs));

                if (eta_[index_j] > threshold_)
                {
                    Real theta_i = SMIN((threshold_ - eta_i) / increment / mass_j, 1.0);
                    Real theta_j = SMIN((eta_[index_j] - threshold_) / increment / mass_i, 1.0);
                    increment *= SMIN(theta_i, theta_j);
                    eta_i += increment * mass_j;
                    eta_[index_j] -= increment * mass_i;
                }
            }
        }
    }
    //=================================================================================================//
    CoefficientEvolutionExplicit::
        CoefficientEvolutionExplicit(BaseInnerRelation &inner_relation,
                                     const std::string &variable_name,
                                     const std::string &coefficient_name, Real source)
        : LocalDynamics(inner_relation.sph_body_), DissipationDataInner(inner_relation),
          rho_(particles_->rho_),
          variable_(*particles_->getVariableByName<Real>(variable_name)),
          eta_(*particles_->template getVariableByName<Real>(coefficient_name)),
          source_(source)
    {
        particles_->registerVariable(change_rate_, "DiffusionCoefficientChangeRate");
    }
    //=================================================================================================//
    void CoefficientEvolutionExplicit::interaction(size_t index_i, Real dt)
    {
        Real variable_i = variable_[index_i];
        Real eta_i = eta_[index_i];

        Real change_rate = source_;
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            Real b_ij = 2.0 * inner_neighborhood.dW_ijV_j_[n] / inner_neighborhood.r_ij_[n];
            size_t index_j = inner_neighborhood.j_[n];

            Real variable_diff = (variable_i - variable_[index_j]);
            Real variable_diff_abs = ABS(variable_diff);
            Real coefficient_ave = 0.5 * (eta_i + eta_[index_j]);
            Real coefficient_diff = 0.5 * (eta_i - eta_[index_j]);

            change_rate += b_ij * (coefficient_ave * variable_diff + coefficient_diff * variable_diff_abs);
        }
        change_rate_[index_i] = change_rate / rho_[index_i];
    }
    //=================================================================================================//
    void CoefficientEvolutionExplicit::update(size_t index_i, Real dt)
    {
        eta_[index_i] += change_rate_[index_i] * dt;
    }
    //=================================================================================================//
    CoefficientEvolutionWithWallExplicit::
        CoefficientEvolutionWithWallExplicit(ComplexRelation &complex_relation,
                                             const std::string &variable_name,
                                             const std::string &coefficient_name, Real source)
        : CoefficientEvolutionExplicit(complex_relation.getInnerRelation(), variable_name,
                                       coefficient_name, source),
          DissipationDataWithWall(complex_relation.getContactRelation())
    {
        for (size_t k = 0; k != contact_particles_.size(); ++k)
        {
            wall_variable_.push_back(contact_particles_[k]->template getVariableByName<Real>(variable_name));
        }
    }
    //=================================================================================================//
    void CoefficientEvolutionWithWallExplicit::interaction(size_t index_i, Real dt)
    {
        CoefficientEvolutionExplicit::interaction(index_i, dt);

        Real variable_i = variable_[index_i];
        Real eta_i = eta_[index_i];

        Real change_rate = 0.0;
        for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            const StdLargeVec<Real> &variable_k = *(wall_variable_[k]);
            const Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                Real b_ij = 2.0 * contact_neighborhood.dW_ijV_j_[n] / contact_neighborhood.r_ij_[n];
                size_t index_j = contact_neighborhood.j_[n];

                Real variable_diff = (variable_i - variable_k[index_j]);
                change_rate += b_ij * eta_i * variable_diff;
            }
        }
        change_rate_[index_i] += change_rate / rho_[index_i];
    }
    //=================================================================================================//
    CoefficientEvolutionFromWall::
        CoefficientEvolutionFromWall(BaseContactRelation &contact_relation,
                                     const std::string &variable_name,
                                     const std::string &coefficient_name, Real threshold)
        : LocalDynamics(contact_relation.sph_body_),
          DataDelegateContact<BaseParticles, SolidParticles>(contact_relation),
          Vol_(particles_->Vol_), mass_(particles_->mass_),
          variable_(*particles_->getVariableByName<Real>(variable_name)),
          eta_(*this->particles_->template getVariableByName<Real>(coefficient_name)),
          threshold_(threshold)
    {
        for (size_t k = 0; k != contact_particles_.size(); ++k)
        {
            wall_variable_.push_back(contact_particles_[k]->template getVariableByName<Real>(variable_name));
        }
    }
    //=================================================================================================//
    void CoefficientEvolutionFromWall::interaction(size_t index_i, Real dt)
    {
        Real Vol_i = Vol_[index_i];
        Real mass_i = mass_[index_i];
        const Real &variable_i = variable_[index_i];
        Real &eta_i = eta_[index_i];

        if (eta_i > threshold_)
        {
            // interaction with wall
            Real dt2 = dt * 0.5;
            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                const StdLargeVec<Real> &variable_k = *(wall_variable_[k]);
                const Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
                // forward sweep
                for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
                {
                    Real parameter_b = 2.0 * contact_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / contact_neighborhood.r_ij_[n];
                    size_t index_j = contact_neighborhood.j_[n];

                    Real variable_diff = (variable_i - variable_k[index_j]);
                    Real d_eta = parameter_b * eta_i * variable_diff / (mass_i - parameter_b * variable_diff);
                    Real theta_i = SMIN((threshold_ - eta_i) / d_eta, 1.0);
                    eta_i += theta_i * d_eta;
                }

                // backward sweep
                for (size_t n = contact_neighborhood.current_size_; n != 0; --n)
                {
                    Real parameter_b = 2.0 * contact_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / contact_neighborhood.r_ij_[n - 1];
                    size_t index_j = contact_neighborhood.j_[n - 1];

                    Real variable_diff = (variable_i - variable_k[index_j]);
                    Real d_eta = parameter_b * eta_i * variable_diff / (mass_i - parameter_b * variable_diff);
                    Real theta_i = SMIN((threshold_ - eta_i) / d_eta, 1.0);
                    eta_i += theta_i * d_eta;
                }
            }
        }
    }
    //=================================================================================================//
    DampingCoefficient::
        DampingCoefficient(BaseInnerRelation &inner_relation, const std::string &variable_name,
                           const std::string &coefficient_name, Real strength)
        : LocalDynamics(inner_relation.sph_body_), DissipationDataInner(inner_relation),
          Vol_(particles_->Vol_), mass_(particles_->mass_),
          variable_(*particles_->getVariableByName<Real>(variable_name)),
          eta_(*this->particles_->template getVariableByName<Real>(coefficient_name)),
          strength_(strength) {}
    //=================================================================================================//
    void DampingCoefficient::interaction(size_t index_i, Real dt)
    {
        Real Vol_i = Vol_[index_i];
        Real mass_i = mass_[index_i];
        const Real &variable_i = variable_[index_i];
        Real &eta_i = eta_[index_i];

        Real dt2 = dt * 0.5;
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        // forward sweep
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            Real parameter_b = 2.0 * inner_neighborhood.dW_ijV_j_[n] * Vol_i * dt2 / inner_neighborhood.r_ij_[n];
            size_t index_j = inner_neighborhood.j_[n];
            Real mass_j = mass_[index_j];

            Real variable_diff_abs = strength_ * ABS(variable_i - variable_[index_j]);
            Real coefficient_diff = 0.5 * (eta_i - eta_[index_j]);
            Real increment = parameter_b * coefficient_diff * variable_diff_abs /
                             (mass_i * mass_j - parameter_b * (mass_j + mass_i) * variable_diff_abs);

            eta_i += increment * mass_j;
            eta_[index_j] -= increment * mass_i;
        }

        // backward sweep
        for (size_t n = inner_neighborhood.current_size_; n != 0; --n)
        {
            Real parameter_b = 2.0 * inner_neighborhood.dW_ijV_j_[n - 1] * Vol_i * dt2 / inner_neighborhood.r_ij_[n - 1];
            size_t index_j = inner_neighborhood.j_[n - 1];
            Real mass_j = mass_[index_j];

            Real variable_diff_abs = strength_ * ABS(variable_i - variable_[index_j]);
            Real coefficient_ave = 0.5 * (eta_i + eta_[index_j]);
            Real coefficient_diff = 0.5 * (eta_i - eta_[index_j]);
            Real increment = parameter_b * coefficient_diff * variable_diff_abs /
                             (mass_i * mass_j - parameter_b * (mass_j + mass_i) * variable_diff_abs);

            eta_i += increment * mass_j;
            eta_[index_j] -= increment * mass_i;
        }
    }
    //=================================================================================================//
}
