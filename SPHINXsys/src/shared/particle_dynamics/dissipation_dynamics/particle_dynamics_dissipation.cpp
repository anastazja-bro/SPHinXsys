/**
 * @file 	particle_dynamics_dissipation.cpp
 * @author	Xiangyu Hu
 */

#include "particle_dynamics_dissipation.h"

namespace SPH
{
    //=================================================================================================//
    DampingCoefficientEvolution::
        DampingCoefficientEvolution(BaseInnerRelation &inner_relation, const std::string &variable_name,
                                    const std::string &coefficient_name, Real threshold)
        : LocalDynamics(inner_relation.sph_body_), DissipationDataInner(inner_relation),
          Vol_(particles_->Vol_), mass_(particles_->mass_),
          variable_(*particles_->getVariableByName<Real>(variable_name)),
          eta_(*this->particles_->template getVariableByName<Real>(coefficient_name)),
          threshold_(threshold) {}
    //=================================================================================================//
    void DampingCoefficientEvolution::interaction(size_t index_i, Real dt)
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
    DampingCoefficientEvolutionExplicit::
        DampingCoefficientEvolutionExplicit(BaseInnerRelation &inner_relation,
                                            const std::string &variable_name,
                                            const std::string &coefficient_name, Real threshold)
        : LocalDynamics(inner_relation.sph_body_), DissipationDataInner(inner_relation),
          rho_(particles_->rho_),
          variable_(*particles_->getVariableByName<Real>(variable_name)),
          eta_(*particles_->template getVariableByName<Real>(coefficient_name)),
          threshold_(threshold)
    {
        particles_->registerVariable(change_rate_, "DiffusionCoefficientChangeRate");
    }
    //=================================================================================================//
    void DampingCoefficientEvolutionExplicit::interaction(size_t index_i, Real dt)
    {
        Real variable_i = variable_[index_i];
        Real eta_i = eta_[index_i];

        Real change_rate = 0.0;
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
    void DampingCoefficientEvolutionExplicit::update(size_t index_i, Real dt)
    {
        eta_[index_i] += change_rate_[index_i] * dt;
    }
    //=================================================================================================//
    DampingCoefficientEvolutionFromWall::
        DampingCoefficientEvolutionFromWall(BaseContactRelation &contact_relation,
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
    void DampingCoefficientEvolutionFromWall::interaction(size_t index_i, Real dt)
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
    } //=================================================================================================//
}
