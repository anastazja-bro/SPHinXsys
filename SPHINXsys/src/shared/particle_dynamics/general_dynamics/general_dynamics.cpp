#include "general_dynamics.h"

namespace SPH
{
	//=================================================================================================//
	TimeStepInitialization::TimeStepInitialization(SPHBody &sph_body, SharedPtr<Gravity> gravity_ptr)
		: BaseTimeStepInitialization(sph_body, gravity_ptr), GeneralDataDelegateSimple(sph_body),
		  pos_(particles_->pos_), acc_prior_(particles_->acc_prior_) {}
	//=================================================================================================//
	void TimeStepInitialization::update(size_t index_i, Real dt)
	{
		acc_prior_[index_i] = gravity_->InducedAcceleration(pos_[index_i]);
	}
	//=================================================================================================//
	RandomizeParticlePosition::RandomizeParticlePosition(SPHBody &sph_body)
		: LocalDynamics(sph_body), GeneralDataDelegateSimple(sph_body),
		  pos_(particles_->pos_), randomize_scale_(sph_body.sph_adaptation_->MinimumSpacing()) {}
	//=================================================================================================//
	void RandomizeParticlePosition::update(size_t index_i, Real dt)
	{
		Vecd &pos_n_i = pos_[index_i];
		for (int k = 0; k < pos_n_i.size(); ++k)
		{
			pos_n_i[k] += dt * (((double)rand() / (RAND_MAX)) - 0.5) * 2.0 * randomize_scale_;
		}
	}
	//=================================================================================================//
	VelocityBoundCheck::
		VelocityBoundCheck(SPHBody &sph_body, Real velocity_bound)
		: LocalDynamicsReduce<bool, ReduceOR>(sph_body, false),
		  GeneralDataDelegateSimple(sph_body),
		  vel_(particles_->vel_), velocity_bound_(velocity_bound) {}
	//=================================================================================================//
	bool VelocityBoundCheck::reduce(size_t index_i, Real dt)
	{
		return vel_[index_i].norm() > velocity_bound_;
	}
	//=================================================================================================//
	UpperFrontInXDirection::UpperFrontInXDirection(SPHBody &sph_body)
		: LocalDynamicsReduce<Real, ReduceMax>(sph_body, Real(0)),
		  GeneralDataDelegateSimple(sph_body),
		  pos_(particles_->pos_)
	{
		quantity_name_ = "UpperFrontInXDirection";
	}
	//=================================================================================================//
	Real UpperFrontInXDirection::reduce(size_t index_i, Real dt)
	{
		return pos_[index_i][0];
	}
	//=================================================================================================//
	MaximumSpeed::MaximumSpeed(SPHBody &sph_body)
		: LocalDynamicsReduce<Real, ReduceMax>(sph_body, Real(0)),
		  GeneralDataDelegateSimple(sph_body),
		  vel_(particles_->vel_)
	{
		quantity_name_ = "MaximumSpeed";
	}
	//=================================================================================================//
	Real MaximumSpeed::reduce(size_t index_i, Real dt)
	{
		return vel_[index_i].norm();
	}
	//=================================================================================================//
	PositionLowerBound::PositionLowerBound(SPHBody &sph_body)
		: LocalDynamicsReduce<Vecd, ReduceLowerBound>(sph_body, MaxRealNumber * Vecd::Ones()),
		  GeneralDataDelegateSimple(sph_body),
		  pos_(particles_->pos_)
	{
		quantity_name_ = "PositionLowerBound";
	}
	//=================================================================================================//
	Vecd PositionLowerBound::reduce(size_t index_i, Real dt)
	{
		return pos_[index_i];
	}
	//=================================================================================================//
	PositionUpperBound::PositionUpperBound(SPHBody &sph_body)
		: LocalDynamicsReduce<Vecd, ReduceUpperBound>(sph_body, MinRealNumber * Vecd::Ones()),
		  GeneralDataDelegateSimple(sph_body),
		  pos_(particles_->pos_)
	{
		quantity_name_ = "PositionUpperBound";
	}
	//=================================================================================================//
	Vecd PositionUpperBound::reduce(size_t index_i, Real dt)
	{
		return pos_[index_i];
	}
	//=================================================================================================//
	TotalMechanicalEnergy::TotalMechanicalEnergy(SPHBody &sph_body, SharedPtr<Gravity> gravity_ptr)
		: LocalDynamicsReduce<Real, ReduceSum<Real>>(sph_body, Real(0)),
		  GeneralDataDelegateSimple(sph_body), mass_(particles_->mass_),
		  vel_(particles_->vel_), pos_(particles_->pos_),
		  gravity_(gravity_ptr_keeper_.assignPtr(gravity_ptr))
	{
		quantity_name_ = "TotalMechanicalEnergy";
	}
	//=================================================================================================//
	Real TotalMechanicalEnergy::reduce(size_t index_i, Real dt)
	{
		return 0.5 * mass_[index_i] * vel_[index_i].squaredNorm() + mass_[index_i] * gravity_->getPotential(pos_[index_i]);
	}
	//=================================================================================================//
	GlobalCorrectConfigurationInner::
		GlobalCorrectConfigurationInner(BaseInnerRelation& inner_relation)
		: LocalDynamics(inner_relation.sph_body_), GeneralDataDelegateInner(inner_relation),
		Vol_(particles_->Vol_)
	{
		particles_->registerVariable(A_, "OriginalCorrectionMatrix");
		particles_->registerVariable(L_, "WeightedCorrectionMatrix");
	}
	//=================================================================================================//
	void GlobalCorrectConfigurationInner::interaction(size_t index_i, Real dt)
	{
		Matd local_configuration = Eps * Matd::Identity(); // a small number added to diagonal to avoid divide zero
		const Neighborhood& inner_neighborhood = inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];

			Vecd gradW_ij = inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
			Vecd r_ji = inner_neighborhood.r_ij_[n] * inner_neighborhood.e_ij_[n];
			local_configuration -= r_ji * gradW_ij.transpose();
		}
		A_[index_i] = local_configuration;
	}
	//=================================================================================================//
	void GlobalCorrectionMatrixComplex::interaction(size_t index_i, Real dt)
	{
		GlobalCorrectConfigurationInner::interaction(index_i, dt);

		Matd local_configuration = Eps * Matd::Identity();
		for (size_t k = 0; k < contact_configuration_.size(); ++k)
		{
			StdLargeVec<Real>& Vol_k = *(contact_Vol_[k]);
			StdLargeVec<Real>& contact_mass_k = *(contact_mass_[k]);
			Neighborhood& contact_neighborhood = (*contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];
				Vecd gradW_ij = contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];
				Vecd r_ji = contact_neighborhood.r_ij_[n] * contact_neighborhood.e_ij_[n];
				local_configuration -= r_ji * gradW_ij.transpose();
			}
		}
		A_[index_i] += local_configuration;
		L_[index_i] = (pow(A_[index_i].determinant(), 2) * A_[index_i].inverse() + 0.3 * Matd::Identity()) / (0.3 * 1.0 + pow(A_[index_i].determinant(), 2));
	}
}
//=================================================================================================//