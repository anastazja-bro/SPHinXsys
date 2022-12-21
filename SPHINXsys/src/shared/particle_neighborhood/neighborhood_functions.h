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
 * @file 	neighborhood_functions.h
 * @brief 	There are the functions with operation in the neighborhood.
 * @author	Xiangyu Hu
 */

#ifndef NEIGHBORHOOD_FUNCTIONS_H
#define NEIGHBORHOOD_FUNCTIONS_H

#include "neighborhood.h"

namespace SPH
{
    /** Laplacian operator */
    template <typename DataType>
    class Laplacian
    {
        StdLargeVec<DataType> &variable_;

    public:
		using VariableType = DataType;
        Laplacian(StdLargeVec<DataType> &variable) : variable_(variable){};
        template <typename CoefficientType>
        inline DataType operator()(size_t index_i, const Neighborhood &neighborhood,
                                   StdLargeVec<DataType> &neighbor_variable,
                                   const CoefficientType &coefficient)
        {
            DataType sum = ZeroData<DataType>::value;
            for (size_t n = 0; n != neighborhood.current_size_; ++n)
            {
                size_t index_j = neighborhood.j_[n];
                sum += 2.0 * coefficient(index_i, index_j) *
                       (variable_[index_i] - neighbor_variable[index_j]) *
                       neighborhood.dW_ijV_j_[n] / neighborhood.r_ij_[n];
            }
            return sum;
        };
    };
}
#endif // NEIGHBORHOOD_FUNCTIONS_H