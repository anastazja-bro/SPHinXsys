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
 * @file 	base_variables.h
 * @brief 	Here gives classes for the base variables used in simulation.
 * @details These variables are those discretized in spaces and time.
 * @author	Xiangyu Hu
 */

#ifndef BASE_VARIABLES_H
#define BASE_VARIABLES_H

#include "base_data_package.h"

namespace SPH
{
    /**
     * @class DiscreteVariable
     * @brief template base class for all discrete variables.
     */
    template <typename DataType>
    class DiscreteVariable
    {
    public:
        virtual ~DiscreteVariable(){};
        size_t IndexInContainer() const { return index_in_container_; };
        std::string VariableName() const { return name_; };

    private:
        size_t index_in_container_;
        const std::string name_;
        friend class DiscreteVariableManager;
        /** only allowed to be constructed by DiscreteVariableManager */
        DiscreteVariable(size_t index_in_container, const std::string &name)
            : index_in_container_(index_in_container), name_(name){};
    };

    const bool sharedVariable = true;
    typedef DataContainerAssemble<DiscreteVariable> DiscreteVariableAssemble;
    template <typename DataType>
    using VariableContainer = StdVec<DiscreteVariable<DataType>>;

    /**
     * @class DiscreteVariableManager
     * @brief an assemble discrete variables and its management.
     */
    class DiscreteVariableManager
    {
    public:
        DiscreteVariableManager(){};
        virtual ~DiscreteVariableManager(){};
        DiscreteVariableAssemble &VariableAssemble() { return variable_assemble_; };
        /** here, we need return a copy because the elements in std::vector can be relocated */
        template <typename DataType>
        DiscreteVariable<DataType> registerDiscreteVariable(
            const std::string &name, bool is_shared = !sharedVariable)
        {
            constexpr int type_index = DataTypeIndex<DataType>::value;
            VariableContainer<DataType> &variable_container = std::get<type_index>(variable_assemble_);
            size_t assigned_index = initializeIndex(variable_container, name, is_shared);
            return variable_container[assigned_index];
        };
        /** should be used for accessing data assemble immediately only */
        template <typename DataType, template <typename DataTypeInContainer> typename DataContainerType>
        DiscreteVariable<DataType> &DiscreteVariableByName(const std::string &name)
        {
            constexpr int type_index = DataTypeIndex<DataType>::value;
            VariableContainer<DataType> &variable_container = std::get<type_index>(variable_assemble_);
            size_t determined_index = determineIndex(variable_container, name);
            if (determined_index == variable_container.size())
            {
                std::cout << "\n Error: the variable: " << name << " is not registered yet!" << std::endl;
                std::cout << __FILE__ << ':' << __LINE__ << std::endl;
                exit(1);
            }
            return variable_container[determined_index];
        };

    protected:
        DiscreteVariableAssemble variable_assemble_;

        template <typename DataType>
        size_t initializeIndex(VariableContainer<DataType> &variable_container, const std::string &name, bool is_shared)
        {
            size_t determined_index = determineIndex(variable_container, name);

            if (determined_index == variable_container.size()) // determined a new index
            {
                variable_container.push_back(DiscreteVariable<DataType>(determined_index, name));
            }
            else if (!is_shared)
            {
                std::cout << "\n Error: the variable: " << name << " is already used!" << std::endl;
                std::cout << "\n Please check if " << name << " is a sharable variable." << std::endl;
                std::cout << __FILE__ << ':' << __LINE__ << std::endl;
                exit(1);
            }

            return determined_index;
        };

        template <typename DataType>
        size_t determineIndex(const VariableContainer<DataType> &variable_container, const std::string &name)
        {
            size_t i = 0;
            while (i != variable_container.size())
            {
                if (variable_container[i].VariableName() == name)
                {
                    return i;
                }
                ++i;
            }
            return variable_container.size();
        };
    };
}
#endif // BASE_VARIABLES_H
