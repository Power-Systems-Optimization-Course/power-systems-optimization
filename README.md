Power Systems Optimization
-

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

These course materials are jointly developed by [Michael Davidson](http://mdavidson.org/) and [Jesse Jenkins](https://mae.princeton.edu/people/faculty/jenkins) for introducing constrained optimization models applied to power systems. The materials will be used for:
- [MAE / ENE 539](https://registrar.princeton.edu/course-offerings/course-details?term=1212&courseid=008273) Optimization Methods for Energy Systems Engineering (Advanced Topics in Combustion I) [Princeton]
- [MAE 243](https://catalog.ucsd.edu/courses/MAE.html#mae243) Electric Power Systems Modeling [UC San Diego]

### Description

This course will teach students constrained optimization problems and associated solution methods, how to implement and apply linear and mixed integer linear programs to solve such problems using [Julia](https://julialang.org/)/[JuMP](https://jump.dev/JuMP.jl/dev/), and the practical application of such techniques in energy systems engineering.

The course will first introduce students to the theory and mathematics of constrained optimization problems and provide a brief introduction to linear programming, including problem formation and solution algorithms.

Next, to build hands-on experience with optimization methods for energy systems engineering, the course will introduce students to several canonical problems in electric power systems planning and operations, including: economic dispatch, unit commitment, optimal network power flow, and capacity planning.

Finally, several datasets of realistic power systems are provided which students will use in conjunction with building a model for a course project that answers a specific power systems question.

### Notebooks

1. [Constrained Optimization](Notebooks/01-Constrained-Optimization.ipynb)

2. [Using Julia and JuMP for Constrained Optimization](Notebooks/02-Anatomy-of-a-Model.ipynb)

3. [Basic Capacity Expansion Planning](Notebooks/03-Basic-Capacity-Expansion.ipynb)

4. [Economic Dispatch](Notebooks/04-Economic-Dispatch.ipynb)

5. [Unit Commitment](Notebooks/05-Unit-Commitment.ipynb)

6. [DC Optimal Network Power Flow](Notebooks/06-Optimal-Power-Flow.ipynb)

7. [Complex Capacity Expansion Planning](Notebooks/07-Complex-Capacity-Expansion.ipynb)

### Homeworks

1. [Homework 1 - Building Your First Model](Homeworks/Homework-01.ipynb)

2. [Homework 2 - Basic Capacity Expansion](Homeworks/Homework-02.ipynb)

3. [Homework 3 - Unit Commitment](Homeworks/Homework-03.ipynb)

4. [Homework 4 - Optimal Power Flow](Homeworks/Homework-04.ipynb)

5. [Homework 5 - Complex Capacity Expansion](Homeworks/Homework-05.ipynb)

### Project

[Project dataset descriptions](Project/)

1. [ERCOT 120-bus 500kV simulated system for optimal power flow and economic dispatch problems](Project/ercot_500kV/)

2. [ERCOT 3-zone 2040 brownfield expansion system for capacity expansion planning problems](Project/ercot_brownfield_expansion) - (See [Notebook 7](Notebooks/07-Complex-Capacity-Expansion.ipynb) for description)

3. [WECC 6-zone 2045 brownfield expansion system w/100% clean resources for capacity planning problems](Project/wecc_2045_all_clean_expansion)

4. [WECC 12-zone 2020 current capacity for unit commitment and economic dispatch problems](Project/wecc_2020_unit_commitment)

### Tutorials

1. [Julia Tutorial](Tutorials/julia_tutorial.ipynb)

2. [JuMP: Diagnosing infeasible models](Tutorials/jump_infeasibilities.ipynb)

3. [Debugging a Julia script with VS Code](Tutorials/Debugging%20with%20VS%20Code.md)


### License and copyright

If you would like to use these materials, please see the [license and copyright page](LICENSE.md).


