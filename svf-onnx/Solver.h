

#ifndef SVF_SOLVER_H
#define SVF_SOLVER_H

#include "Eigen/Dense"
#include "Graphs/NNGraph.h"
namespace  SVF
{

class SolverEvaluate
{
public:
    Matrices data_matrix;

    Matrices in_x;

    SolverEvaluate(const Matrices& in) : data_matrix(in) {}

    inline void setIRMatrix(Matrices x)
    {
        in_x = x;
    }

    Matrices ReLuNeuronNodeevaluate() const;

    Matrices BasicOPNeuronNodeevaluate(
        const BasicOPNeuronNode* basic) const;

    Matrices MaxPoolNeuronNodeevaluate(
        const MaxPoolNeuronNode* maxpool) const;

    Matrices FullyConNeuronNodeevaluate(
        const FullyConNeuronNode* fully) const;

    Matrices ConvNeuronNodeevaluate(
        const ConvNeuronNode* conv) const;

    Matrices ConstantNeuronNodeevaluate() const;
};
}

#endif // SVF_SOLVER_H
