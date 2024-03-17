

#ifndef SVF_SOLVER_H
#define SVF_SOLVER_H

#include "Eigen/Dense"
#include "Graphs/NNGraph.h"
namespace  SVF
{

class SolverEvaluate
{
public:
    std::vector<Eigen::MatrixXd> data_matrix;

    std::vector<Eigen::MatrixXd> in_x;

    SolverEvaluate(const std::vector<Eigen::MatrixXd>& in) : data_matrix(in) {}

    inline void setIRMatrix(std::vector<Eigen::MatrixXd> x)
    {
        in_x = x;
    }

    std::vector<Eigen::MatrixXd> ReLuNeuronNodeevaluate() const;

    std::vector<Eigen::MatrixXd> BasicOPNeuronNodeevaluate(
        const BasicOPNeuronNode* basic) const;

    std::vector<Eigen::MatrixXd> MaxPoolNeuronNodeevaluate(
        const MaxPoolNeuronNode* maxpool) const;

    std::vector<Eigen::MatrixXd> FullyConNeuronNodeevaluate(
        const FullyConNeuronNode* fully) const;

    std::vector<Eigen::MatrixXd> ConvNeuronNodeevaluate(
        const ConvNeuronNode* conv) const;

    std::vector<Eigen::MatrixXd> ConstantNeuronNodeevaluate() const;
};
}

#endif // SVF_SOLVER_H
