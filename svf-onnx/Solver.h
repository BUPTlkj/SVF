

#ifndef SVF_SOLVER_H
#define SVF_SOLVER_H

#include "Eigen/Dense"
#include "Graphs/NNGraph.h"

class SolverEvaluate{
public:
    //    std::vector<SVF::NeuronNode::NodeK> typelist;
    std::vector<Eigen::MatrixXd> data_matrix;

    std::vector<Eigen::MatrixXd> in_x;

    SolverEvaluate(const std::vector<Eigen::MatrixXd>& in): data_matrix(in){
    }

    inline void setIRMatrix(std::vector<Eigen::MatrixXd> x){
        in_x = x;
    }


    std::vector<Eigen::MatrixXd> ReLuNeuronNodeevaluate() const;

    std::vector<Eigen::MatrixXd> BasicOPNeuronNodeevaluate(const SVF::BasicOPNeuronNode *basic) const;

    std::vector<Eigen::MatrixXd> MaxPoolNeuronNodeevaluate(const SVF::MaxPoolNeuronNode *maxpool) const;

    std::vector<Eigen::MatrixXd> FullyConNeuronNodeevaluate(const SVF::FullyConNeuronNode *fully) const;

    std::vector<Eigen::MatrixXd> ConvNeuronNodeevaluate(const SVF::ConvNeuronNode *conv) const;

    std::vector<Eigen::MatrixXd> ConstantNeuronNodeevaluate() const;
//    std::vector<Eigen::MatrixXd> ReLuNeuronNodeevaluate(const std::vector<Eigen::MatrixXd>& x_in) const;
//
//    std::vector<Eigen::MatrixXd> BasicOPNeuronNodeevaluate(const std::vector<Eigen::MatrixXd>& in_x, const SVF::BasicOPNeuronNode *basic) const;
//
//    std::vector<Eigen::MatrixXd> MaxPoolNeuronNodeevaluate(const std::vector<Eigen::MatrixXd>& in_x, const SVF::MaxPoolNeuronNode *maxpool) const;
//
//    std::vector<Eigen::MatrixXd> FullyConNeuronNodeevaluate(const std::vector<Eigen::MatrixXd>& in_x, const SVF::FullyConNeuronNode *fully) const;
//
//    std::vector<Eigen::MatrixXd> ConvNeuronNodeevaluate(const std::vector<Eigen::MatrixXd>& x, const SVF::ConvNeuronNode *conv) const;

//    std::vector<Eigen::MatrixXd> ConstantNeuronNodeevaluate(const std::vector<Eigen::MatrixXd>& x) const;

};

#endif // SVF_SOLVER_H
