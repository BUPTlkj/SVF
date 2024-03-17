#ifndef SVF_INTERVALSOLVER_H
#define SVF_INTERVALSOLVER_H
#include "../svf/include/AE/Core/IntervalValue.h"
#include "Eigen/Dense"
#include "../svf/include/Graphs//NNNode.h"

namespace SVF
{

class IntervalSolver
{
public:

    /// Input pixel matrix
    std::vector<Eigen::MatrixXd> data_matrix;
    /// Converted to Interval pixel matrix
    IntervalMatrices interval_data_matrix;
    /// solver
    IntervalMatrices in_x;

    /// Constructor, only accepts pixel matrix, converts to IntervalMatrix
    IntervalSolver(const std::vector<Eigen::MatrixXd>& in): data_matrix(in){
              interval_data_matrix = convertMatricesToIntervalMatrices(data_matrix);
          };
    IntervalSolver(){};

    inline IntervalMatrices get_in_x(){
              return in_x;
    }

    /// TEST
    void initializeMatrix();

    /// std::vector<Eigen::MatrixXd> -> IntervalMatrix
    IntervalMatrices convertMatricesToIntervalMatrices(const std::vector<Eigen::MatrixXd>& matrices);
    IntervalMat convertMatToIntervalMat(const Eigen::MatrixXd& matrix);
    IntervalMat convertVectorXdToIntervalVector(const Eigen::VectorXd& vec);

    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> splitIntervalMatrices(const std::vector<Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic>>& intervalMatrices);

    inline void setIRMatrix(IntervalMatrices x)
    {
        in_x = x;
    }

    IntervalMatrices ReLuNeuronNodeevaluate() const;

    IntervalMatrices BasicOPNeuronNodeevaluate(
        const BasicOPNeuronNode* basic);

    IntervalMatrices MaxPoolNeuronNodeevaluate(
        const MaxPoolNeuronNode* maxpool);

    IntervalMatrices FullyConNeuronNodeevaluate(
        const FullyConNeuronNode* fully);

    IntervalMatrices ConvNeuronNodeevaluate(
        const ConvNeuronNode* conv);

    IntervalMatrices ConstantNeuronNodeevaluate();

};

}/// SVF namespace

#endif // SVF_INTERVALSOLVER_H
