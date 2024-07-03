#ifndef SVF_NNGRAPHINTERVALSOLVER_H
#define SVF_NNGRAPHINTERVALSOLVER_H
#include "AE/Core/IntervalValue.h"
// #include "Eigen/Dense"

#include "Eigen/Core"
#include "Graphs/NNNode.h"

namespace SVF
{

class NNgraphIntervalSolver
{
public:

    /// Interval pixel matrix
    IntervalMatrices interval_data_matrix;
    /// solver
    IntervalMatrices in_x;

    /// Constructor, only accepts pixel matrix, converts to IntervalMatrix
    NNgraphIntervalSolver(const IntervalMatrices in): interval_data_matrix(in){
          };
    NNgraphIntervalSolver(){};

    inline IntervalMatrices get_in_x(){
              return in_x;
    }

    /// TEST
    void Test_1();

    std::pair<Matrices, Matrices> splitIntervalMatrices(const IntervalMatrices & intervalMatrices);

    inline void setIRMatrix(IntervalMatrices x){
        in_x = x;
    }

    IntervalMatrices ReLuNeuronNodeevaluate() const;

    IntervalMatrices FlattenNeuronNodeevaluate() const;

    IntervalMatrices BasicOPNeuronNodeevaluate(
        const BasicOPNeuronNode* basic);

    IntervalMatrices MaxPoolNeuronNodeevaluate(
        const MaxPoolNeuronNode* maxpool);

    IntervalMatrices FullyConNeuronNodeevaluate(
        const FullyConNeuronNode* fully);

    IntervalMatrices ConvNeuronNodeevaluate(
        const ConvNeuronNode* conv);

    IntervalMatrices ConstantNeuronNodeevaluate() const;

};

}/// SVF namespace

#endif // SVF_NNGRAPHINTERVALSOLVER_H
