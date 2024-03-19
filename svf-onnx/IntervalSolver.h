#ifndef SVF_INTERVALSOLVER_H
#define SVF_INTERVALSOLVER_H
#include "../svf/include/AE/Core/IntervalValue.h"
#include "Eigen/Dense"
#include "../svf/include/Graphs/NNNode.h"

namespace SVF
{

class IntervalSolver
{
public:

    /// Interval pixel matrix
    IntervalMatrices interval_data_matrix;
    /// solver
    IntervalMatrices in_x;

    /// Constructor, only accepts pixel matrix, converts to IntervalMatrix
    IntervalSolver(const IntervalMatrices in): interval_data_matrix(in){
          };
    IntervalSolver(){};

    inline IntervalMatrices get_in_x(){
              return in_x;
    }

    /// TEST
    void initializeMatrix();

    std::pair<Matrices, Matrices> splitIntervalMatrices(const IntervalMatrices & intervalMatrices);

    inline void setIRMatrix(IntervalMatrices x){
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

    IntervalMatrices ConstantNeuronNodeevaluate() const;

};

}/// SVF namespace

#endif // SVF_INTERVALSOLVER_H
