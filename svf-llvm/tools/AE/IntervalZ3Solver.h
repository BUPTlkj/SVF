#ifndef SVF_INTERVALZ3SOLVER_H
#define SVF_INTERVALZ3SOLVER_H

#include "Graphs/NNNode.h"
namespace SVF{

class IntervalZ3Solver{
public:
    /// Each layer' input matrix
//    IntervalMatrices in_x;
    ///
    /// \param W
    /// \param X
    /// \return

    IntervalMatrices Multiply(const Mat& W, const IntervalMatrices& X, const Mat& Biase);
    IntervalMatrices Multiply1(const Matrices& W, const IntervalMatrices& X);



};








} // End of SVF Namespace

#endif // SVF_INTERVALZ3SOLVER_H
