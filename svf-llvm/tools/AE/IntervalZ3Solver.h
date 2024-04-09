#ifndef SVF_INTERVALZ3SOLVER_H
#define SVF_INTERVALZ3SOLVER_H

#include "Graphs/NNNode.h"
namespace SVF{

class IntervalZ3Solver{
    z3::context ctx;
public:
    /// Each layer' input matrix
//    IntervalMatrices in_x;
    IntervalZ3Solver() : ctx() {}
    ///
    /// \param W
    /// \param X
    /// \param B
    /// \return

    std::vector<std::vector<z3::expr_vector>> FullyConZ3RelationSolver(const Mat& W, const IntervalMatrices& X, const Mat& B);
    void ConvlutionalZ3ReltionSolver();




};








} // End of SVF Namespace

#endif // SVF_INTERVALZ3SOLVER_H
