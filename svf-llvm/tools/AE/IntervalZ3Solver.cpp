#include "IntervalZ3Solver.h"

using namespace SVF;

/// Relations for WX+B
/// Relations for Con: Filter * Sub_X + 0
std::vector<std::vector<z3::expr_vector>> IntervalZ3Solver::FullyConZ3RelationSolver(const Mat& W, const IntervalMatrices& X, const Mat& B) {
    // Sizes of W and X are appropriately matched
    u32_t a = W.size(); // Size of the first dimension
    u32_t b = W.rows(); // Size of the second dimension of W (X's second dimension is 'c')
    u32_t e = X[0].cols(); // Size of the third dimension of X

    std::vector<std::vector<z3::expr_vector>> matrixOfExprVectors;
    std::vector<z3::expr_vector> row;

    // Define the floating-point type (using double precision)
    auto f64 = ctx.fpa_sort(11, 53); // 11-bit exponent, 53-bit mantissa (corresponding to IEEE754 double precision)

    IntervalMatrices Y(a, IntervalMat(b, e)); // Create the result matrix Y

    // Create expression vectors to store equations
    z3::expr_vector equations(ctx);
    z3::expr x = ctx.real_val("0");
    z3::expr weight = ctx.real_val("0");
    z3::expr biase = ctx.real_val("0");
    std::string id;

    // Iterate over each "slice"
    for (u32_t i = 0; i < a; ++i) {
        // Rows
        for (u32_t j = 0; j < b; ++j) {
            // Columns
            for (u32_t k = 0; k < e; ++k) {
                // Compute dot product of W[i] row and X[i] column for each element of Yi
                u32_t g = 0;
                for (u32_t l = 0; l < W.cols(); ++l) {
                    // Encode input_x
                    // Assign a new z3::expr for each matrix element based on a unique ID
                    id = "m" + std::to_string(i) + "_r" + std::to_string(g) + "_c" + std::to_string(k);
                    g++;
                    x = ctx.constant(ctx.str_symbol(id.c_str()), f64);

                    weight = z3::to_expr(ctx, Z3_mk_fpa_numeral_double(ctx, W(j, l), f64));

                    equations.push_back(weight * x);
                }
                biase = z3::to_expr(ctx, Z3_mk_fpa_numeral_double(ctx, B(j, k), f64));
                equations.push_back(biase);

                // Create a copy of equations for the row
                z3::expr_vector copy_for_row(ctx);
                for (unsigned int idx = 0; idx < equations.size(); ++idx) {
                    copy_for_row.push_back(equations[idx]);
                }

                row.push_back(copy_for_row);
                equations.resize(0);
            }

            matrixOfExprVectors.push_back(row);
            row.clear();
        }
    }

    return matrixOfExprVectors;
}


