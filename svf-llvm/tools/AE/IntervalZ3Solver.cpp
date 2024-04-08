#include "IntervalZ3Solver.h"

using namespace SVF;

typedef Eigen::Matrix<z3::expr, Eigen::Dynamic, Eigen::Dynamic> Z3Matrix;
typedef std::vector<Z3Matrix> Z3MatrixVector;

IntervalMatrices IntervalZ3Solver::Multiply1(const Matrices& W, const IntervalMatrices& X) {
    // W和X的大小已经适当匹配
    u32_t a = W.size(); // 第一维的大小
    u32_t b = W[0].rows(); // W的第二维大小，X的第二维是c，我们不直接需要它
    u32_t e = X[0].cols(); // X的第三维大小

    z3::context ctx;
    // 定义浮点数类型，这里使用双精度浮点数
    auto f64 = ctx.fpa_sort(11, 53); // 11位指数，53位尾数，对应于IEEE754中的双精度

    IntervalMatrices Y(a, IntervalMat( b, e)); // 创建结果矩阵Y

    // 创建表达式向量 存储表达式
    z3::expr_vector equations(ctx);

    // 遍历每个"切片"
    for (u32_t i = 0; i < a; ++i) {
        for (u32_t j = 0; j < b; ++j) {
            for (u32_t k = 0; k < e; ++k) {

                // 对于Y[i](j, k)的每个元素，计算W[i]行和X[i]列的点积
                z3::expr sum_l = ctx.real_val(0);
                z3::expr sum_u = ctx.real_val(0);
                for (u32_t l = 0; l < W[i].cols(); ++l) {
                    // 将double类型的W[i](j, l)转换为z3::expr
//                    z3::expr w_val = ctx.real_val(std::to_string(W[i](j, l)).c_str());
                    // 将input_x编码
                    // todo
                    std::string id = "m" + std::to_string(i) + "_r" + std::to_string(j) + "_c" + std::to_string(k);
                    // 为每个矩阵元素赋值一个新的z3::expr，基于唯一的ID

                    z3::expr x = ctx.constant(ctx.str_symbol(id.c_str()), f64);


//                    z3::expr x = ctx.constant(id, f64);
////                    z3::expr num = ctx.fpa_val(1.5, f64);
//                    z3::expr num = z3::to_expr(ctx, Z3_mk_fpa_numeral_double(ctx, W[i](j, l), f64));
//
//
//                    equations.push_back(num * x);

                    equations.push_back(z3::to_expr(ctx, Z3_mk_fpa_numeral_double(ctx, W[i](j, l), f64)) * ctx.constant("x", f64));
//                    equations.push_back(W[i](j, l) * x);


                }
                // Now, let's sum up all expressions in the expr_vector
                z3::expr sum = equations[0];
                for (u32_t m = 1; m < equations.size(); m++) {
                    sum = sum + equations[m];
                }
//                Y[i](j, k) = IntervalValue{sum_l, sum_u};
                // Printing the sum of expressions
                std::cout << "Sum of expressions: " << sum << std::endl;
            }
        }
    }

    return Y;
}



//Z3MatrixVector encodeMatricesAsZ3(const std::vector<Eigen::MatrixXd>& matrices, z3::context& ctx) {
//    Z3MatrixVector z3Matrices;
//
//    for (u32_t matIndex = 0; matIndex < matrices.size(); ++matIndex) {
//        const auto& matrix = matrices[matIndex];
//        // 先创建一个空的Z3Matrix
//        Z3Matrix z3Matrix;
//        // 使用resize来设置矩阵的大小
//        z3Matrix.resize(matrix.rows(), matrix.cols());
//
//        for (u32_t i = 0; i < matrix.rows(); ++i) {
//            for (u32_t j = 0; j < matrix.cols(); ++j) {
//                std::string id = "m" + std::to_string(matIndex) + "_r" + std::to_string(i) + "_c" + std::to_string(j);
//                // 为每个矩阵元素赋值一个新的z3::expr，基于唯一的ID
//                z3Matrix(i, j) = z3::expr(ctx.real_const(id.c_str()));
//            }
//        }
//
//        z3Matrices.push_back(z3Matrix);
//    }
//
//    return z3Matrices;
//}

// todo Add or set W into 1;


/// Relations for WX+B
/// Relations for Con: Filter * Sub_X + 0
IntervalMatrices IntervalZ3Solver::Multiply(const Mat& W, const IntervalMatrices& X, const Mat& B) {
    // W和X的大小已经适当匹配
    u32_t a = W.size(); // 第一维的大小
    u32_t b = W.rows(); // W的第二维大小，X的第二维是c
    u32_t e = X[0].cols(); // X的第三维大小

    z3::context ctx;
    // 定义浮点数类型，这里使用双精度浮点数
    auto f64 = ctx.fpa_sort(11, 53); // 11位指数，53位尾数，对应于IEEE754中的双精度

    IntervalMatrices Y(a, IntervalMat(b, e)); // 创建结果矩阵Y

    // 创建表达式向量 存储表达式
    z3::expr_vector equations_upper(ctx);
    z3::expr_vector equations_lower(ctx);
    z3::expr_vector equations(ctx);
    z3::expr x_upper = ctx.real_val("0");
    z3::expr x_lower = ctx.real_val("0");
    z3::expr x = ctx.real_val("0");
    z3::expr weight = ctx.real_val("0");
    z3::expr biase = ctx.real_val("0");
    std::string id_upper;
    std::string id_lower;
    std::string id;

    // 遍历每个"切片"
    // 结果矩阵
    // 维度
    for (u32_t i = 0; i < a; ++i) {
        // 行数
        for (u32_t j = 0; j < b; ++j) {
            // 列数
            for (u32_t k = 0; k < e; ++k) {

                // 对于Y[i](j, k)的每个元素，计算W[i]行和X[i]列的点积
                z3::expr sum_l = ctx.real_val(0);
                z3::expr sum_u = ctx.real_val(0);
                u32_t g = 0;
                for (u32_t l = 0; l < W.cols(); ++l) {

                    // 将input_x编码
                    // 为每个矩阵元素赋值一个新的z3::expr，基于唯一的ID
//                    id_upper = "m" + std::to_string(i) + "_r" + std::to_string(g) + "_c" + std::to_string(k);
//                    id_lower = "m" + std::to_string(i) + "_r" + std::to_string(g) + "_c" + std::to_string(k);
                    id = "m" + std::to_string(i) + "_r" + std::to_string(g) + "_c" + std::to_string(k);
                    g++;
//                    x_upper = ctx.constant(ctx.str_symbol(id_upper.c_str()), f64);
//                    x_lower = ctx.constant(ctx.str_symbol(id_lower.c_str()), f64);
                    x = ctx.constant(ctx.str_symbol(id.c_str()), f64);

                    weight = z3::to_expr(ctx, Z3_mk_fpa_numeral_double(ctx, W(j, l), f64));
                    biase = z3::to_expr(ctx, Z3_mk_fpa_numeral_double(ctx, B(j, l), f64));

//                    equations_upper.push_back(weight * x_upper + biase);
//                    equations_lower.push_back(weight * x_upper + biase);
                    equations.push_back(weight * x + biase);

                }
                // Now, let's sum up all expressions in the expr_vector
//                z3::expr sum_upper = equations_upper[0];
//                z3::expr sum_lower = equations_lower[0];
                z3::expr sum = equations[0];
                for (u32_t m = 1; m < equations.size(); m++) {
//                    sum_upper = sum_upper + equations_upper[m];
//                    sum_lower = sum_lower + equations_lower[m];
                    sum = sum + equations[m];
                }
                // Printing the sum of expressions
                std::cout << "Sum of expressions: " << sum << std::endl;
                equations.resize(0);
                equations_upper.resize(0);
                equations_lower.resize(0);
            }
        }
    }

    return Y;
}

//IntervalMatrices IntervalZ3Solver::Multiply(const Matrices& W, const IntervalMatrices& X) {
//    // 假设W和X的大小已经适当匹配
//    u32_t a = W.size(); // 第一维的大小
//    u32_t b = W[0].rows(); // W的第二维大小
//    u32_t c = W[0].cols(); // 假设c是W的第三维大小，也是X的第二维大小
//    u32_t e = X[0].cols(); // X的第三维大小
//
//    z3::context ctx;
//    auto f64 = ctx.fpa_sort(11, 53); // 定义浮点数类型，这里使用双精度浮点数
//
//    IntervalMatrices Y(a, IntervalMat(b, e)); // 创建结果矩阵Y
//
//    // 创建表达式向量 存储表达式
//    z3::expr_vector equations(ctx);
//    std::string id;
//    z3::expr y_ij = ctx.real_val("0");
//
//        // 遍历每个"切片"
//    for (u32_t i = 0; i < a; ++i) {
//        for (u32_t j = 0; j < b; ++j) {
//            for (u32_t k = 0; k < e; ++k) {
//                // 初始化求和表达式
//                z3::expr sum_upper = ctx.real_val(0);
//                z3::expr sum_lower = ctx.real_val(0);
//
//                for (u32_t l = 0; l < c; ++l) {
//                    // 生成变量ID，注意这里我们不再需要创建X矩阵的变量
//                    // 因为X矩阵是已知的，我们直接使用其值
//                    id = "y_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k);
//                    std::cout<<id<<" * "<<W[i](j, l)<<"   ";
//                    y_ij = ctx.constant(ctx.str_symbol(id.c_str()), f64);
//
//                    // 直接从W和X中获取值，这里假设可以直接获取
//                    double w_val = W[i](j, l); // 获取W[i]的第j行第l列的值
////                    double x_val_upper = X[i](l, k).ub().getRealNumeral(); // 获取X[i]的第l行第k列的值
////                    double x_val_lower = X[i](l, k).lb().getRealNumeral(); // 获取X[i]的第l行第k列的值
//
//                    // 将W的值转换为浮点数表达式
//                    z3::expr w_expr = z3::to_expr(ctx, Z3_mk_fpa_numeral_double(ctx, w_val, f64));
//
//                    // 计算乘积并累加
////                    sum_lower = sum_lower + w_expr * x_val_lower; // 注意这里x_val应该是一个具体的值，而不是变量
////                    sum_upper = sum_upper + w_expr * x_val_upper; // 注意这里x_val应该是一个具体的值，而不是变量
//                }
//                std::cout<<std::endl;
//
//                // 最后，将计算得到的sum表达式赋值给对应的Y矩阵的元素
//                // 注意：这里需要你根据实际情况调整，如何将表达式赋值给Y矩阵的元素
//                // Y[i](j, k) = sum; // 这是伪代码，需要根据实际情况实现
////                std::cout<<"AAAAUPPER: "<<sum_upper<<std::endl;
//            }
//        }
//    }
//
//    return Y;
//}
