#include "IntervalSolver.h"

using namespace SVF;
template<typename T>
bool isDouble(T&& var) {
    return std::is_same<typename std::remove_cv<typename std::remove_reference<T>::type>::type, double>::value;
}

template<typename T>
bool isInt(T&& var) {
    return std::is_same<typename std::remove_cv<typename std::remove_reference<T>::type>::type, int>::value;
}


//typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> myMatrixXd;

void IntervalSolver::initializeMatrix() {

//    typedef Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> myMatrixXd;
//    int a = 2;
//    int b = 3;
//    // int c = 2;
//
//    IntervalValue inv(0,1);
//    IntervalValue inv2 = inv + inv;
//    std::cout << "INV:\n" << inv << std::endl;
//    std::cout << "INV2:\n" << inv2 << std::endl;
//
//    // [a,b] * [c,c] works fine
//    // Eigen::Matrix<IntervalValue, 2, 2> matrix;
//    // Eigen::Matrix<IntervalValue, a, a> matrix;
//    myMatrixXd matrix(a,b);
//    matrix(0,0) = IntervalValue(2,2);
//    matrix(0,1) = IntervalValue(1,1);
//    matrix(0,2) = IntervalValue(0,0);
//    matrix(1,0) = IntervalValue(0,0);
//    matrix(1,1) = IntervalValue(2,2);
//    matrix(1,2) = IntervalValue(0,0);
//
//
//    // Eigen::Matrix<IntervalValue, 2, 2> rhs;
//    myMatrixXd rhs(b,a);
//    rhs(0,0) = inv;
//    rhs(0,1) = inv + inv;
//    rhs(1,0) = inv ;
//    rhs(1,1) = inv;
//    rhs(2,0) = inv ;
//    rhs(2,1) = inv;
//
//    myMatrixXd output = matrix * rhs;
//
//    std::cout << "Matrix 107:\n" << output(0,0) << std::endl;
//    for (int i = 0; i < output.rows(); ++i) {
//        for (int j = 0; j < output.cols(); ++j) {
//            std::cout << output(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }






    /// 需要定义ab
    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> matrixq(2, 2);
    matrixq(0,0) = IntervalValue(2.1, 2.1);
    matrixq(0,1) = IntervalValue(0,0);
    matrixq(1,0) = IntervalValue(0,0);
    matrixq(1,1) = IntervalValue(2.34, 2.45);

    IntervalValue invv(0.34, 1.45);
    // 2.1*0.34+0.34 ,

    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> aw(2, 2);
    aw(0,0) = invv;
    aw(0,1) = invv + invv;
    aw(1,0) = invv ;
    aw(1,1) = invv;

    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> Matrix4 = matrixq * aw;

    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> Matrix1;
    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> Matrix2;
    Eigen::Matrix<IntervalValue, 2, 2> Matrix3;
////    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> Matrix4 = Matrix1 * Matrix2;
//
//    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> Matrix6 = Matrix1 * Matrix3;
//
//    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> Matrix5 = Matrix1 + Matrix2;
//

    Eigen::Matrix<IntervalValue, Eigen::Dynamic,  Eigen::Dynamic>  c = aw * matrixq + aw;
    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            std::cout <<"[" << aw(i,j).lb().getRealNumeral() << ", "  << aw(i,j).ub().getRealNumeral()<< "]\t";
            std::cout << aw(i,j).toString()<< "]\t";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            std::cout <<"[" << c(i,j).lb().getRealNumeral() << ", "  << c(i,j).ub().getRealNumeral()<< "]\t";
//            std::cout << c(i,j).toString() << "\t";
        }
        std::cout << std::endl;
    }
}

IntervalMatrices IntervalSolver::convertMatricesToIntervalMatrices(const std::vector<Eigen::MatrixXd>& matrices) {
    IntervalMatrices intervalMatrices;

    for (const auto& mat : matrices) {
        Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> intervalMatrix(mat.rows(), mat.cols());
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                // 对于每个元素x，创建一个区间[x, x]
//                std::cout<<"Matrix: ("<<i<<", "<<j<<") :  "<<mat(i,j)<<" Is double? "<<isDouble(mat(i,j))<<std::endl;
                intervalMatrix(i, j) = IntervalValue(mat(i, j), mat(i, j));
//                intervalMatrix(i, j) = IntervalValue(static_cast<double>(mat(i, j)), static_cast<double>(mat(i, j)));
            }
        }
        intervalMatrices.push_back(intervalMatrix);
    }
    return intervalMatrices;
}

IntervalMat IntervalSolver::convertMatToIntervalMat(const Eigen::MatrixXd& matrix){

    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> intervalMatrix(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            // 对于每个元素x，创建一个区间[x, x]
//            std::cout<<"Matrix: ("<<i<<", "<<j<<") :  "<<matrix(i,j)<<" Is double? "<<isDouble(matrix(i,j))<<std::endl;
            intervalMatrix(i, j) = IntervalValue(matrix(i, j), matrix(i, j));
        }
    }
    return intervalMatrix;
}

IntervalMat IntervalSolver::convertVectorXdToIntervalVector(const Eigen::VectorXd& vec) {
    // 创建一个具有相同行数的IntervalEigen，列数为1
    IntervalMat intervalMat(vec.size(), 1);

    // 遍历VectorXd中的每个元素
    for (int i = 0; i < vec.size(); ++i) {
        // 将每个double值转换为一个具有相同上下界的IntervalValue
        intervalMat(i, 0) = IntervalValue(vec(i), vec(i)); // 需要IntervalValue有一个接受两个double的构造函数
    }

    return intervalMat;
}


std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> IntervalSolver::splitIntervalMatrices(const std::vector<Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic>>& intervalMatrices) {
    std::vector<Eigen::MatrixXd> lowerBounds, upperBounds;

    for (const auto& intervalMatrix : intervalMatrices) {
        Eigen::MatrixXd lower(intervalMatrix.rows(), intervalMatrix.cols());
        Eigen::MatrixXd upper(intervalMatrix.rows(), intervalMatrix.cols());

        for (int i = 0; i < intervalMatrix.rows(); ++i) {
            for (int j = 0; j < intervalMatrix.cols(); ++j) {
                lower(i, j) = intervalMatrix(i, j).lb().getNumeral();
                upper(i, j) = intervalMatrix(i, j).ub().getNumeral();
            }
        }

        lowerBounds.push_back(lower);
        upperBounds.push_back(upper);
    }

    return {lowerBounds, upperBounds};
}



IntervalMatrices IntervalSolver::ReLuNeuronNodeevaluate() const {
    std::cout << "Reluing....." << std::endl;
    IntervalMatrices x_out;
    for (const auto& mat : in_x) { // 直接使用auto&避免拷贝
        Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> o(mat.rows(), mat.cols());
        for (u32_t i = 0; i < mat.rows(); i++) {
            for (u32_t j = 0; j < mat.cols(); j++) {
                const auto& iv = mat(i, j); // 获取当前区间
                double lb = iv.lb().getNumeral(); // 获取区间下界
                double ub = iv.ub().getNumeral(); // 获取区间上界

                // 应用ReLU函数逻辑
                double newLb = std::max(0.0, lb);
                double newUb = std::max(0.0, ub);

                // 创建新的IntervalValue，这里假设有一个接受两个double参数的构造函数
                o(i, j) = IntervalValue(newLb, newUb);
            }
        }
        x_out.push_back(o);
    }
    return x_out;
}


IntervalMatrices IntervalSolver::BasicOPNeuronNodeevaluate( const BasicOPNeuronNode *basic){
    std::cout<<"BasicNoding...... The input size: "<<in_x.size()<<std::endl;


    IntervalMatrices result;
//    std::vector<Eigen::MatrixXd> result;

    /// ensure A and B the number of depth equal
    auto constant = basic->constant;

    auto Intervalconstant = convertMatricesToIntervalMatrices(constant);

    if (in_x.size() != Intervalconstant.size()) {
        std::cout<<"In_x.size(): "<<in_x.size()<<",  "<<"Intervalconstant.size()): "<<Intervalconstant.size()<<std::endl;
        std::cerr << "Error: The matrix of channels must be the same." << std::endl;
        return result;
    }

    for (size_t i = 0; i < in_x.size(); ++i) {
        /// ensure the size of each channel correct
        if (constant[i].rows() != 1 || constant[i].cols() != 1) {
            std::cerr << "Error: B's channel matrices must be 1x1." << std::endl;
            return result;
        }

        /// Create a matrix of the same size as the channel of A,
        /// where all elements are the corresponding channel values of B
        Eigen::MatrixXd temp = Eigen::MatrixXd::Constant(in_x[i].rows(), in_x[i].cols(), constant[i](0,0));


        auto Intervaltemp = convertMatToIntervalMat(temp);

        /// Subtract the channel of A from the above matrix

        if(basic->get_type() == 7){
            result.emplace_back(in_x[i] + Intervaltemp);
        }else if(basic->get_type() == 6){
            result.emplace_back(in_x[i] - Intervaltemp);
        }else if(basic->get_type() == 8){
            result.emplace_back(in_x[i].cwiseProduct(Intervaltemp));
        }else if(basic->get_type() == 9){
            if (constant[i](0, 0) == 0) {
                std::cerr << "Error: Division by zero." << std::endl;
                assert(!(constant[i](0, 0) == 0));
            }else{
                result.emplace_back(in_x[i].cwiseQuotient(Intervaltemp));
            }
        }
    }
    std::cout<<"The result matrix: ("<<result.size()<<", "<<result[0].rows()<<", "<<result[0].cols()<<") "<<std::endl;
    return result;
}


IntervalMatrices IntervalSolver::MaxPoolNeuronNodeevaluate( const MaxPoolNeuronNode *maxpool){
    std::cout<<"Maxpooling....."<<std::endl;
    IntervalMatrices out_x;
    u32_t in_height = in_x[0].rows();
    u32_t in_width = in_x[0].cols();
    u32_t pad_height = maxpool->pad_height;
    u32_t pad_width = maxpool->pad_width;
    u32_t window_width = maxpool->window_width;
    u32_t window_height = maxpool->window_height;
    u32_t stride_height = maxpool->stride_height;
    u32_t stride_width = maxpool->stride_width;

    /// IntervalMatrix -> (upper mat, lower mat)->  ( maxpooling(upper mat), maxpooling(lower mat)) ->IntervalMatrix

    /// convert IntervalMatrix into UpperMatrix & LowerMatrix
    /// std::vector<Eigen::MatrixXd>
    std::vector<Eigen::MatrixXd> Uppermat = splitIntervalMatrices(in_x).second;
    std::vector<Eigen::MatrixXd> Lowermat = splitIntervalMatrices(in_x).first;

    for (u32_t depth = 0; depth < in_x.size(); ++depth) {
        /// Padding
        u32_t padded_height = in_height + 2 * pad_height;
        u32_t padded_width = in_width + 2 * pad_width;
        Eigen::MatrixXd paddedMatrixU = Eigen::MatrixXd::Zero(padded_height, padded_width);
        Eigen::MatrixXd paddedMatrixL = Eigen::MatrixXd::Zero(padded_height, padded_width);

        paddedMatrixU.block(pad_height, pad_width, in_height, in_width) = Uppermat[depth];
        paddedMatrixL.block(pad_height, pad_width, in_height, in_width) = Lowermat[depth];

        /// Calculate the size of the output feature map
        u32_t outHeight = (padded_height - window_height) / stride_height + 1;
        u32_t outWidth = (padded_width - window_width) / stride_width + 1;
        Eigen::MatrixXd outMatrixU(outHeight, outWidth);
        Eigen::MatrixXd outMatrixL(outHeight, outWidth);

        for (u32_t i = 0; i < outHeight; ++i) {
            for (u32_t j = 0; j < outWidth; ++j) {
                double maxValU = -std::numeric_limits<double>::infinity();
                double maxValL = -std::numeric_limits<double>::infinity();
                for (u32_t m = 0; m < window_height; ++m) {
                    for (u32_t n = 0; n < window_width; ++n) {
                        u32_t rowIndex = i * stride_height + m;
                        u32_t colIndex = j * stride_width + n;
                        double currentValU = paddedMatrixU(rowIndex, colIndex);
                        double currentValL = paddedMatrixL(rowIndex, colIndex);
                        if (currentValU > maxValU) {
                            maxValU = currentValU;
                        }if(currentValL > maxValL){
                            maxValL = currentValL;
                        }
                    }
                }
                outMatrixU(i, j) = maxValU;
                outMatrixL(i, j) = maxValL;
            }
        }
        // 初始化IntervalEigen对象，大小与outMatrixU和outMatrixL相同
        IntervalMat intervalMatrix(outHeight, outWidth);

        // 遍历矩阵元素，为每个元素创建IntervalValue
        for (int i = 0; i < outHeight; ++i) {
            for (int j = 0; j < outWidth; ++j) {
                double lower = outMatrixL(i, j); // 下界值
                double upper = outMatrixU(i, j); // 上界值

                // 创建IntervalValue对象并赋值给intervalMatrix的相应位置
                // 假设SVF::IntervalValue有一个接受两个double参数的构造函数，分别代表下界和上界
                intervalMatrix(i, j) = IntervalValue(lower, upper);
            }
        }
        out_x.push_back(intervalMatrix);
    }
    return out_x;
}

IntervalMatrices IntervalSolver::FullyConNeuronNodeevaluate( const FullyConNeuronNode *fully){
    std::cout<<"FullyConing......"<<std::endl;
    /// The step of processing input flattening operation is equivalent to the GEMM node operation in ONNX
    u32_t in_depth = in_x.size();
    u32_t in_height = in_x[0].rows();
    u32_t in_width = in_x[0].cols();

    IntervalMat Intervalweight = convertMatToIntervalMat(fully->weight);
    IntervalMat Intervalbias = convertVectorXdToIntervalVector(fully->bias);

    ///  1, b.size(), 1
    u32_t out_width = 1;
    u32_t out_height = Intervalbias.size();
    u32_t out_depth = 1;

    const u32_t rowsize = in_depth * in_height * in_width;
    IntervalMat x_ser(rowsize,1);

    for (u32_t i = 0; i < in_depth; i++) {
        for (u32_t j = 0; j < in_height; j++) {
            for (u32_t k = 0; k < in_width; k++) {
                x_ser(in_width * in_depth * j + in_depth * k + i, 0) = in_x[i](j, k);
            }
        }
    }

    ///wx+b
    Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> val = Intervalweight * x_ser + Intervalbias;


    /// Restore output
    IntervalMatrices out;

    /// Assignment
    for (u32_t i = 0; i < out_depth; i++) {
        out.push_back(Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic>(out_height, out_width));
        for (u32_t j = 0; j < out_height; j++) {
            for (u32_t k = 0; k < out_width; k++) {
                out[i](j, k) = val(out_width * out_depth * j + out_depth * k + i,0);
            }
        }
    }
    return out;
}

IntervalMatrices IntervalSolver::ConvNeuronNodeevaluate( const ConvNeuronNode *conv){
    std::cout<<"ConvNodeing......"<<conv->getId()<<std::endl;

    u32_t filter_num = conv->get_filter_num();
    u32_t stride = conv->get_stride();
    u32_t padding = conv->get_padding();
    std::vector<FilterSubNode> filter = conv->get_filter();

    u32_t filter_depth = conv->get_filter_depth();
    u32_t filter_height = conv->get_filter_height();
    u32_t filter_width = conv->get_filter_width();
    std::vector<double> bias = conv->get_bias();
    std::vector<SVF::IntervalValue> intervalbias;

    for (double val : bias) {
        SVF::IntervalValue intervalValue(val, val);
        intervalbias.push_back(intervalValue);
    }

    u32_t out_height = ((in_x[0].rows() - filter[0].get_height() + 2*padding) / stride) + 1;
    u32_t out_width = ((in_x[0].cols() - filter[0].get_width() + 2*padding) / stride) + 1;

    /// Padding
//    std::vector<Eigen::MatrixXd> padded_x(in_x.size());
    IntervalMatrices padded_x(in_x.size());
    for (u32_t i = 0; i < in_x.size(); ++i) {
        padded_x[i] = convertMatToIntervalMat(Eigen::MatrixXd::Zero(in_x[i].rows() + 2*padding, in_x[i].cols() + 2*padding));
        padded_x[i].block(padding, padding, in_x[i].rows(), in_x[i].cols()) = in_x[i];
    }

    /// Calculate the output feature map based on filling and step size
    IntervalMatrices out(filter_num, IntervalMat (out_height, out_width));

    for (u32_t i = 0; i < filter_num; i++) {
        for (u32_t j = 0; j < out_width; j++) {
            for (u32_t k = 0; k < out_height; k++) {
                IntervalValue sum(0.0, 0.0);
                for (u32_t i_ = 0; i_ < filter_depth; i_++) {
                    for (u32_t j_ = 0; j_ < filter_height; j_++) {
                        for (u32_t k_ = 0; k_ < filter_width; k_++) {
                            /// Strides
                            u32_t row = k * stride + j_;
                            int col = j * stride + k_;
//                            std::vector<Eigen::MatrixXd> filtervalue;
                            if (row < padded_x[i_].rows() && col < padded_x[i_].cols()) {
                                sum += convertMatricesToIntervalMatrices(filter[i].value)[i_](j_, k_) * padded_x[i_](row, col);
                            }
                        }
                    }
                }
                /// Calculate the output at the current position and add a bias
                out[i](k, j) = sum + intervalbias[i];
            }
        }
    }
    return out;
}

IntervalMatrices IntervalSolver::ConstantNeuronNodeevaluate(){
    std::cout<<"Constanting....... The input size: ("<<in_x.size()<<", "<<in_x[0].rows()<<", "<<in_x[0].cols()<<")"<<std::endl;
    /// This is an entry, nothing needs to do.
    return in_x;
}
