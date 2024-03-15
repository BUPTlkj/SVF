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


typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> myMatrixXd;

void IntervalSolver::initializeMatrix() {
    Eigen::Matrix<SVF::IntervalValue, 2, 2> matrix;
    matrix(0,0) = SVF::IntervalValue(2.1, 2.1);
    matrix(0,1) = SVF::IntervalValue(0,0);
    matrix(1,0) = SVF::IntervalValue(0,0);
    matrix(1,1) = SVF::IntervalValue(2.34, 2.45);

    SVF::IntervalValue inv(0.34, 1.45);

    Eigen::Matrix<SVF::IntervalValue, 2, 2> a;
    a(0,0) = inv;
    a(0,1) = inv + inv;
    a(1,0) = inv ;
    a(1,1) = inv;

    Eigen::Matrix<SVF::IntervalValue, 2, 2>  c = a * matrix;

    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            std::cout << a(i,j).toString() << "\t";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            std::cout << c(i,j).toString() << "\t";
        }
        std::cout << std::endl;
    }
}

IntervalMatrix IntervalSolver::convertMatricesToIntervalMatrices(const std::vector<Eigen::MatrixXd>& matrices) {
    IntervalMatrix intervalMatrices;

    for (const auto& mat : matrices) {
        Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic> intervalMatrix(mat.rows(), mat.cols());
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                // 对于每个元素x，创建一个区间[x, x]
                std::cout<<"Matrix: ("<<i<<", "<<j<<") :  "<<mat(i,j)<<" Is double? "<<isDouble(mat(i,j))<<std::endl;
                intervalMatrix(i, j) = SVF::IntervalValue(mat(i, j), mat(i, j));
//                intervalMatrix(i, j) = SVF::IntervalValue(static_cast<double>(mat(i, j)), static_cast<double>(mat(i, j)));
            }
        }
        intervalMatrices.push_back(intervalMatrix);
    }
    return intervalMatrices;
}

IntervalEigen IntervalSolver::convertEigenToIntervalEigen(const Eigen::MatrixXd& matrix){

    Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic> intervalMatrix(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            // 对于每个元素x，创建一个区间[x, x]
            std::cout<<"Matrix: ("<<i<<", "<<j<<") :  "<<matrix(i,j)<<" Is double? "<<isDouble(matrix(i,j))<<std::endl;
            intervalMatrix(i, j) = SVF::IntervalValue(matrix(i, j), matrix(i, j));
        }
    }
    return intervalMatrix;
}

//IntervalVector IntervalSolver::convertVectorXdToIntervalVector(const Eigen::VectorXd& vec) {
//    // 创建一个具有相同大小的IntervalVector
//    IntervalVector intervalVec(vec.size());
//
//    // 遍历VectorXd中的每个元素
//    for (int i = 0; i < vec.size(); ++i) {
//        // 将每个double值转换为一个具有相同上下界的IntervalValue
//        intervalVec(i) = SVF::IntervalValue(vec(i), vec(i)); // 假设IntervalValue有一个接受两个double的构造函数
//    }
//
//    return intervalVec;
//}

IntervalEigen IntervalSolver::convertVectorXdToIntervalVector(const Eigen::VectorXd& vec) {
    // 创建一个具有相同行数的IntervalEigen，列数为1
    IntervalEigen intervalMat(vec.size(), 1);

    // 遍历VectorXd中的每个元素
    for (int i = 0; i < vec.size(); ++i) {
        // 将每个double值转换为一个具有相同上下界的IntervalValue
        intervalMat(i, 0) = SVF::IntervalValue(vec(i), vec(i)); // 需要IntervalValue有一个接受两个double的构造函数
    }

    return intervalMat;
}


std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> IntervalSolver::splitIntervalMatrices(const std::vector<Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic>>& intervalMatrices) {
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



IntervalMatrix SVF::IntervalSolver::ReLuNeuronNodeevaluate() const {
    std::cout << "Reluing....." << std::endl;
    IntervalMatrix x_out;
    for (const auto& mat : in_x) { // 直接使用auto&避免拷贝
        Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic> o(mat.rows(), mat.cols());
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
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


IntervalMatrix SVF::IntervalSolver::BasicOPNeuronNodeevaluate( const SVF::BasicOPNeuronNode *basic){
    std::cout<<"BasicNoding......"<<std::endl;


    IntervalMatrix result;
//    std::vector<Eigen::MatrixXd> result;

    /// ensure A and B the number of depth equal
    auto constant = basic->constant;

    auto Intervalconstant = convertMatricesToIntervalMatrices(constant);

    if (in_x.size() != Intervalconstant.size()) {
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


        auto Intervaltemp = convertEigenToIntervalEigen(temp);

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
    return result;
}


IntervalMatrix IntervalSolver::MaxPoolNeuronNodeevaluate( const SVF::MaxPoolNeuronNode *maxpool){
    std::cout<<"Maxpooling....."<<std::endl;
    IntervalMatrix out_x;
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

    for (size_t depth = 0; depth < in_x.size(); ++depth) {
        /// Padding
        unsigned padded_height = in_height + 2 * pad_height;
        unsigned padded_width = in_width + 2 * pad_width;
        Eigen::MatrixXd paddedMatrixU = Eigen::MatrixXd::Zero(padded_height, padded_width);
        Eigen::MatrixXd paddedMatrixL = Eigen::MatrixXd::Zero(padded_height, padded_width);

        paddedMatrixU.block(pad_height, pad_width, in_height, in_width) = Uppermat[depth];
        paddedMatrixL.block(pad_height, pad_width, in_height, in_width) = Lowermat[depth];

        /// Calculate the size of the output feature map
        unsigned outHeight = (padded_height - window_height) / stride_height + 1;
        unsigned outWidth = (padded_width - window_width) / stride_width + 1;
        Eigen::MatrixXd outMatrixU(outHeight, outWidth);
        Eigen::MatrixXd outMatrixL(outHeight, outWidth);

        for (unsigned i = 0; i < outHeight; ++i) {
            for (unsigned j = 0; j < outWidth; ++j) {
                double maxValU = -std::numeric_limits<double>::infinity();
                double maxValL = -std::numeric_limits<double>::infinity();
                for (unsigned m = 0; m < window_height; ++m) {
                    for (unsigned n = 0; n < window_width; ++n) {
                        unsigned rowIndex = i * stride_height + m;
                        unsigned colIndex = j * stride_width + n;
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
        IntervalEigen intervalMatrix(outHeight, outWidth);

        // 遍历矩阵元素，为每个元素创建IntervalValue
        for (int i = 0; i < outHeight; ++i) {
            for (int j = 0; j < outWidth; ++j) {
                double lower = outMatrixL(i, j); // 下界值
                double upper = outMatrixU(i, j); // 上界值

                // 创建IntervalValue对象并赋值给intervalMatrix的相应位置
                // 假设SVF::IntervalValue有一个接受两个double参数的构造函数，分别代表下界和上界
                intervalMatrix(i, j) = SVF::IntervalValue(lower, upper);
            }
        }
        out_x.push_back(intervalMatrix);
    }
    return out_x;
}

SVF::IntervalMatrix SVF::IntervalSolver::FullyConNeuronNodeevaluate( const SVF::FullyConNeuronNode *fully){
    std::cout<<"FullyConing......"<<std::endl;
    /// The step of processing input flattening operation is equivalent to the GEMM node operation in ONNX
    u32_t in_depth = in_x.size();
    u32_t in_height = in_x[0].rows();
    u32_t in_width = in_x[0].cols();

    IntervalEigen Intervalweight = convertEigenToIntervalEigen(fully->weight);
    IntervalEigen Intervalbias = convertVectorXdToIntervalVector(fully->bias);

    ///  1, b.size(), 1
    u32_t out_width = 1;
    u32_t out_height = Intervalbias.size();
    u32_t out_depth = 1;

    Eigen::VectorXd ser(in_depth * in_height * in_width);
    IntervalEigen x_ser =convertVectorXdToIntervalVector(ser);
    for (u32_t i = 0; i < in_depth; i++) {
        for (u32_t j = 0; j < in_height; j++) {
            for (u32_t k = 0; k < in_width; k++) {
                x_ser(in_width * in_depth * j + in_depth * k + i, 0) = in_x[i](j, k);
            }
        }
    }

    ///wx+b
    IntervalEigen val = Intervalweight * x_ser + Intervalbias;

    /// Restore output
    IntervalMatrix out;

    /// Assignment
    for (u32_t i = 0; i < out_depth; i++) {
        out.push_back(convertEigenToIntervalEigen(Eigen::MatrixXd(out_height, out_width)));
        for (u32_t j = 0; j < out_height; j++) {
            for (u32_t k = 0; k < out_width; k++) {
                u32_t index = out_width * out_depth * j + out_depth * k + i;
                out[i](j, k) = val(index, 0);
            }
        }
    }
    return val;
}

SVF::IntervalMatrix SVF::IntervalSolver::ConvNeuronNodeevaluate( const SVF::ConvNeuronNode *conv) const{
    std::cout<<"ConvNodeing......"<<conv->getId()<<std::endl;

    unsigned filter_num = conv->get_filter_num();
    auto stride = conv->get_stride();
    auto padding = conv->get_padding();
    auto filter = conv->get_filter();
    auto filter_depth = conv->get_filter_depth();
    auto filter_height = conv->get_filter_height();
    auto filter_width = conv->get_filter_width();
    auto bias = conv->get_bias();
    auto out_height = ((in_x[0].rows() - filter[0].get_height() + 2*padding) / stride) + 1;
    auto out_width = ((in_x[0].cols() - filter[0].get_width() + 2*padding) / stride) + 1;

    /// Padding
    std::vector<Eigen::MatrixXd> padded_x(in_x.size());
    for (size_t i = 0; i < in_x.size(); ++i) {
        padded_x[i] = Eigen::MatrixXd::Zero(in_x[i].rows() + 2*padding, in_x[i].cols() + 2*padding);
        padded_x[i].block(padding, padding, in_x[i].rows(), in_x[i].cols()) = in_x[i];
    }

    /// Calculate the output feature map based on filling and step size
    std::vector<Eigen::MatrixXd> out(filter_num, Eigen::MatrixXd(out_height, out_width));
    for (int i = 0; i < filter_num; i++) {
        for (int j = 0; j < out_width; j++) {
            for (int k = 0; k < out_height; k++) {
                double sum = 0;
                for (int i_ = 0; i_ < filter_depth; i_++) {
                    for (int j_ = 0; j_ < filter_height; j_++) {
                        for (int k_ = 0; k_ < filter_width; k_++) {
                            /// Strides
                            int row = k * stride + j_;
                            int col = j * stride + k_;
                            if (row < padded_x[i_].rows() && col < padded_x[i_].cols()) {
                                sum += filter[i].value[i_](j_, k_) * padded_x[i_](row, col);
                            }
                        }
                    }
                }
                /// Calculate the output at the current position and add a bias
                out[i](k, j) = sum + bias[i];
            }
        }
    }
    return out;
}

SVF::IntervalMatrix SVF::IntervalSolver::ConstantNeuronNodeevaluate() const{
    std::cout<<"Constanting......."<<std::endl;
    /// This is an entry, nothing needs to do.
    return in_x;
}
