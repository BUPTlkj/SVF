#include "IntervalSolver.h"

using namespace SVF;
template<typename T>
bool isDouble(T&& var) {
    return std::is_same<typename std::remove_cv<typename std::remove_reference<T>::type>::type, double>::value;
}

template<typename T>
bool isInt(T&& var) {
    return std::is_same<typename std::remove_cv<typename std::remove_reference<T>::type>::type, u32_t>::value;
}

//typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> myMatrixXd;
void IntervalSolver::initializeMatrix() {

//    typedef Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> myMatrixXd;
//    u32_t a = 2;
//    u32_t b = 3;
//    // u32_t c = 2;
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
//    for (u32_t i = 0; i < output.rows(); ++i) {
//        for (u32_t j = 0; j < output.cols(); ++j) {
//            std::cout << output(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }

    /// Define ab
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

    Eigen::Matrix<IntervalValue, Eigen::Dynamic,  Eigen::Dynamic>  c = aw * matrixq + aw;
    for (u32_t i = 0; i < c.rows(); ++i) {
        for (u32_t j = 0; j < c.cols(); ++j) {
            std::cout <<"[" << aw(i,j).lb().getRealNumeral() << ", "  << aw(i,j).ub().getRealNumeral()<< "]\t";
            std::cout << aw(i,j).toString()<< "]\t";
        }
        std::cout << std::endl;
    }

    for (u32_t i = 0; i < c.rows(); ++i) {
        for (u32_t j = 0; j < c.cols(); ++j) {
            std::cout <<"[" << c(i,j).lb().getRealNumeral() << ", "  << c(i,j).ub().getRealNumeral()<< "]\t";
        }
        std::cout << std::endl;
    }
}

std::pair<Matrices, Matrices> IntervalSolver::splitIntervalMatrices(const IntervalMatrices & intervalMatrices) {
    Matrices lowerBounds, upperBounds;

    for (const auto& intervalMatrix : intervalMatrices) {
        Mat lower(intervalMatrix.rows(), intervalMatrix.cols());
        Mat upper(intervalMatrix.rows(), intervalMatrix.cols());

        for (u32_t i = 0; i < intervalMatrix.rows(); ++i) {
            for (u32_t j = 0; j < intervalMatrix.cols(); ++j) {
                lower(i, j) = intervalMatrix(i, j).lb().getRealNumeral();
                upper(i, j) = intervalMatrix(i, j).ub().getRealNumeral();
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
    for (const auto& mat : in_x) { /// using auto& avoid copy
        IntervalMat o(mat.rows(), mat.cols());
        for (u32_t i = 0; i < mat.rows(); i++) {
            for (u32_t j = 0; j < mat.cols(); j++) {
                const auto& iv = mat(i, j); /// Current interval
                double lb = iv.lb().getRealNumeral(); /// lower bound
                double ub = iv.ub().getRealNumeral(); /// upper bound

                /// Relu
                double newLb = std::max(0.0, lb);
                double newUb = std::max(0.0, ub);

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

    /// ensure A and B the number of depth equal
    IntervalMatrices Intervalconstant = basic->Intervalconstant;

    if (in_x.size() != Intervalconstant.size()) {
        std::cout<<"In_x.size(): "<<in_x.size()<<",  "<<"Intervalconstant.size()): "<<Intervalconstant.size()<<std::endl;
        std::cerr << "Error: The matrix of channels must be the same." << std::endl;
        return result;
    }

    for (size_t i = 0; i < in_x.size(); ++i) {
        /// ensure the size of each channel correct
        if (Intervalconstant[i].rows() != 1 || Intervalconstant[i].cols() != 1) {
            std::cerr << "Error: B's channel matrices must be 1x1." << std::endl;
            return result;
        }

        /// Create a matrix of the same size as the channel of A,
        /// where all elements are the corresponding channel values of B
        IntervalMat Intervaltemp(
            in_x[i].rows(), in_x[i].cols());
        for (u32_t j = 0; j < Intervaltemp.rows(); ++j) {
            for (u32_t k = 0; k < Intervaltemp.cols(); ++k) {
                Intervaltemp(j, k) = IntervalValue(Intervalconstant[i](0,0).lb(), Intervalconstant[i](0,0).ub());
            }
        }

        /// Subtract the channel of A from the above matrix
        if(basic->get_type() == 7){
            result.emplace_back(in_x[i] + Intervaltemp);
        }else if(basic->get_type() == 6){
            result.emplace_back(in_x[i] - Intervaltemp);
        }else if(basic->get_type() == 8){
            result.emplace_back(in_x[i].cwiseProduct(Intervaltemp));
        }else if(basic->get_type() == 9){
            if (Intervalconstant[i](0, 0).ub().getRealNumeral() == 0) {
                std::cerr << "Error: Division by zero." << std::endl;
                assert(!(Intervalconstant[i](0, 0).getRealNumeral() == 0));
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
    /// Matrices
    Matrices Uppermat = splitIntervalMatrices(in_x).second;
    Matrices Lowermat = splitIntervalMatrices(in_x).first;

    for (u32_t depth = 0; depth < in_x.size(); ++depth) {
        /// Padding
        u32_t padded_height = in_height + 2 * pad_height;
        u32_t padded_width = in_width + 2 * pad_width;
        Mat paddedMatrixU = Mat::Zero(padded_height, padded_width);
        Mat paddedMatrixL = Mat::Zero(padded_height, padded_width);

        paddedMatrixU.block(pad_height, pad_width, in_height, in_width) = Uppermat[depth];
        paddedMatrixL.block(pad_height, pad_width, in_height, in_width) = Lowermat[depth];

        /// Calculate the size of the output feature map
        u32_t outHeight = (padded_height - window_height) / stride_height + 1;
        u32_t outWidth = (padded_width - window_width) / stride_width + 1;
        Mat outMatrixU(outHeight, outWidth);
        Mat outMatrixL(outHeight, outWidth);

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
        /// IntervalMat£¬(outMatrixU, outMatrixL)
        IntervalMat intervalMatrix(outHeight, outWidth);

        for (u32_t i = 0; i < outHeight; ++i) {
            for (u32_t j = 0; j < outWidth; ++j) {
                double lower = outMatrixL(i, j); /// upper
                double upper = outMatrixU(i, j); /// lower

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

    IntervalMat Intervalweight = fully->Intervalweight;
    IntervalMat Intervalbias = fully->Intervalbias;

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
    IntervalMat val = Intervalweight * x_ser + Intervalbias;

    /// Restore output
    IntervalMatrices out;

    /// Assignment
    for (u32_t i = 0; i < out_depth; i++) {
        out.push_back(IntervalMat(out_height, out_width));
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
    std::vector<SVF::IntervalValue> intervalbias = conv->Intervalbias;

    u32_t out_height = ((in_x[0].rows() - filter[0].get_height() + 2*padding) / stride) + 1;
    u32_t out_width = ((in_x[0].cols() - filter[0].get_width() + 2*padding) / stride) + 1;

    /// Padding
    IntervalMatrices padded_x(in_x.size());
    for (u32_t i = 0; i < in_x.size(); ++i) {
        for (size_t j = 0; j < in_x.size(); ++j){
            padded_x[j] = IntervalMat(in_x[j].rows() + 2 * padding, in_x[j].cols() + 2 * padding);

            /// [0.0, 0.0]
            for (u32_t k = 0; k < padded_x[j].rows(); ++k){
                for (u32_t l = 0; l < padded_x[j].cols(); ++l){
                    padded_x[j](k, l) = IntervalValue(0.0, 0.0);
                }
            }
        }
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
                            u32_t col = j * stride + k_;
                            if (row < padded_x[i_].rows() && col < padded_x[i_].cols()) {
                                sum += filter[i].Intervalvalue[i_](j_, k_) * padded_x[i_](row, col);
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

IntervalMatrices IntervalSolver::ConstantNeuronNodeevaluate() const{
    std::cout<<"Constanting....... "<<std::endl;
    /// This is an entry, nothing needs to do.
    return in_x;
}
