#include "AE/Nnexe/Solver.h"

using namespace SVF;

Matrices SolverEvaluate::ReLuNeuronNodeevaluate() const{
    std::cout<<"Reluing....."<<std::endl;
    Matrices x_out;
    for (Mat mat : in_x) {
        /* rows cols
            * Construct a List
         */
        Mat o(mat.rows(), mat.cols());
        for (u32_t i = 0; i < mat.rows(); i++) {
            for (u32_t j = 0; j < mat.cols(); j++) {
                o(i, j) = std::max(0.0, mat(i, j));
            }
        }
        /// Channel
        x_out.push_back(o);
    }
    return x_out;
}

Matrices SolverEvaluate::FlattenNeuronNodeevaluate() const{
    std::cout<<"Flattening....."<<std::endl;

    u32_t in_depth = in_x.size();
    u32_t in_height = in_x[0].rows();
    u32_t in_width = in_x[0].cols();

    Matrices x_out;

    Vector flatten_x(in_depth * in_height * in_width);
    for (u32_t i = 0; i < in_depth; i++) {
        for (u32_t j = 0; j < in_height; j++) {
            for (u32_t k = 0; k < in_width; k++) {
                flatten_x(in_width * in_depth * j + in_depth * k + i) = in_x[i](j, k);
            }
        }
    }
    Mat temp = flatten_x;
    x_out.push_back(temp);

    return x_out;
}

Matrices SolverEvaluate::BasicOPNeuronNodeevaluate( const BasicOPNeuronNode *basic) const{
    std::cout<<"BasicNoding......"<<std::endl;

    Matrices result;

    /// ensure A and B the number of depth equal
    auto constant = basic->constant;

    if (in_x.size() != constant.size()) {
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
        Mat temp = Mat::Constant(in_x[i].rows(), in_x[i].cols(), constant[i](0,0));

        /// Subtract the channel of A from the above matrix

        if(basic->get_type() == 7){
            result.emplace_back(in_x[i] + temp);
        }else if(basic->get_type() == 6){
            result.emplace_back(in_x[i] - temp);
        }else if(basic->get_type() == 8){
            result.emplace_back(in_x[i].cwiseProduct(temp));
        }else if(basic->get_type() == 9){
            if (constant[i](0, 0) == 0) {
                std::cerr << "Error: Division by zero." << std::endl;
                assert(!(constant[i](0, 0) == 0));
            }else{
                result.emplace_back(in_x[i].cwiseQuotient(temp));
            }
        }
    }
    return result;
}

Matrices SolverEvaluate::MaxPoolNeuronNodeevaluate( const MaxPoolNeuronNode *maxpool) const{
    std::cout<<"Maxpooling....."<<std::endl;
    Matrices out_x;
    u32_t in_height = in_x[0].rows();
    u32_t in_width = in_x[0].cols();
    u32_t pad_height = maxpool->pad_height;
    u32_t pad_width = maxpool->pad_width;
    u32_t window_width = maxpool->window_width;
    u32_t window_height = maxpool->window_height;
    u32_t stride_height = maxpool->stride_height;
    u32_t stride_width = maxpool->stride_width;

    for (size_t depth = 0; depth < in_x.size(); ++depth) {
        /// Padding
        u32_t padded_height = in_height + 2 * pad_height;
        u32_t padded_width = in_width + 2 * pad_width;
        Mat paddedMatrix = Mat::Zero(padded_height, padded_width);
        paddedMatrix.block(pad_height, pad_width, in_height, in_width) = in_x[depth];

        /// Calculate the size of the output feature map
        u32_t outHeight = (padded_height - window_height) / stride_height + 1;
        u32_t outWidth = (padded_width - window_width) / stride_width + 1;
        Mat outMatrix(outHeight, outWidth);

        for (u32_t i = 0; i < outHeight; ++i) {
            for (u32_t j = 0; j < outWidth; ++j) {
                double maxVal = -std::numeric_limits<double>::infinity();
                for (u32_t m = 0; m < window_height; ++m) {
                    for (u32_t n = 0; n < window_width; ++n) {
                        u32_t rowIndex = i * stride_height + m;
                        u32_t colIndex = j * stride_width + n;
                        double currentVal = paddedMatrix(rowIndex, colIndex);
                        if (currentVal > maxVal) {
                            maxVal = currentVal;
                        }
                    }
                }
                outMatrix(i, j) = maxVal;
            }
        }
        out_x.push_back(outMatrix);
    }
    return out_x;
}

Matrices SolverEvaluate::FullyConNeuronNodeevaluate( const FullyConNeuronNode *fully) const{
    std::cout<<"FullyConing......"<<std::endl;
    /// The step of processing input flattening operation is equivalent to the GEMM node operation in ONNX
    u32_t in_depth = in_x.size();
    Mat weight = fully->weight;
    Mat bias = fully->bias;

    ///wx+b
    /// ensure(1, y, 1)
    for (const auto& mat : in_x) {
        if (mat.rows() != weight.cols() || mat.cols() != 1) {
            throw std::runtime_error("In Flatten, false init");
        }
    }

    /// c(x, 1)
    if (bias.rows() != weight.rows() || bias.cols() != 1) {
        throw std::runtime_error("In flatten, bias is wrong");
    }

    Matrices out_x;
    out_x.reserve(in_depth);

    /// weight * in_x[i] + bias -> out_x
    for (const auto& mat : in_x) {
        Eigen::MatrixXd tempResult = weight * mat + bias;
        out_x.push_back(tempResult);
    }

    return out_x;
}

Matrices SolverEvaluate::ConvNeuronNodeevaluate( const ConvNeuronNode *conv) const{
    std::cout<<"ConvNodeing......"<<conv->getId()<<std::endl;

    u32_t filter_num = conv->get_filter_num();
    u32_t stride = conv->get_stride();
    u32_t padding = conv->get_padding();
    std::vector<FilterSubNode> filter = conv->get_filter();
    u32_t filter_depth = conv->get_filter_depth();
    u32_t filter_height = conv->get_filter_height();
    u32_t filter_width = conv->get_filter_width();
    std::vector<double> bias = conv->get_bias();
    u32_t out_height = ((in_x[0].rows() - filter[0].get_height() + 2*padding) / stride) + 1;
    u32_t out_width = ((in_x[0].cols() - filter[0].get_width() + 2*padding) / stride) + 1;

    /// Padding
    Matrices padded_x(in_x.size());
    for (u32_t i = 0; i < in_x.size(); ++i) {
        padded_x[i] = Mat::Zero(in_x[i].rows() + 2*padding, in_x[i].cols() + 2*padding);
        padded_x[i].block(padding, padding, in_x[i].rows(), in_x[i].cols()) = in_x[i];
    }

    /// Calculate the output feature map based on filling and step size
    Matrices out(filter_num, Mat(out_height, out_width));
    for (u32_t i = 0; i < filter_num; i++) {
        for (u32_t j = 0; j < out_width; j++) {
            for (u32_t k = 0; k < out_height; k++) {
                double sum = 0;
                for (u32_t i_ = 0; i_ < filter_depth; i_++) {
                    for (u32_t j_ = 0; j_ < filter_height; j_++) {
                        for (u32_t k_ = 0; k_ < filter_width; k_++) {
                            /// Strides
                            u32_t row = k * stride + j_;
                            u32_t col = j * stride + k_;
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

Matrices SolverEvaluate::ConstantNeuronNodeevaluate() const{
    std::cout<<"Constanting......."<<std::endl;
    /// This is an entry, nothing needs to do.
    return in_x;
}
