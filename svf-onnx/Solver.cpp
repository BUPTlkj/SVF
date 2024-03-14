#include "Solver.h"

//// FOR Test
void TraverseVectorOfMatrices(const std::vector<Eigen::MatrixXd>& in_x) {
    for (const auto& matrix : in_x) {
        std::cout << "Matrix size: " << matrix.rows() << "x" << matrix.cols() << std::endl;
        std::cout << "Matrix content:\n" << matrix << std::endl << std::endl;
    }
}

std::vector<Eigen::MatrixXd> SolverEvaluate::ReLuNeuronNodeevaluate() const{
    std::cout<<"Reluing....."<<std::endl;
    std::vector<Eigen::MatrixXd> x_out;
    for (Eigen::MatrixXd mat : in_x) {
        /* rows cols
            * Construct a List
         */
        Eigen::MatrixXd o(mat.rows(), mat.cols());
        for (unsigned i = 0; i < mat.rows(); i++) {
            for (unsigned j = 0; j < mat.cols(); j++) {
                o(i, j) = std::max(0.0, mat(i, j));
            }
        }
        /// Channel
        x_out.push_back(o);
    }
    std::cout<<"***************"<<std::endl;
    std::cout<<in_x[0].cols()<<"   "<<in_x[0].rows()<<std::endl;
    std::cout<<x_out[0].cols()<<"   "<<x_out[0].rows()<<std::endl;
    return x_out;
}

std::vector<Eigen::MatrixXd> SolverEvaluate::BasicOPNeuronNodeevaluate( const SVF::BasicOPNeuronNode *basic) const{
    std::cout<<"BasicNoding......"<<std::endl;
//    TraverseVectorOfMatrices(in_x);
//    TraverseVectorOfMatrices(basic->constant);

    std::vector<Eigen::MatrixXd> result;

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
        Eigen::MatrixXd temp = Eigen::MatrixXd::Constant(in_x[i].rows(), in_x[i].cols(), constant[i](0,0));

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
//    TraverseVectorOfMatrices(result);
    return result;
}

std::vector<Eigen::MatrixXd> SolverEvaluate::MaxPoolNeuronNodeevaluate( const SVF::MaxPoolNeuronNode *maxpool) const{
    std::cout<<"Maxpooling....."<<std::endl;
    std::vector<Eigen::MatrixXd> out_x;
    auto in_height = in_x[0].rows();
    auto in_width = in_x[0].cols();
    auto pad_height = maxpool->pad_height;
    auto pad_width = maxpool->pad_width;
    auto window_width = maxpool->window_width;
    auto window_height = maxpool->window_height;
    auto stride_height = maxpool->stride_height;
    auto stride_width = maxpool->stride_width;

    for (size_t depth = 0; depth < in_x.size(); ++depth) {
        /// Padding
        unsigned padded_height = in_height + 2 * pad_height;
        unsigned padded_width = in_width + 2 * pad_width;
        Eigen::MatrixXd paddedMatrix = Eigen::MatrixXd::Zero(padded_height, padded_width);
        paddedMatrix.block(pad_height, pad_width, in_height, in_width) = in_x[depth];

        /// Calculate the size of the output feature map
        unsigned outHeight = (padded_height - window_height) / stride_height + 1;
        unsigned outWidth = (padded_width - window_width) / stride_width + 1;
        Eigen::MatrixXd outMatrix(outHeight, outWidth);

        for (unsigned i = 0; i < outHeight; ++i) {
            for (unsigned j = 0; j < outWidth; ++j) {
                double maxVal = -std::numeric_limits<double>::infinity();
                for (unsigned m = 0; m < window_height; ++m) {
                    for (unsigned n = 0; n < window_width; ++n) {
                        unsigned rowIndex = i * stride_height + m;
                        unsigned colIndex = j * stride_width + n;
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

std::vector<Eigen::MatrixXd> SolverEvaluate::FullyConNeuronNodeevaluate( const SVF::FullyConNeuronNode *fully) const{
    std::cout<<"FullyConing......"<<std::endl;
    /// The step of processing input flattening operation is equivalent to the GEMM node operation in ONNX
    auto in_depth = in_x.size();
    auto in_height = in_x[0].rows();
    auto in_width = in_x[0].cols();
    auto weight = fully->weight;
    auto bias = fully->bias;
//       1, b.size(), 1
    unsigned out_width = 1;
    unsigned out_height = bias.size();
    unsigned out_depth = 1;

    Eigen::VectorXd x_ser(in_depth * in_height * in_width);
    for (unsigned i = 0; i < in_depth; i++) {
        for (unsigned j = 0; j < in_height; j++) {
            for (unsigned k = 0; k < in_width; k++) {
                x_ser(in_width * in_depth * j + in_depth * k + i) = in_x[i](j, k);
            }
        }
    }

    ///wx+b
    Eigen::VectorXd val = weight * x_ser + bias;

    /// Restore output
    std::vector<Eigen::MatrixXd> out;

    /// Assignment
    for (unsigned i = 0; i < out_depth; i++) {
        out.push_back(Eigen::MatrixXd(out_height, out_width));
        for (unsigned j = 0; j < out_height; j++) {
            for (unsigned k = 0; k < out_width; k++) {
                out[i](j, k) = val(out_width * out_depth * j + out_depth * k + i);
            }
        }
    }
    return out;
}

std::vector<Eigen::MatrixXd> SolverEvaluate::ConvNeuronNodeevaluate( const SVF::ConvNeuronNode *conv) const{
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

    std::cout<<"IMPORT FILTER NUMBER: "<<filter_num<<"  "<<out_height<<"   "<<out_width<<std::endl;
    std::cout<<"FILTER NUMBER: "<<filter[0].get_height()<<"  "<<filter[0].get_width()<<"   "<<std::endl;
    std::cout<<"FILTER NUMBER: "<<filter_height<<"  "<<filter_width<<"   "<<std::endl;
    std::cout<<"MATRIX NUMBER: "<<in_x[0].rows()<<"  "<<in_x[0].cols()<<"   "<<std::endl;

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

std::vector<Eigen::MatrixXd> SolverEvaluate::ConstantNeuronNodeevaluate() const{
    std::cout<<"Constanting......."<<std::endl;
    /// This is an entry, nothing needs to do.
    return in_x;
}
