#include "Graphs/NNGraph.h"
#include "Util/SVFUtil.h"

using namespace SVF;

/// ReLu

NeuronNode::NodeK ReLuNeuronNode::get_type() const{
    return ReLuNode;
}

std::vector<Eigen::MatrixXd> ReLuNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& x_in) const{
    std::vector<Eigen::MatrixXd> x_out;
    for (Eigen::MatrixXd mat : x_in) {
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
    return x_out;
}

std::vector<Eigen::MatrixXd> ReLuNeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& in_x, const std::vector<Eigen::MatrixXd>& grad) const{
    std::vector<Eigen::MatrixXd> x_grad;
    for (unsigned i = 0; i < in_depth; i++) {
        x_grad.push_back(Eigen::MatrixXd(in_height, in_width));
        for (unsigned j = 0; j < in_height; j++) {
            for (unsigned k = 0; k < in_width; k++) {
                if (in_x[i](j, k) < 0) {
                    x_grad[i](j, k) = 0.0;
                }
                else {
                    x_grad[i](j, k) = in_x[i](j, k);
                }
            }
        }
    }
    return x_grad;
}

/// filter
unsigned int FilterSubNode::get_depth() const{
    return value.size();
}

unsigned int FilterSubNode::get_height() const{
    if (value.size() > 0) {
        /// The first channel
        return value[0].rows();
    }
    else {
        return 0;
    }
}

unsigned int FilterSubNode::get_width() const{
    if (value.size() > 0) {
        return value[0].cols();
    }
    else {
        return 0;
    }
}

double FilterSubNode::dot_product(const SVF::FilterSubNode& val_f) const{
    double sum = 0.0;
    /// Test the size of windows
    if (val_f.get_depth() != get_depth()) {
        throw std::runtime_error("Dimension-d mismatch in Filter.dot_product");
    }
    if (val_f.get_height() != get_height()) {
        throw std::runtime_error("Dimension-h mismatch in Filter.dot_product");
    }
    if (val_f.get_width() != get_width()) {
        throw std::runtime_error("Dimension-w mismatch in Filter.dot_product");
    }

    /// dot
    for (unsigned i = 0; i < get_depth(); i++) {
        for (unsigned j = 0; j < get_height(); j++) {
            for (unsigned k = 0; k < get_width(); k++) {
                sum += val_f.value[i](j, k) * value[i](j, k);
            }
        }
    }
    return sum;
}

/// BasicOPNeuronNode:including Sub, Mul, Div, and Add.
NeuronNode::NodeK BasicOPNeuronNode::get_type() const{
    if(oper == "Sub"){
        return Sub;
    }else if(oper == "Add"){
        return Add;
    }else if(oper == "Mul"){
        return Mul;
    }else if(oper == "Div"){
        return Div;
    }
    return BasicOPNode;
}

std::vector<Eigen::MatrixXd> BasicOPNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& in_x) const{
    std::vector<Eigen::MatrixXd> result;

    /// ensure A and B the number of depth equal
    if (in_x.size() != constant.size()) {
        std::cerr << "Error: The number of channels must be the same." << std::endl;
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
        if(oper == "Add"){
            result.push_back(in_x[i] + temp);
        }else if(oper == "Sub"){
            result.push_back(in_x[i] - temp);
        }else if(oper == "Mul"){
            result.push_back(in_x[i].cwiseProduct(temp));
        }else if(oper == "Div"){
            if (constant[i](0, 0) == 0) {
                std::cerr << "Error: Division by zero." << std::endl;
                assert(!(constant[i](0, 0) == 0));
            }else{
                result.push_back(in_x[i].cwiseQuotient(temp));
            }
        }
    }
    return result;
}

std::vector<Eigen::MatrixXd> BasicOPNeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& x, const std::vector<Eigen::MatrixXd>& grad) const{
    /// Flattening operation, column vector
    Eigen::VectorXd grad_x(out_height * out_width * out_depth);
    for (unsigned i = 0; i < out_depth; i++) {
        for (unsigned j = 0; j < out_height; j++) {
            for (unsigned k = 0; k < out_width; k++) {
                grad_x(out_width * out_depth * j + out_depth * k + i) = grad[i](j, k);
            }
        }
    }
    /// Gradient, this needs to be modified according to the situation todo
    Eigen::VectorXd out_col = grad_x.transpose();
    /// Rebuild the output
    std::vector<Eigen::MatrixXd> out_grad;
    for (unsigned i = 0; i < in_depth; i++) {
        out_grad.push_back(Eigen::MatrixXd(out_height, out_width));
        for (unsigned j = 0; j < in_height; j++) {
            for (unsigned k = 0; k < in_width; k++) {
                out_grad[i](j, k) = out_col(out_width * out_depth * j + out_depth * k + i);
            }
        }
    }
    return out_grad;
}

/// Maxpooling

NeuronNode::NodeK MaxPoolNeuronNode::get_type() const{
    return MaxPoolNode;
}

std::vector<Eigen::MatrixXd> MaxPoolNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& in_x) const{
    std::vector<Eigen::MatrixXd> out_x;
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

std::vector<Eigen::MatrixXd> MaxPoolNeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& in_x, const std::vector<Eigen::MatrixXd>& grad) const{
    std::vector<Eigen::MatrixXd> x_grad;
    for (unsigned i = 0; i < in_depth; i++) {
        x_grad.push_back(Eigen::MatrixXd(in_height, in_width));
        for (unsigned j = 0; j < in_height; j++) {
            for (unsigned k = 0; k < in_width; k++) {
                unsigned j_max = 0;
                unsigned k_max = 0;
                double max = in_x[i](window_height* j, window_width* k);
                for (unsigned j_ = 0; j_ < window_height; j_++) {
                    for (unsigned k_ = 0; k_ < window_width; k_++) {
                        double current_val = in_x[i](window_height* j + j_, window_width* k + k_);
                        if (current_val > max) {
                            j_max = j_;
                            k_max = k_;
                            max = current_val;
                        }
                    }
                }

                for (unsigned j_ = 0; j_ < window_height; j_++) {
                    for (unsigned k_ = 0; k_ < window_width; k_++) {
                        if ((j_ == j_max) && (k_ == k_max)) {
                            x_grad[i](window_height* j + j_, window_width* k + k_) = grad[i](j, k);
                        }
                        else {
                            x_grad[i](window_height* j + j_, window_width* k + k_) = 0.0;
                        }
                    }
                }

            }
        }
    }
    return x_grad;
}

/// FullyCon

NeuronNode::NodeK FullyConNeuronNode::get_type() const{
    return FullyConNode;
}

std::vector<Eigen::MatrixXd> FullyConNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& in_x) const{
    /// The step of processing input flattening operation is equivalent to the GEMM node operation in ONNX
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
                out[i](j, k) = x_ser(out_width * out_depth * j + out_depth * k + i);
            }
        }
    }
    return out;
}

std::vector<Eigen::MatrixXd> FullyConNeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& x, const std::vector<Eigen::MatrixXd>& grad) const{
    /// Flatten
    Eigen::VectorXd grad_x(out_height * out_width * out_depth);
    for (unsigned i = 0; i < out_depth; i++) {
        for (unsigned j = 0; j < out_height; j++) {
            for (unsigned k = 0; k < out_width; k++) {
                grad_x(out_width * out_depth * j + out_depth * k + i) = grad[i](j, k);
            }
        }
    }

    Eigen::VectorXd out_col = weight * grad_x.transpose();
    /// Rebuild the output
    std::vector<Eigen::MatrixXd> out_grad;
    for (unsigned i = 0; i < in_depth; i++) {
        out_grad.push_back(Eigen::MatrixXd(out_height, out_width));
        for (unsigned j = 0; j < in_height; j++) {
            for (unsigned k = 0; k < in_width; k++) {
                out_grad[i](j, k) = out_col(out_width * out_depth * j + out_depth * k + i);
            }
        }
    }
    return out_grad;
}

NeuronNode::NodeK ConvNeuronNode::get_type() const{
    return ConvNode;
}

std::vector<Eigen::MatrixXd> ConvNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& x) const{
    /// Padding
    std::cout<<"fun"<<std::endl;
    std::vector<Eigen::MatrixXd> padded_x(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout<<i<<std::endl;
        padded_x[i] = Eigen::MatrixXd::Zero(x[i].rows() + 2*padding, x[i].cols() + 2*padding);
        padded_x[i].block(padding, padding, x[i].rows(), x[i].cols()) = x[i];
    }
    std::cout<<"ks"<<std::endl;

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

std::vector<Eigen::MatrixXd> ConvNeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& in_x, const std::vector<Eigen::MatrixXd>& grad) const{
    return fullyer.backpropagate(in_x, grad);
}

NeuronNode::NodeK ConstantNeuronNode::get_type() const{
    return ConstantNode;
}

std::vector<Eigen::MatrixXd> ConstantNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& x) const{
    /// This is an entry, nothing needs to do.
    return x;
}

std::vector<Eigen::MatrixXd> ConstantNeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& in_x, const std::vector<Eigen::MatrixXd>& grad) const{
    std::vector<Eigen::MatrixXd> x_grad;
    for (unsigned i = 0; i < in_depth; i++) {
        x_grad.push_back(Eigen::MatrixXd(in_height, in_width));
        for (unsigned j = 0; j < in_height; j++) {
            for (unsigned k = 0; k < in_width; k++) {
                if (in_x[i](j, k) < 0) {
                    x_grad[i](j, k) = 0.0;
                }
                else {
                    x_grad[i](j, k) = in_x[i](j, k);
                }
            }
        }
    }
    return x_grad;
}

/// ques
void NeuronNode::dump() const{
    SVFUtil::outs() << this->toString() << "\n";
}

const std::string ReLuNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr <<ReLuNeuronNode::get_type() << "NNNode" << getId();
    return rawstr.str();
}

const std::string BasicOPNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << BasicOPNeuronNode::get_type() <<BasicOPNeuronNode::get_type() << "NNNode" << getId();
    return rawstr.str();
}

const std::string MaxPoolNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << MaxPoolNeuronNode::get_type() << "NNNode" << getId();
    return rawstr.str();
}

const std::string FullyConNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << FullyConNeuronNode::get_type() << "NNNode" << getId();
    return rawstr.str();
}

const std::string ConvNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << ConvNeuronNode::get_type() << "NNNode" << getId();
    return rawstr.str();
}

const std::string ConstantNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << ConstantNeuronNode::get_type() << "NNNode" << getId();
    return rawstr.str();
}

// ±ßµÄ¾ßÌå
const std::string Direct2NeuronEdge::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << "NNEdge: [NNNode" << getDstID() << " <-- NNNode" << getSrcID() << "]\t";
    return rawstr.str();
}



//NeuronNet::~NeuronNet()
//{}


NeuronEdge* NeuronNet::hasNeuronEdge(SVF::NeuronNode* src, SVF::NeuronNode* dst, NeuronEdge::NeuronEdgeK kind)
{
    NeuronEdge edge(src, dst, kind);
    NeuronEdge* outEdge = src->hasOutgoingEdge(&edge);
    NeuronEdge* inEdge = dst->hasIncomingEdge(&edge);
    if (outEdge && inEdge)
    {
        assert(outEdge == inEdge && "edges not match");
        return outEdge;
    }
    else
        return nullptr;
}

NeuronEdge* NeuronNet::getNeuronEdge(const SVF::NeuronNode* src, const SVF::NeuronNode* dst, NeuronEdge::NeuronEdgeK kind){
    NeuronEdge* edge = nullptr;
    u32_t counter = 0;
    for(NeuronEdge::NeuronGraphEdgeSetTy::iterator iter = src->OutEdgeBegin();
         iter != src->OutEdgeEnd(); ++iter){
        if ((*iter)->getDstID() == dst->getId() && (*iter)->getEdgeKind() == kind)
        {
            counter++;
            edge = (*iter);
        }
    }
    assert(counter <= 1 && "there's more than one edge between two Neuron nodes");
    return edge;
}

const std::string NeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << "NNNode" << getId();
    return rawstr.str();
}

void NeuronNet::dump(const std::string& file, bool simple)
{
    GraphPrinter::WriteGraphToFile(SVFUtil::outs(), file, this, simple);
}

void NeuronNet::view()
{
    SVF::ViewGraph(this, "SVF NeuronNet Graph");
}
//
//namespace SVF
//{
//template <> struct DOTGraphTraits<NeuronNet*> : public DOTGraphTraits<SVFIR*>
//{
//    typedef ICFGNode NodeType;
//    DOTGraphTraits(bool isSimple = false) : DOTGraphTraits<SVFIR*>(isSimple) {}
//
//    /// Get the Graph's name
//    static std::string getGraphName(NeuronNet*)
//    {
//        return "Neuronnet Graph";
//    }
//
//    static std::string getSimpleNodeLabel(NodeType* node, NeuronNet*)
//    {
//        return node->toString();
//    }
//
//    std::string getNodeLabel(NodeType* node, NeuronNet* graph)
//    {
//        return getSimpleNodeLabel(node, graph);
//    }
//
//    static std::string getNodeAttributes(NodeType* node, ICFG*)
//    {
//        std::string str;
//        std::stringstream rawstr(str);
//
//        if (SVFUtil::isa<ReLuNeuronNode>(node))
//        {
//            rawstr << "color=black";
//        }
//        else if (SVFUtil::isa<BasicOPNeuronNode>(node))
//        {
//            rawstr << "color=yellow";
//        }
//        else if (SVFUtil::isa<FullyConNeuronNode>(node))
//        {
//            rawstr << "color=green";
//        }
//        else if (SVFUtil::isa<ConvNeuronNode>(node))
//        {
//            rawstr << "color=red";
//        }
//        else if (SVFUtil::isa<MaxPoolNeuronNode>(node))
//        {
//            rawstr << "color=blue";
//        }
//        else if (SVFUtil::isa<ConstantNeuronNode>(node))
//        {
//            rawstr << "color=purple";
//        }
//        else
//            assert(false && "no such kind of node!!");
//
//        rawstr << "";
//
//        return rawstr.str();
//    }
//
//    template <class EdgeIter>
//    static std::string getEdgeAttributes(NodeType*, EdgeIter EI, NeuronNet*)
//    {
//        NeuronEdge* edge = *(EI.getCurrent());
//        assert(edge && "No edge found!!");
//        if (SVFUtil::isa<Direct2NeuronEdge>(edge))
//            return "style=solid,color=red";
//        else
//            return "style=solid";
//        return "";
//    }
//
//    template <class EdgeIter>
//    static std::string getEdgeSourceLabel(NodeType*, EdgeIter EI)
//    {
//        NeuronEdge* edge = *(EI.getCurrent());
//        assert(edge && "No edge found!!");
//
//        std::string str;
//        std::stringstream rawstr(str);
//        if (Direct2NeuronEdge* dirCall =
//                SVFUtil::dyn_cast<Direct2NeuronEdge>(edge))
//            rawstr << dirCall->getCallSite();
//        return rawstr.str();
//    };
//
//}
//}
