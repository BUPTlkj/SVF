//
// Created by Kaijie Liu on 2024/2/9.
//
// 图的具体实现

#include "Graphs/NNGraph.h"
#include "Graphs/NNNode.h"
#include "Graphs/NNEdge.h"
#include "Util/SVFUtil.h"
#include "SVFIR/SVFIR.h"

using namespace SVF;

// ReLu
ReLuNeuronNode::ReLuNeuronNode(SVF::NodeID id, unsigned int in_w, unsigned int in_h, unsigned int in_d):
      NeuronNode(id, ReLuNode, in_w, in_h, in_d, in_w, in_h, in_d){}

NeuronNode::NodeK ReLuNeuronNode::get_type() const{
    return ReLuNode;
}

std::vector<Eigen::MatrixXd> NeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& x_in) const{
    std::vector<Eigen::MatrixXd> x_out;
    for (Eigen::MatrixXd mat : x_in) {
        /* rows表示行数 cols表示列数
            * 构建一个列表
         */
        Eigen::MatrixXd o(mat.rows(), mat.cols());
        for (unsigned i = 0; i < mat.rows(); i++) {
            for (unsigned j = 0; j < mat.cols(); j++) {
                o(i, j) = std::max(0.0, mat(i, j));
            }
        }
        // 通道
        x_out.push_back(o);
    }
    return x_out;
}

std::vector<Eigen::MatrixXd> NeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& in_x, const std::vector<Eigen::MatrixXd>& grad) const{
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

// filter
FilterSubNode::FilterSubNode(const std::vector<Eigen::MatrixXd>& x) {
    for (unsigned i = 0; i < x.size(); i++) {
        for (unsigned j = 0; j < x.size(); j++) {
            if (x[i].rows() != x[j].rows() || x[i].cols() != x[j].cols()) {
                throw std::runtime_error("Bad construction of Filter");
            }
        }
    }
    value = x;
}

unsigned int FilterSubNode::get_depth() const{
    return value.size();
}

unsigned int FilterSubNode::get_height() const{
    if (value.size() > 0) {
        //第一通道
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
    //校验窗口大小
    if (val_f.get_depth() != get_depth()) {
        throw std::runtime_error("Dimension-d mismatch in Filter.dot_product");
    }
    if (val_f.get_height() != get_height()) {
        throw std::runtime_error("Dimension-h mismatch in Filter.dot_product");
    }
    if (val_f.get_width() != get_width()) {
        throw std::runtime_error("Dimension-w mismatch in Filter.dot_product");
    }

    //点积运算
    for (unsigned i = 0; i < get_depth(); i++) {
        for (unsigned j = 0; j < get_height(); j++) {
            for (unsigned k = 0; k < get_width(); k++) {
                sum += val_f.value[i](j, k) * value[i](j, k);
            }
        }
    }
    return sum;
}

//BasicOPNeuronNode 包括Sub, Mul, Div, Add
BasicOPNeuronNode::BasicOPNeuronNode(SVF::NodeID id, const std::string op, const std::vector<Eigen::MatrixXd>& w, unsigned int in_w, unsigned int in_h, unsigned int in_d):
      NeuronNode(id, BasicOPNode, in_w, in_h, in_d, in_w, in_h, in_d), constant{w}, oper{op}{};

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

    // 确保A和B有相同数量的通道
    if (in_x.size() != constant.size()) {
        std::cerr << "Error: The number of channels must be the same." << std::endl;
        return result;
    }

    for (size_t i = 0; i < in_x.size(); ++i) {
        // 确保每个通道的尺寸正确
        if (constant[i].rows() != 1 || constant[i].cols() != 1) {
            std::cerr << "Error: B's channel matrices must be 1x1." << std::endl;
            return result;
        }

        // 创建一个与A的通道相同尺寸的矩阵，其所有元素都是B的相应通道值
        Eigen::MatrixXd temp = Eigen::MatrixXd::Constant(in_x[i].rows(), in_x[i].cols(), constant[i](0,0));

        // 将A的通道与上述矩阵相减
        if(oper == "+"){
            result.push_back(in_x[i] + temp);
        }else if(oper == "-"){
            result.push_back(in_x[i] - temp);
        }else if(oper == "*"){
            result.push_back(in_x[i].cwiseProduct(temp));
        }else if(oper == "/"){
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
    //扁平化操作,列向量
    Eigen::VectorXd grad_x(out_height * out_width * out_depth);
    for (unsigned i = 0; i < out_depth; i++) {
        for (unsigned j = 0; j < out_height; j++) {
            for (unsigned k = 0; k < out_width; k++) {
                grad_x(out_width * out_depth * j + out_depth * k + i) = grad[i](j, k);
            }
        }
    }
    //梯度，这里要根据情况修改 todo
    Eigen::VectorXd out_col = grad_x.transpose();
    // 重构输出
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

//maxpool
MaxPoolNeuronNode::MaxPoolNeuronNode(SVF::NodeID id, unsigned int ww, unsigned int wh, unsigned int in_w, unsigned int in_h, unsigned int in_d):
      NeuronNode(id, MaxPoolNode,in_w, in_h, in_d, in_w, in_h, in_d), window_width{ww}, window_height{wh}{
    //计算窗口与输入tensor的关系
    if (in_w % ww != 0 || in_h % wh != 0) {
        throw std::runtime_error("Bad initialization of MaxPoolLayer");
    }
}

NeuronNode::NodeK MaxPoolNeuronNode::get_type() const{
    return MaxPoolNode;
}

std::vector<Eigen::MatrixXd> MaxPoolNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& in_x) const{
    std::vector<Eigen::MatrixXd> out_x;
    double doub_max = -std::numeric_limits<double>::max();
    //chanel
    for (unsigned i = 0; i < in_depth; i++) {
        //tensor/window
        out_x.push_back(Eigen::MatrixXd(in_height / window_height, in_width / window_width));
        for (unsigned j = 0; j < in_height / window_height; j++) {
            for (unsigned k = 0; k < in_width / window_width; k++) {
                // 找出下限
                double max = doub_max;
                for (unsigned j_ = 0; j_ < window_height; j_++) {
                    for (unsigned k_ = 0; k_ < window_width; k_++) {
                        double current_val = in_x[i](window_height* j + j_, window_width* k + k_);
                        if (current_val > max) {
                            max = current_val;
                        }
                    }
                }
                out_x[i](j, k) = max;
            }
        }
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

// FullyCon全连接节点
FullyConNeuronNode::FullyConNeuronNode(SVF::NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned int in_w, unsigned int in_h, unsigned int in_d):
      NeuronNode(id, FullyConNode, in_w, in_h, in_d, 1, b.size(), 1), weight{w}, bias{b}{
    if (w.rows() != b.size()) {
        throw std::runtime_error("Bad initialization of FCLayer");
    }
}

FullyConNeuronNode::FullyConNeuronNode(SVF::NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b):
      NeuronNode(id, FullyConNode, 1, w.cols(), 1, 1, b.size(), 1), weight{ w }, bias{ b }{
    if (w.rows() != b.size()) {
        throw std::runtime_error("Bad initialization of FCLayer");
    }
}

FullyConNeuronNode::FullyConNeuronNode(SVF::NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d, unsigned out_w, unsigned out_h, unsigned out_d):
      NeuronNode(id, FullyConNode, in_w, in_h, in_d, out_w, out_h, out_d), weight{ w }, bias{ b } {
    if (w.rows() != b.size()) {
        throw std::runtime_error("Bad initialization of FCLayer");
    }
}

NeuronNode::NodeK FullyConNeuronNode::get_type() const{
    return FullyConNode;
}

std::vector<Eigen::MatrixXd> FullyConNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& in_x) const{
    //处理输入扁平化操作 这一步相当于ONNX中的GEMM节点操作
    Eigen::VectorXd x_ser(in_depth * in_height * in_width);
    for (unsigned i = 0; i < in_depth; i++) {
        for (unsigned j = 0; j < in_height; j++) {
            for (unsigned k = 0; k < in_width; k++) {
                x_ser(in_width * in_depth * j + in_depth * k + i) = in_x[i](j, k);
            }
        }
    }

    //wx+b
    Eigen::VectorXd val = weight * x_ser + bias;

    // 还原输出
    std::vector<Eigen::MatrixXd> out;

    // 赋值
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
    //扁平化操作,列向量
    Eigen::VectorXd grad_x(out_height * out_width * out_depth);
    for (unsigned i = 0; i < out_depth; i++) {
        for (unsigned j = 0; j < out_height; j++) {
            for (unsigned k = 0; k < out_width; k++) {
                grad_x(out_width * out_depth * j + out_depth * k + i) = grad[i](j, k);
            }
        }
    }

    Eigen::VectorXd out_col = weight * grad_x.transpose();
    // 重构输出
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

ConvNeuronNode::ConvNeuronNode(SVF::NodeID id, const std::vector<FilterSubNode>& fil, const std::vector<double> b, unsigned int in_w, unsigned int in_h):
      NeuronNode(id, ConstantNode, in_w, in_h, fil[0].get_depth(), in_w - fil[0].get_width() + 1, in_h - fil[0].get_height() + 1, fil.size()),
      filter_depth{ fil[0].get_depth() }, filter_width{ fil[0].get_width() },filter_height{ fil[0].get_height() }, filter_num(fil.size()), filter(fil),
      bias{b}{
    // to do
    unsigned ful_con_cols = in_depth * in_height * in_width;
    unsigned ful_con_rows = out_depth * out_height * out_width;

    // 构建一个
    Eigen::MatrixXd ful_weight = Eigen::MatrixXd::Zero(ful_con_rows, ful_con_cols);

    //偏置项
    Eigen::VectorXd ful_con_bias(ful_con_rows);

    for (unsigned i = 0; i < in_height - filter_height + 1; i++) {
        for (unsigned j = 0; j < in_width - filter_width; j++) {
            for (unsigned k = 0; k < filter_num; k++) {
                unsigned row = (in_width - filter_width + 1) * filter_num * i + filter_num * j + k;
                for (unsigned i_ = 0; i_ < filter_height; i_++) {
                    for (unsigned j_ = 0; j_ < filter_width; j_++) {
                        for (unsigned k_ = 0; k_ < filter_depth; k_++) {
                            unsigned col = in_width * in_depth * (i + i_) + in_depth * (j + j_) + k_;
                            ful_weight(row, col) = filter[k].value[k_](i_,j_);
                        }
                    }
                }
                ful_con_bias(row) = bias[k];
            }
        }
    }
    fullyer = FullyConNeuronNode(id, ful_weight, ful_con_bias, in_width, in_height, in_depth, out_width, out_height, out_depth);
}

NeuronNode::NodeK ConvNeuronNode::get_type() const{
    return ConvNode;
}

std::vector<Eigen::MatrixXd> ConvNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& x) const{
    std::vector<Eigen::MatrixXd> out;
    // tensor
    for (unsigned i = 0; i < filter_num; i++) {
        out.push_back(Eigen::MatrixXd(out_height, out_width));
        for (unsigned j = 0; j < out_width; j++) {
            for (unsigned k = 0; k < out_height; k++) {
                // filter_window内的赋值
                std::vector<Eigen::MatrixXd> win;
                for (unsigned i_ = 0; i_ < filter_depth; i_++) {
                    win.push_back(Eigen::MatrixXd(filter_height, filter_width));
                    for (unsigned j_ = 0; j_ < filter_width; j_++) {
                        for (unsigned k_ = 0; k_ < filter_height; k_++) {
                            win[i_](k_, j_) = x[i_](k_ + k, j_ + j);
                        }
                    }
                }
                out[i](k, i) = filter[i].dot_product(win) + bias[k];

            }
        }
    }
    return out;
}

std::vector<Eigen::MatrixXd> ConvNeuronNode::backpropagate(const std::vector<Eigen::MatrixXd>& in_x, const std::vector<Eigen::MatrixXd>& grad) const{
    return fullyer.backpropagate(in_x, grad);
}

ConstantNeuronNode::ConstantNeuronNode(SVF::NodeID id, unsigned in_w, unsigned in_h, unsigned in_d ):
      NeuronNode(id, ConstantNode, in_w, in_h, in_d, in_w, in_h, in_d){}

NeuronNode::NodeK ConstantNeuronNode::get_type() const{
    return ConstantNode;
}

std::vector<Eigen::MatrixXd> ConstantNeuronNode::evaluate(const std::vector<Eigen::MatrixXd>& x) const{
    // This is an entry, nothing needs to do.
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

// ques
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

// 边的具体
const std::string Direct2NeuronEdge::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << "NNEdge: [NNNode" << getDstID() << " <-- NNNode" << getSrcID() << "]\t";
    return rawstr.str();
}

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
//    // 返回图的名字
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
