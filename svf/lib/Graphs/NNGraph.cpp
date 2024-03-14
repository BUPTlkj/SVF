#include "../svf-onnx/CheckModels.h"
#include "../svf-onnx/Solver.h"
#include "Graphs/NNGraph.h"
#include "SVFIR/SVFIR.h"
#include "Util/SVFUtil.h"
#include "iomanip"

using namespace SVF;

/// ReLu

NeuronNode::NodeK ReLuNeuronNode::get_type() const{
    return ReLuNode;
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



/// Maxpooling

NeuronNode::NodeK MaxPoolNeuronNode::get_type() const{
    return MaxPoolNode;
}

unsigned int MaxPoolNeuronNode::get_window_width() const{
    return window_height;
}

unsigned int MaxPoolNeuronNode::get_window_height() const{
    return window_width;
}

unsigned int MaxPoolNeuronNode::get_stride_width() const{
    return stride_width;
}

unsigned int MaxPoolNeuronNode::get_stride_height() const{
    return stride_height;
}

unsigned int MaxPoolNeuronNode::get_pad_width() const{
    return pad_width;
}

unsigned int MaxPoolNeuronNode::get_pad_height() const{
    return pad_height;
}


/// FullyCon

NeuronNode::NodeK FullyConNeuronNode::get_type() const{
    return FullyConNode;
}

Eigen::MatrixXd FullyConNeuronNode::get_weight() const{
    return weight;
}

Eigen::VectorXd FullyConNeuronNode::get_bias() const{
    return bias;
}


/// Conv
NeuronNode::NodeK ConvNeuronNode::get_type() const{
    return ConvNode;
}

//unsigned int ConvNeuronNode::get_filter_depth() const{
//    return filter_depth;
//}


//unsigned int ConvNeuronNode::get_filter_width() const{
//    return filter_width;
//}
//
//unsigned int ConvNeuronNode::get_filter_height() const{
//    return filter_height;
//}
//
//unsigned int ConvNeuronNode::get_filter_num() const{
//    return filter_num;
//}
//
//std::vector<FilterSubNode> ConvNeuronNode::get_filter() const{
//    return filter;
//}
//
//std::vector<double> ConvNeuronNode::get_bias() const{
//    return bias;
//}
//
//unsigned int ConvNeuronNode::get_padding() const{
//    return padding;
//}
//
//unsigned int ConvNeuronNode::get_stride() const{
//    return stride;
//}

/// Constant
NeuronNode::NodeK ConstantNeuronNode::get_type() const{
    return ConstantNode;
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

/// Edge
const std::string Direct2NeuronEdge::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << "NNEdge: [NNNode" << getDstID() << " <-- NNNode" << getSrcID() << "]\t";
    return rawstr.str();
}

const std::string NeuronEdge::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << "NNEdge: [NNNode" << getDstID() << " <-- NNNode" << getSrcID() << "]\t";
    return rawstr.str();
}

NeuronNet::~NeuronNet()
{}

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


namespace SVF
{
template <> struct DOTGraphTraits<NeuronNet*> : public DOTGraphTraits<SVFIR*>
{
    typedef NeuronNode NodeType;
    DOTGraphTraits(bool isSimple = false) : DOTGraphTraits<SVFIR*>(isSimple) {}

    /// Get the Graph's name
    static std::string getGraphName(NeuronNet*)
    {
        return "Neuronnet Graph";
    }

    static std::string getSimpleNodeLabel(NodeType* node, NeuronNet*)
    {
        return node->toString();
    }

    std::string getNodeLabel(NodeType* node, NeuronNet* graph)
    {
        return getSimpleNodeLabel(node, graph);
    }

    static std::string getNodeAttributes(NodeType* node, NeuronNet*)
    {
        std::string str;
        std::stringstream rawstr(str);

        if (SVFUtil::isa<ReLuNeuronNode>(node))
        {
            rawstr << "color=black";
        }
        else if (SVFUtil::isa<BasicOPNeuronNode>(node))
        {
            rawstr << "color=yellow";
        }
        else if (SVFUtil::isa<FullyConNeuronNode>(node))
        {
            rawstr << "color=green";
        }
        else if (SVFUtil::isa<ConvNeuronNode>(node))
        {
            rawstr << "color=red";
        }
        else if (SVFUtil::isa<MaxPoolNeuronNode>(node))
        {
            rawstr << "color=blue";
        }
        else if (SVFUtil::isa<ConstantNeuronNode>(node))
        {
            rawstr << "color=purple";
        }
        else
            assert(false && "no such kind of node!!");

        rawstr << "";

        return rawstr.str();
    }

    template <class EdgeIter>
    static std::string getEdgeAttributes(NodeType*, EdgeIter EI, NeuronNet*)
    {
        NeuronEdge* edge = *(EI.getCurrent());
        assert(edge && "No edge found!!");
        if (SVFUtil::isa<Direct2NeuronEdge>(edge))
            return "style=solid,color=red";
        else
            return "style=solid";
        return "";
    }

    template <class EdgeIter>
    static std::string getEdgeSourceLabel(NodeType*, EdgeIter EI)
    {
        NeuronEdge* edge = *(EI.getCurrent());
        assert(edge && "No edge found!!");

        std::string str;
        std::stringstream rawstr(str);

        rawstr << "";
        return rawstr.str();
    };

};
}

/// By Edge odo
void GraphTraversalE::printPathE(std::vector<const SVF::Direct2NeuronEdge *> &path1){
    std::string pathStr = "START: ";
    pathStr += std::to_string(path1.front()->getDstNode()->getId()) + "->";
    for (size_t i = 1; i < path1.size(); ++i) // Started from the first not zero.
    {
        pathStr += std::to_string(path1[i]->getDstNode()->getId()) + "->";
    }
    pathStr += "END";
    paths.insert(pathStr);
    std::cout << pathStr << std::endl;

}



void GraphTraversalE::DFSE(const SVF::Direct2NeuronEdge *src_edge, const SVF::NeuronNodeVariant *dst, std::vector<Eigen::MatrixXd> in_x){


//     Mark the source node as visited
//    auto *src = src_edge->getSrcNode();



}





/// BY NODE todo

void GraphTraversal::printPath(std::vector<const SVF::NeuronNode *> &path){
    std::string output = "START: ";
    for (size_t i = 0; i < path.size(); ++i) {
        output += std::to_string(path[i]->getId());
        //            output += std::to_string(path[i]->getNodeID());
        if (i < path.size() - 1) {
            output += "->";
        }
    }
//    for (const auto& nodeVariant : path) {
//        std::visit([&output](auto&& arg) {
//            using T = std::decay_t<decltype(arg)>;
//            if constexpr (!std::is_same_v<T, std::monostate>) {
//                output += std::to_string(arg->getId());
//            }
//        }, *nodeVariant);
//        output += "->";
//    }
//    output += "->END";
    paths.insert(output);
};

SVF::NeuronNodeVariant GraphTraversal::convertToVariant(SVF::NeuronNode* node) {
    // 检查node的具体类型并相应地构造NeuronNodeVariant
    if (auto* constantNode = SVFUtil::dyn_cast<SVF::ConstantNeuronNode>(node)) {
        return constantNode;
    } else if (auto* basicOPNode = SVFUtil::dyn_cast<SVF::BasicOPNeuronNode>(node)) {
        return basicOPNode;
    } else if (auto* fullyConNode = SVFUtil::dyn_cast<SVF::FullyConNeuronNode>(node)) {
        return fullyConNode;
    } else if (auto* convNode = SVFUtil::dyn_cast<SVF::ConvNeuronNode>(node)) {
        return convNode;
    } else if (auto* reLuNode = SVFUtil::dyn_cast<SVF::ReLuNeuronNode>(node)) {
        return reLuNode;
    } else if (auto* maxPoolNode = SVFUtil::dyn_cast<SVF::MaxPoolNeuronNode>(node)) {
        return maxPoolNode;
    }
    // 如果node不匹配任何已知类型，可以返回std::monostate或抛出异常
    return std::monostate{};
}

//void GraphTraversal::DFS(std::set<const SVF::NeuronNodeVariant *> &visited, std::vector<const SVF::NeuronNodeVariant *> &path, const SVF::NeuronNodeVariant *src, const SVF::NeuronNodeVariant *dst, std::vector<Eigen::MatrixXd> in_x) {
//    visited.insert(src); // 标记当前节点为已访问
//    path.push_back(src); // 将当前节点加入路径
//
//    if (src == dst) {
//        printPath(path); // 打印路径
//    }
//
//    std::vector<Eigen::MatrixXd> IRRes;
//    SolverEvaluate solver(in_x);
//    std::visit([&](auto&& arg) {
//    using T = std::decay_t<decltype(arg)>;
//    if constexpr (!std::is_same_v<T, std::monostate>) {
//        // 遍历当前节点的所有出边
//        for (const auto& edge : arg->getOutEdges()) {
//            auto *neighbor = edge->getDstNode(); // 获取邻居节点
//
//            // 需要将neighbor转换为SVF::NeuronNodeVariant
//            SVF::NeuronNodeVariant variantNeighbor = convertToVariant(neighbor); // 完成转换
//
//                if (visited.count(&variantNeighbor) == 0) { // 如果邻居未被访问
//                    if (neighbor->get_type() == 0) {
//                        IRRes = solver.ReLuNeuronNodeevaluate(IRRes);
//                    } else if (neighbor->get_type() == 1 || neighbor->get_type() == 6 || neighbor->get_type() == 7 || neighbor->get_type() == 8 || neighbor->get_type() == 9) {
//                        const SVF::BasicOPNeuronNode *node = SVFUtil::dyn_cast<SVF::BasicOPNeuronNode>(neighbor);
//                        IRRes = solver.BasicOPNeuronNodeevaluate(IRRes, node);
//                    } else if (neighbor->get_type() == 2) {
//                        SVF::MaxPoolNeuronNode *node = SVFUtil::dyn_cast<SVF::MaxPoolNeuronNode>(neighbor);
//                        IRRes = solver.MaxPoolNeuronNodeevaluate(IRRes, node);
//                    } else if (neighbor->get_type() == 3) {
//                        SVF::ConvNeuronNode *node = SVFUtil::dyn_cast<SVF::ConvNeuronNode>(neighbor);
//                        IRRes = solver.ConvNeuronNodeevaluate(IRRes, node);
//                    } else if (neighbor->get_type() == 4) {
//                        SVF::FullyConNeuronNode *node = SVFUtil::dyn_cast<SVF::FullyConNeuronNode>(neighbor);
//                        IRRes = solver.FullyConNeuronNodeevaluate(IRRes, node);
//                    } else if (neighbor->get_type() == 5) {
//                        IRRes = solver.ConstantNeuronNodeevaluate(IRRes);
//                    }
//
//                    // 递归调用DFS
//                    DFS(visited, path, src, dst, IRRes);
//                }
//            }
//        }
//    }, *src);
//
//    visited.erase(src); // 在回溯时从已访问集合中移除当前节点
//    path.pop_back(); // 从路径中移除当前节点
//}
// 检查 current 是否在 dst variant 中

bool GraphTraversal::checkNodeInVariant(const SVF::NeuronNode* current, const NeuronNodeVariant& dst) {
    return std::visit([current](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            return current == static_cast<const SVF::NeuronNode*>(arg);
        } else {
            return false;
        }
    }, dst);
}

SVF::NeuronNode* GraphTraversal::getNeuronNodePtrFromVariant(const NeuronNodeVariant& variant) {
    return std::visit([](auto&& arg) -> SVF::NeuronNode* {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return nullptr; // 对应于variant未持有任何NeuronNode指针的情况
        } else {
            return arg; // 直接返回NeuronNode指针
        }
    }, variant);
}


/// todo 3.13
/// matrix into interval
void GraphTraversal::DFS(std::set<const SVF::NeuronNode *> &visited, std::vector<const SVF::NeuronNode *> &path, const SVF::NeuronNodeVariant *src, const SVF::NeuronNodeVariant *dst, std::vector<Eigen::MatrixXd> in_x) {
    std::stack<std::pair<SVF::NeuronNode*, std::vector<Eigen::MatrixXd>>> stack;

    stack.emplace(getNeuronNodePtrFromVariant(*src), in_x); // 确保 src 是通过解引用得到 SVF::NeuronNode

    SolverEvaluate solver(in_x);
    int i = 0;
    std::vector<Eigen::MatrixXd> IRRes;

    while (!stack.empty()) {
        std::cout<<" Node: "<<i<<std::endl;
        i++;
        auto currentPair = stack.top();
        stack.pop();

        const SVF::NeuronNode* current = currentPair.first;
        IRRes = currentPair.second;

        std::cout<<&current<<std::endl<<" STACK SIZE: "<<stack.size()<<std::endl;
        std::cout<<" The size of Matrix: "<<IRRes.size()<<std::endl;

        if (!visited.insert(current).second) { // 如果节点已经访问过，跳过
            std::cout<<"CONTINUE: This Node "<<&current<<" has already been visited!"<<std::endl;
            continue;
        }

        path.push_back(current); // 加入路径

        if (checkNodeInVariant(current, *dst)) {
            printPath(path);
        }


        for (const auto& edge : current->getOutEdges()) {
            auto *neighbor = edge->getDstNode();

            SVF::NeuronNodeVariant variantNeighbor = convertToVariant(neighbor);
            std::cout<<"SrcNode: "<<current<<" ->  DSTNode: "<<neighbor<<" -> "<<&variantNeighbor<<std::endl;

            std::cout<<"NodeTYpe:  "<<neighbor->get_type()<<std::endl;

            if (visited.count(neighbor) == 0) {
                // 根据邻居节点的类型处理 IRRes
                std::vector<Eigen::MatrixXd> newIRRes; // 复制当前 IRRes 以避免修改原始数据
                if (neighbor->get_type() == 0) {
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ReLuNeuronNodeevaluate();
                    std::cout<<"FINISH RELU"<<std::endl;
                } else if (neighbor->get_type() == 1 || neighbor->get_type() == 6 || neighbor->get_type() == 7 || neighbor->get_type() == 8 || neighbor->get_type() == 9) {
                    const SVF::BasicOPNeuronNode *node = SVFUtil::dyn_cast<SVF::BasicOPNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.BasicOPNeuronNodeevaluate(node);
                    std::cout<<"FINISH BAIC"<<std::endl;
                } else if (neighbor->get_type() == 2) {
                    const SVF::MaxPoolNeuronNode *node = SVFUtil::dyn_cast<SVF::MaxPoolNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.MaxPoolNeuronNodeevaluate(node);
                    std::cout<<"FINISH MAXPOOLING"<<std::endl;
                } else if (neighbor->get_type() == 3) {
                    const SVF::ConvNeuronNode *node = static_cast<SVF::ConvNeuronNode *>(neighbor);
                    std::cout<<"TEST  ID  COnv"<<neighbor->getId()<<std::endl;
                    std::cout<<"TEST  ID  COnv"<<node->getId()<<std::endl;
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ConvNeuronNodeevaluate(node);
                    std::cout<<"FINISH Conv"<<std::endl;
                } else if (neighbor->get_type() == 4) {
                    const SVF::FullyConNeuronNode *node = SVFUtil::dyn_cast<SVF::FullyConNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    std::cout<<"*******************"<<std::endl;
                    newIRRes = solver.FullyConNeuronNodeevaluate(node);
                    std::cout<<"FINISH FullyConnected"<<std::endl;
                } else if (neighbor->get_type() == 5) {
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ConstantNeuronNodeevaluate();
                    std::cout<<"FINISH Constant"<<std::endl;
                }
                // 将邻居节点和新的 IRRes 压入栈中
//                    TraverseVectorOfMatrices(IRRes);
                IRRes = newIRRes;
                stack.emplace(neighbor, newIRRes);
                std::cout<<"FINISH PUSHING STACK! "<<stack.size()<<std::endl<<std::endl<<std::endl;
            }
        }
        std::cout << "IRRes content after the loop iteration:" << i << std::endl;
        std::cout.precision(12);
        std::cout << std::fixed;
        for (size_t j = 0; j < IRRes.size(); ++j) {
            std::cout << "Matrix " << j << ":\n";
            std::cout << "Rows: " << IRRes[j].rows() << ", Columns: " << IRRes[j].cols() << "\n";
            std::cout << IRRes[j] << "\n\n"; // 打印矩阵的维度和内容
        }
        visited.erase(current); // 处理
        path.pop_back(); // 从路径中移除当前节点
    }
}