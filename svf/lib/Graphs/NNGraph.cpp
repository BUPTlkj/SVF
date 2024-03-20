#include "Graphs/NNGraph.h"
#include "SVFIR/SVFIR.h"
#include "Util/SVFUtil.h"
#include "iomanip"

using namespace SVF;

/// ReLu

NeuronNode::NodeK ReLuNeuronNode::get_type() const{
    return ReLuNode;
}

NeuronNode::NodeK FlattenNeuronNode::get_type() const{
    return FlattenNode;
}

/// filter
u32_t FilterSubNode::get_depth() const{
    return value.size();
}

u32_t FilterSubNode::get_height() const{
    if (value.size() > 0) {
        /// The first channel
        return value[0].rows();
    }
    else {
        return 0;
    }
}

u32_t FilterSubNode::get_width() const{
    if (value.size() > 0) {
        return value[0].cols();
    }
    else {
        return 0;
    }
}

double FilterSubNode::dot_product(const FilterSubNode& val_f) const{
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
    for (u32_t i = 0; i < get_depth(); i++) {
        for (u32_t j = 0; j < get_height(); j++) {
            for (u32_t k = 0; k < get_width(); k++) {
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

u32_t MaxPoolNeuronNode::get_window_width() const{
    return window_height;
}

u32_t MaxPoolNeuronNode::get_window_height() const{
    return window_width;
}

u32_t MaxPoolNeuronNode::get_stride_width() const{
    return stride_width;
}

u32_t MaxPoolNeuronNode::get_stride_height() const{
    return stride_height;
}

u32_t MaxPoolNeuronNode::get_pad_width() const{
    return pad_width;
}

u32_t MaxPoolNeuronNode::get_pad_height() const{
    return pad_height;
}


/// FullyCon

NeuronNode::NodeK FullyConNeuronNode::get_type() const{
    return FullyConNode;
}

Mat FullyConNeuronNode::get_weight() const{
    return weight;
}

Vector FullyConNeuronNode::get_bias() const{
    return bias;
}


/// Conv
NeuronNode::NodeK ConvNeuronNode::get_type() const{
    return ConvNode;
}

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
    rawstr <<ReLuNeuronNode::get_type() << "ReluNode" << getId();
    return rawstr.str();
}

const std::string FlattenNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr <<FlattenNeuronNode::get_type() << "FlattenNode" << getId();
    return rawstr.str();
}

const std::string BasicOPNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << BasicOPNeuronNode::get_type() <<BasicOPNeuronNode::get_type() << "BasicNode" << getId();
    return rawstr.str();
}

const std::string MaxPoolNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << MaxPoolNeuronNode::get_type() << "MaxPoolingNode" << getId();
    return rawstr.str();
}

const std::string FullyConNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << FullyConNeuronNode::get_type() << "FullyConnectedNode" << getId();
    return rawstr.str();
}

const std::string ConvNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << ConvNeuronNode::get_type() << "ConvNode" << getId();
    return rawstr.str();
}

const std::string ConstantNeuronNode::toString() const{
    std::string str;
    std::stringstream rawstr(str);
    rawstr << ConstantNeuronNode::get_type() << "ConstantNode" << getId();
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

NeuronEdge* NeuronNet::hasNeuronEdge(NeuronNode* src, NeuronNode* dst, NeuronEdge::NeuronEdgeK kind)
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

NeuronEdge* NeuronNet::getNeuronEdge(const NeuronNode* src, const NeuronNode* dst, NeuronEdge::NeuronEdgeK kind){
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
    ViewGraph(this, "SVF NeuronNet Graph");
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
        }else if (SVFUtil::isa<FlattenNeuronNode>(node))
        {
            rawstr << "color=gray";
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
