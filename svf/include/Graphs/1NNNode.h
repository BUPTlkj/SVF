#ifndef SVF_NNNODE_H
#define SVF_NNNODE_H


#include "GenericGraph.h"
#include "1NNEdge.h"
#include <Eigen/Dense>

namespace SVF{

class NeuronNode;
class ReLuNeuronNode;
class BasicOPNeuronNode;
class MaxPoolNeuronNode;
class FullyConNeuronNode;
class ConvNeuronNode;
class ConstantNeuronNode;

using NeuronNodeVariant =
    std::variant<std::monostate, SVF::ConstantNeuronNode*,
                 SVF::BasicOPNeuronNode*, SVF::FullyConNeuronNode*,
                 SVF::ConvNeuronNode*, SVF::ReLuNeuronNode*,
                 SVF::MaxPoolNeuronNode*>;

typedef GenericNode<NeuronNode, NeuronEdge> GenericNeuronNodeTy;
class NeuronNode: public GenericNeuronNodeTy{

public:
    /// ReLu, Add Sub Mul Div, Maxpooling(Flatten), Conv, FullyCon, Constant
    enum NodeK{
        ReLuNode,     //0
        BasicOPNode,  //1
        MaxPoolNode,  //2
        ConvNode,     //3
        FullyConNode, //4
        ConstantNode, //5
        Sub,          //6
        Add,          //7
        Mul,          //8
        Div           //9
    };

    typedef NeuronEdge::NeuronGraphEdgeSetTy::iterator iterator; // 边 点  可以修改
    typedef NeuronEdge::NeuronGraphEdgeSetTy::const_iterator const_iterator; //边 点，不可以修改
    typedef std::list<const NeuronNode*> NeuronNodeList;
    typedef std::list<const NeuronEdge*> NeuronEdgeList;

public:
    /// Constructor
    /// Depends
    NeuronNode(NodeID i, NodeK k):
          GenericNeuronNodeTy(i, k){
        NeuronEdges = NeuronEdgeList();
        NeuronNodes = NeuronNodeList();
        std::cout<<"NNNode is initing NodeID: "<<i<<" Type: "<<k<<std::endl;
    }

    /// Rebuild << store NodeID in NNgraph
    friend OutStream &operator<<(OutStream &o, const NeuronNode &node)
    {
        o << node.toString();
        return o;
    }

    /// Add a node
    inline void addNeoronNode(const NeuronNode *nnNode){
        std::cout<<"addNeoronNode Fun is activated, NodeID: "<<nnNode->getId()<<" Type: "<<nnNode->getNodeKind()<<std::endl;
        NeuronNodes.push_back(nnNode);
    }

    /// Get all nodes
    inline const NeuronNodeList& getNeuronNodes() const{
        return NeuronNodes;
    }

    inline void addNeuronEdge(const NeuronEdge *nnEdge){
        NeuronEdges.push_back(nnEdge);
    }

    inline const NeuronEdgeList& getNeuronEdges() const{
        return NeuronEdges;
    }

    virtual const std::string toString() const;

    void dump() const;

public:
    /// Get layer‘s type(node)
    virtual NodeK get_type() const = 0;

protected:
    NeuronEdgeList NeuronEdges;
    NeuronNodeList NeuronNodes;

};

/// For conv
class FilterSubNode{
public:
    std::vector<Eigen::MatrixXd> value;
    FilterSubNode(const std::vector<Eigen::MatrixXd>& x){
        for (unsigned i = 0; i < x.size(); i++) {
            for (unsigned j = 0; j < x.size(); j++) {
                if (x[i].rows() != x[j].rows() || x[i].cols() != x[j].cols()) {
                    throw std::runtime_error("Bad construction of Filter");
                }
            }
        }
        value = x;
    }

    unsigned get_depth() const;
    unsigned get_height() const;
    unsigned get_width() const;

    double dot_product(const FilterSubNode& val_f) const;

};

class ReLuNeuronNode:public NeuronNode{

public:
    /// Build ReLu Node
    ReLuNeuronNode(NodeID id):
          NeuronNode(id, ReLuNode){}

    static inline bool classof(const ReLuNeuronNode *)
    {
        return true;
    }

    static inline bool classof(const NeuronNode *node)
    {
        return node->getNodeKind() == ReLuNode;
    }

    static inline bool classof(const GenericNeuronNodeTy *node)
    {
        return node->getNodeKind() == ReLuNode;
    }

public:
    /// Key in ReLu
    NodeK get_type() const override;

    const std::string toString() const override;

};

class BasicOPNeuronNode:public NeuronNode{
public:
    std::vector<Eigen::MatrixXd> constant;
    std::string oper;

    /// Build Add/Sub/Mul/Div node
    BasicOPNeuronNode(NodeID id, const std::string op, const std::vector<Eigen::MatrixXd>& w):
          NeuronNode(id, BasicOPNode), constant{w}, oper{op}{};

    static inline bool classof(const BasicOPNeuronNode *)
    {
        return true;
    }

    static inline bool classof(const NeuronNode *node)
    {
        return node->getNodeKind() == BasicOPNode;
    }

    static inline bool classof(const GenericNeuronNodeTy *node)
    {
        return node->getNodeKind() == BasicOPNode;
    }
public:
    inline std::vector<Eigen::MatrixXd> get_constant() const;
    inline std::string get_oper() const;

public:
    NodeK get_type() const override;

    const std::string toString() const override;
};

class MaxPoolNeuronNode:public NeuronNode{
public:
    /// Define the windows size
    unsigned window_width;
    unsigned window_height;
    unsigned stride_width;
    unsigned stride_height;
    unsigned pad_width;
    unsigned pad_height;

    /// Bulid a maxpooling node
    MaxPoolNeuronNode(NodeID id, unsigned ww, unsigned wh, unsigned sw, unsigned sh, unsigned pw, unsigned ph):
          NeuronNode(id, MaxPoolNode), window_width{ww}, window_height{wh},
          stride_width(sw), stride_height(sh),
          pad_width(pw), pad_height(ph){

    }

    static inline bool classof(const MaxPoolNeuronNode *)
    {
        return true;
    }

    static inline bool classof(const NeuronNode *node)
    {
        return node->getNodeKind() == MaxPoolNode;
    }

    static inline bool classof(const GenericNeuronNodeTy *node)
    {
        return node->getNodeKind() == MaxPoolNode;
    }

public:
    inline unsigned get_window_width() const;
    inline unsigned get_window_height() const;
    inline unsigned get_stride_width() const;
    inline unsigned get_stride_height() const;
    inline unsigned get_pad_width() const;
    inline unsigned get_pad_height() const;

public:
    NodeK get_type() const override;

    const std::string toString() const override;
};


class FullyConNeuronNode:public NeuronNode{
public:
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
//    FullyConNeuronNode(NodeID id);
    /// the most common
    FullyConNeuronNode(NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b):
          NeuronNode(id, FullyConNode), weight{w}, bias{b}{
    }
    /// For Conv's filter op
    FullyConNeuronNode(): NeuronNode(-1,FullyConNode){};

    static inline bool classof(const FullyConNeuronNode *)
    {
        return true;
    }

    static inline bool classof(const NeuronNode *node)
    {
        return node->getNodeKind() == FullyConNode;
    }

    static inline bool classof(const GenericNeuronNodeTy *node)
    {
        return node->getNodeKind() == FullyConNode;
    }

public:
    inline Eigen::MatrixXd get_weight() const;
    inline Eigen::VectorXd get_bias() const;


public:
    NodeK get_type() const override;

    const std::string toString() const override;
};


class ConvNeuronNode:public NeuronNode{
public:
    /// filter
    unsigned filter_depth;
    unsigned filter_width;
    unsigned filter_height;

    /// filter_num
    unsigned filter_num;

    /// filter
    std::vector<FilterSubNode> filter;

    /// bias
    std::vector<double> bias;

    unsigned padding;
    unsigned stride;
    // u32t
    // s32t

    ConvNeuronNode(NodeID id, const std::vector<FilterSubNode>& fil, const std::vector<double> b, unsigned pad, unsigned str):
          NeuronNode(id, ConstantNode),
          filter_depth{ fil[0].get_depth() }, filter_width{ fil[0].get_width() },filter_height{ fil[0].get_height() }, filter_num(fil.size()), filter(fil),
          bias{b}, padding(pad), stride(str){
        std::cout<<filter_num<<"   "<<filter_depth<<"   "<<filter_width<<"    "<<filter_height<<std::endl;
    }

    static inline bool classof(const ConvNeuronNode *)
    {
        return true;
    }

    static inline bool classof(const NeuronNode *node)
    {
        return node->getNodeKind() == ConvNode;
    }

    static inline bool classof(const GenericNeuronNodeTy *node)
    {
        return node->getNodeKind() == ConvNode;
    }

public:
    inline unsigned get_filter_depth() const{
        return filter_depth;
    };
    inline unsigned get_filter_width() const{
        return filter_width;
    };
    inline unsigned get_filter_height() const{
        return filter_height;
    };
    inline unsigned get_filter_num() const{
        return filter_num;
    };

    inline std::vector<FilterSubNode> get_filter() const{
        return filter;
    };

    inline std::vector<double> get_bias() const{
        return bias;
    };

    inline unsigned get_padding() const{
        return padding;
    };
    inline unsigned get_stride() const{
        return stride;
    };

public:
    NodeK get_type() const override;

    const std::string toString() const override;
};

/// ConstantNeuronNode: Nothing needs to be done, just propagates the input without op.
class ConstantNeuronNode:public NeuronNode{

public:
    ConstantNeuronNode(NodeID id):
          NeuronNode(id, ConstantNode){}

    static inline bool classof(const ConstantNeuronNode *)
    {
        return true;
    }

    static inline bool classof(const NeuronNode *node)
    {
        return node->getNodeKind() == ConstantNode;
    }

    static inline bool classof(const GenericNeuronNodeTy *node)
    {
        return node->getNodeKind() == ConstantNode;
    }

public:
    /// Key in this type
    NodeK get_type() const override;

    const std::string toString() const override;
};

}
#endif // SVF_NNNODE_H
