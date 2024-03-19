#ifndef SVF_NNNODE_H
#define SVF_NNNODE_H


#include "GenericGraph.h"
#include "NNEdge.h"
#include <Eigen/Dense>
#include "AE/Core/IntervalValue.h"

namespace SVF{

typedef std::vector<Eigen::MatrixXd> Matrices;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vector;
typedef Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic> IntervalMat;
typedef std::vector<IntervalMat> IntervalMatrices;
typedef Eigen::Matrix<IntervalValue, Eigen::Dynamic, 1> IntervalVector;


class NeuronNode;
class ReLuNeuronNode;
class BasicOPNeuronNode;
class MaxPoolNeuronNode;
class FullyConNeuronNode;
class ConvNeuronNode;
class ConstantNeuronNode;

using NeuronNodeVariant =
    std::variant<std::monostate, ConstantNeuronNode*,
                 BasicOPNeuronNode*, FullyConNeuronNode*,
                 ConvNeuronNode*, ReLuNeuronNode*,
                 MaxPoolNeuronNode*>;

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

    typedef NeuronEdge::NeuronGraphEdgeSetTy::iterator iterator;
    typedef NeuronEdge::NeuronGraphEdgeSetTy::const_iterator const_iterator;
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
    /// Get layer¡®s type(node)
    virtual NodeK get_type() const = 0;

protected:
    NeuronEdgeList NeuronEdges;
    NeuronNodeList NeuronNodes;

};

/// For conv
class FilterSubNode{
public:
    Matrices value;
    IntervalMatrices Intervalvalue;
    FilterSubNode(const Matrices& x, const IntervalMatrices intermat){
        for (u32_t i = 0; i < x.size(); i++) {
            for (u32_t j = 0; j < x.size(); j++) {
                if (x[i].rows() != x[j].rows() || x[i].cols() != x[j].cols()) {
                    throw std::runtime_error("Bad construction of Filter");
                }
            }
        }
        value = x;
        Intervalvalue = intermat;
    }

    /// Interval & Mat
    u32_t get_depth() const;
    u32_t get_height() const;
    u32_t get_width() const;

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
    Matrices constant;
    IntervalMatrices Intervalconstant;
    std::string oper;

    /// Build Add/Sub/Mul/Div node
    BasicOPNeuronNode(NodeID id, const std::string op, const Matrices& w, const IntervalMatrices ic):
          NeuronNode(id, BasicOPNode), constant{w}, Intervalconstant{ic}, oper{op} {};

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
    inline Matrices get_constant() const;
    inline std::string get_oper() const;

public:
    NodeK get_type() const override;

    const std::string toString() const override;
};

class MaxPoolNeuronNode:public NeuronNode{
public:
    /// Define the windows size
    u32_t window_width;
    u32_t window_height;
    u32_t stride_width;
    u32_t stride_height;
    u32_t pad_width;
    u32_t pad_height;

    /// Bulid a maxpooling node
    MaxPoolNeuronNode(NodeID id, u32_t ww, u32_t wh, u32_t sw, u32_t sh, u32_t pw, u32_t ph):
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
    inline u32_t get_window_width() const;
    inline u32_t get_window_height() const;
    inline u32_t get_stride_width() const;
    inline u32_t get_stride_height() const;
    inline u32_t get_pad_width() const;
    inline u32_t get_pad_height() const;

public:
    NodeK get_type() const override;

    const std::string toString() const override;
};


class FullyConNeuronNode:public NeuronNode{
public:
    Mat weight;
    Vector bias;

    IntervalMat Intervalweight;
    IntervalMat Intervalbias;

    /// the most common
    FullyConNeuronNode(NodeID id, const Mat& w, const Vector& b, const IntervalMat iw, const IntervalMat ib):
          NeuronNode(id, FullyConNode), weight{w}, bias{b}, Intervalweight(iw), Intervalbias(ib){
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
    inline Mat get_weight() const;
    inline Vector get_bias() const;


public:
    NodeK get_type() const override;

    const std::string toString() const override;
};


class ConvNeuronNode:public NeuronNode{
public:
    /// filter
    u32_t filter_depth;
    u32_t filter_width;
    u32_t filter_height;

    /// filter_num
    u32_t filter_num;

    /// filter
    std::vector<FilterSubNode> filter;

    /// bias
    std::vector<double> bias;
    std::vector<IntervalValue> Intervalbias;

    u32_t padding;
    u32_t stride;
    // u32t
    // s32t

    ConvNeuronNode(NodeID id, const std::vector<FilterSubNode>& fil, const std::vector<double> b, u32_t pad, u32_t str, const std::vector<IntervalValue> idouble):
          NeuronNode(id, ConstantNode),
          filter_depth{ fil[0].get_depth() }, filter_width{ fil[0].get_width() },filter_height{ fil[0].get_height() }, filter_num(fil.size()), filter(fil),
          bias{b}, Intervalbias(idouble), padding(pad), stride(str){
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
    inline u32_t get_filter_depth() const{
        return filter_depth;
    };
    inline u32_t get_filter_width() const{
        return filter_width;
    };
    inline u32_t get_filter_height() const{
        return filter_height;
    };
    inline u32_t get_filter_num() const{
        return filter_num;
    };

    inline std::vector<FilterSubNode> get_filter() const{
        return filter;
    };

    inline std::vector<double> get_bias() const{
        return bias;
    };

    inline u32_t get_padding() const{
        return padding;
    };
    inline u32_t get_stride() const{
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
