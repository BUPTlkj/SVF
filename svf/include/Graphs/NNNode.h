#ifndef SVF_NNNODE_H
#define SVF_NNNODE_H


#include "GenericGraph.h"
#include "NNEdge.h"
#include <Eigen/Dense>

namespace SVF{

class NeuronNode;
class ReLuNeuronNode;
class BasicOPNeuronNode;
class MaxPoolNeuronNode;
class FullyConNeuronNode;
class ConvNeuronNode;
class ConstantNeuronNode;


typedef GenericNode<NeuronNode, NeuronEdge> GenericNeuronNodeTy;
class NeuronNode: public GenericNeuronNodeTy{

public:
    /// ReLu, Add Sub Mul Div, Maxpooling(Flatten), Conv, FullyCon, Constant
    enum NodeK{
        ReLuNode,
        BasicOPNode,
        MaxPoolNode,
        ConvNode,
        FullyConNode,
        ConstantNode,
        Sub,
        Add,
        Mul,
        Div
    };

    typedef NeuronEdge::NeuronGraphEdgeSetTy::iterator iterator; // 边 点  可以修改
    typedef NeuronEdge::NeuronGraphEdgeSetTy::const_iterator const_iterator; //边 点，不可以修改
    typedef std::list<const NeuronNode*> NeuronNodeList;
    typedef std::list<const NeuronEdge*> NeuronEdgeList;

public:
    /// Constructor
    /// Depends
    NeuronNode(NodeID i, NodeK k, unsigned iw, unsigned ih, unsigned id, unsigned ow, unsigned oh, unsigned od):
          GenericNeuronNodeTy(i, k),
          in_width{ iw }, in_height{ ih }, in_depth{ id }, out_width{ ow },
          out_height{ oh }, out_depth{ od }{
        NeuronEdges = NeuronEdgeList();
        NeuronNodes = NeuronNodeList();

    }

    /// Rebuild << store NodeID in NNgraph
    friend OutStream &operator<<(OutStream &o, const NeuronNode &node)
    {
        o << node.toString();
        return o;
    }

    /// Add a node
    inline void addNeoronNode(const NeuronNode *nnNode){
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
    inline unsigned get_in_width() const {
        return in_width;
    }
    inline unsigned get_in_height() const {
        return in_height;
    }
    inline unsigned get_in_depth() const {
        return in_depth;
    }

    inline unsigned get_out_width() const {
        return out_width;
    }
    inline unsigned get_out_height() const {
        return out_height;
    }
    inline unsigned get_out_depth() const {
        return out_depth;
    }

    /// Get layer‘s type(node)
    virtual NodeK get_type() const = 0;

    /// Forward propagation
    virtual std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>& x) const = 0;

    virtual std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const = 0;


protected:
    NeuronEdgeList NeuronEdges;
    NeuronNodeList NeuronNodes;

    /// Node Info
protected:
    unsigned in_width;
    unsigned in_height;
    unsigned in_depth;

    unsigned out_width;
    unsigned out_height;
    unsigned out_depth;
};

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
    ReLuNeuronNode(NodeID id, unsigned in_w, unsigned in_h, unsigned in_d):
          NeuronNode(id, ReLuNode, in_w, in_h, in_d, in_w, in_h, in_d){}

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
    NodeK get_type() const override;

    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;

    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;

    const std::string toString() const override;

};

class BasicOPNeuronNode:public NeuronNode{
public:
    std::vector<Eigen::MatrixXd> constant;
    std::string oper;

    /// Build Add/Sub/Mul/Div node
    BasicOPNeuronNode(NodeID id, const std::string op, const std::vector<Eigen::MatrixXd>& w, unsigned in_w, unsigned in_h, unsigned in_d):
          NeuronNode(id, BasicOPNode, in_w, in_h, in_d, in_w, in_h, in_d), constant{w}, oper{op}{};

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
    NodeK get_type() const override;

    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;

    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;

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
    MaxPoolNeuronNode(NodeID id, unsigned ww, unsigned wh, unsigned sw, unsigned sh, unsigned pw, unsigned ph, unsigned in_w, unsigned in_h, unsigned in_d):
          NeuronNode(id, MaxPoolNode,in_w, in_h, in_d, in_w, in_h, in_d), window_width{ww}, window_height{wh},
          stride_width(sw), stride_height(sh),
          pad_width(pw), pad_height(ph){
        /// Calculate the relations between windows and input tensor
        if (((in_w + 2*pw - ww) % sw != 0) || ((in_h + 2*ph - wh) % sh != 0)) {
            throw std::runtime_error("Input dimensions and strides do not match");
        }
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
    NodeK get_type() const override;

    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;

    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;

    const std::string toString() const override;
};


class FullyConNeuronNode:public NeuronNode{
public:
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
//    FullyConNeuronNode(NodeID id);
    /// the most common
    FullyConNeuronNode(NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d):
          NeuronNode(id, FullyConNode, in_w, in_h, in_d, 1, b.size(), 1), weight{w}, bias{b}{
        if (w.rows() != b.size()) {
            throw std::runtime_error("Bad initialization of FCLayer");
        }
    }
    /// For Conv's filter op
    FullyConNeuronNode(): NeuronNode(-1,FullyConNode, 0,0,0,0,0,0){};
//    FullyConNeuronNode(const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d);
    FullyConNeuronNode(NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b):
          NeuronNode(id, FullyConNode, 1, w.cols(), 1, 1, b.size(), 1), weight{ w }, bias{ b }{
        if (w.rows() != b.size()) {
            throw std::runtime_error("Bad initialization of FCLayer");
        }
    }

    FullyConNeuronNode(NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d, unsigned out_w, unsigned out_h, unsigned out_d):
          NeuronNode(id, FullyConNode, in_w, in_h, in_d, out_w, out_h, out_d), weight{ w }, bias{ b } {
        if (w.rows() != b.size()) {
            throw std::runtime_error("Bad initialization of FCLayer");
        }
    }


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
    NodeK get_type() const override;

    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;

    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;

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

    FullyConNeuronNode fullyer;

    unsigned padding;
    unsigned stride;
    // u32t
    // s32t

    ConvNeuronNode(NodeID id, const std::vector<FilterSubNode>& fil, const std::vector<double> b, unsigned in_w, unsigned in_h, unsigned pad, unsigned str):
          NeuronNode(id, ConstantNode, in_w, in_h, fil[0].get_depth(),
                     ((in_w - fil[0].get_width() + 2*pad) / str) + 1,
                     ((in_h - fil[0].get_height() + 2*pad) / str) + 1,
                     fil.size()),
          filter_depth{ fil[0].get_depth() }, filter_width{ fil[0].get_width() },filter_height{ fil[0].get_height() }, filter_num(fil.size()), filter(fil),
          bias{b}, padding(pad), stride(str){
        /// to do
        unsigned ful_con_cols = in_depth * in_height * in_width;
        unsigned ful_con_rows = out_depth * out_height * out_width;

        /// Build
        Eigen::MatrixXd ful_weight = Eigen::MatrixXd::Zero(ful_con_rows, ful_con_cols);

        /// bias
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
    NodeK get_type() const override;

    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;

    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;

    const std::string toString() const override;
};


/// ConstantNeuronNode: Nothing needs to be done, just propagates the input without op.
class ConstantNeuronNode:public NeuronNode{

public:
    ConstantNeuronNode(NodeID id, unsigned in_w, unsigned in_h, unsigned in_d):
          NeuronNode(id, ConstantNode, in_w, in_h, in_d, in_w, in_h, in_d){}

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
    NodeK get_type() const override;

    std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>&) const override;

    std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const override;

    const std::string toString() const override;
};

}

#endif // SVF_NNNODE_H
