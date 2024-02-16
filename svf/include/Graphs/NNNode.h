//
// Created by Kaijie Liu on 2024/2/9.
//

#ifndef SVF_NNNODE_H
#define SVF_NNNODE_H


#include "GenericGraph.h"
#include "NNEdge.h"
#include <Eigen/Dense>

namespace SVF{

typedef GenericNode<NeuronNode, NeuronEdge> GenericNeuronNodeTy;
class NeuronNode: public GenericNeuronNodeTy{

public:
    // 点的种类, ReLu, Add Sub Mul Div, Maxpooling(Flatten), Conv, FullyCon, Constant
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
    // 构造函数
    // 后续需要看是否还要添加其他
    NeuronNode(NodeID i, NodeK k, unsigned iw, unsigned ih, unsigned id, unsigned ow, unsigned oh, unsigned od):
          GenericNeuronNodeTy(i, k),
          in_width{ iw }, in_height{ ih }, in_depth{ id }, out_width{ ow },
          out_height{ oh }, out_depth{ od }{
    }

    //重构<<存储图的点的ID
    friend OutStream &operator<<(OutStream &o, const NeuronNode &node)
    {
        o << node.toString();
        return o;
    }

    //添加点的方法
    inline void addNeoronNode(const NeuronNode *nnNode){
        NeuronNodes.push_back(nnNode);
    }

    //获得所有点
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

    // 获得层的类型
    virtual NodeK get_type() const = 0;

    // 正向传播
    virtual std::vector<Eigen::MatrixXd> evaluate(
        const std::vector<Eigen::MatrixXd>& x) const = 0;

    virtual std::vector<Eigen::MatrixXd> backpropagate(
        const std::vector<Eigen::MatrixXd>&,
        const std::vector<Eigen::MatrixXd>&) const = 0;


protected:
    NeuronEdgeList NeuronEdges;
    NeuronNodeList NeuronNodes;

    //节点信息
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
    FilterSubNode(const std::vector<Eigen::MatrixXd>& x);

    unsigned get_depth() const;
    unsigned get_height() const;
    unsigned get_width() const;

    double dot_product(const FilterSubNode& val_f) const;

};

class ReLuNeuronNode:public NeuronNode{

public:
    // 构建一个ReLu节点
    ReLuNeuronNode(NodeID id, unsigned in_w, unsigned in_h, unsigned in_d);

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

    // 构建一个Add/Sub/Mul/Div节点
    BasicOPNeuronNode(NodeID id, const std::string op, const std::vector<Eigen::MatrixXd>& w, unsigned in_w, unsigned in_h, unsigned in_d);

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
    // 定义窗口的size
    unsigned window_width;
    unsigned window_height;

    // 构建一个最大池化节点
    MaxPoolNeuronNode(NodeID id, unsigned ww, unsigned wh, unsigned in_w, unsigned in_h, unsigned in_d);

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
    // 常见
    FullyConNeuronNode(NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d);
    // 以下两个是为了卷积的操作
    FullyConNeuronNode();
//    FullyConNeuronNode(const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d);
    FullyConNeuronNode(NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b);
    FullyConNeuronNode(NodeID id, const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d, unsigned out_w, unsigned out_h, unsigned out_d);

    static inline bool classof(const MaxPoolNeuronNode *)
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
    //filter的维度
    unsigned filter_depth;
    unsigned filter_width;
    unsigned filter_height;

    //filter的个数
    unsigned filter_num;

    // filter
    std::vector<FilterSubNode> filter;

    // 偏置项
    std::vector<double> bias;

    FullyConNeuronNode fullyer;

    ConvNeuronNode(NodeID id, const std::vector<FilterSubNode>& fil, const std::vector<double> b, unsigned in_w, unsigned in_h);

    static inline bool classof(const MaxPoolNeuronNode *)
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


// ONNX 中这个节点不需要做操作
class ConstantNeuronNode:public NeuronNode{

public:
    ConstantNeuronNode(NodeID id, unsigned in_w, unsigned in_h, unsigned in_d);

    static inline bool classof(const ReLuNeuronNode *)
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
