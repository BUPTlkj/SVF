//
// Created by 刘凯杰 on 2024/2/12.
//

#ifndef SVF_NNEDGE_H
#define SVF_NNEDGE_H

#include "GenericGraph.h"


namespace SVF
{

class NeuronNode; // 点
class Direct2NeuronEdge; // 有向边

//GenericEdge<NeuronNode>是调用GenericGraph.h
typedef GenericEdge<NeuronNode> GenericNeuronEdgeTy;
class NeuronEdge : public GenericNeuronEdgeTy {
public:
    enum NeuronEdgeK{
        Direct2Neuron //神经元之间的有向边，针对FFN，CNN
    };

    typedef NeuronEdgeK NeuronGraphEdge;

public:
    //构造函数
    NeuronEdge(NeuronNode* s, NeuronNode* d, GEdgeFlag k) : GenericNeuronEdgeTy(s, d, k) {
    }

    //析构函数
    ~NeuronEdge(){}

    // 判断神经元之间的有向边
    inline bool is_neuronedge() const {
        return getEdgeKind() == Direct2Neuron;
    }

    //判断边的类型是否是有向边, 在当前项目中，与is_neuronedge函数作用一致
    inline bool isDirectEdge() const{
        return getEdgeKind()== Direct2Neuron;
    }

    // 点边集合
    typedef GenericNode<NeuronNode, NeuronEdge>::GEdgeSetTy NeuronGraphEdgeSetTy;

    // 边字符串 插入到输出流
    friend OutStream& operator<<(OutStream& o, const NeuronEdge& edge)
    {
        o << edge.toString();
        return o;
    }

    virtual const std::string toString() const;

};

// 边 Direct2Neuron边
class Direct2NeuronEdge:public NeuronEdge{

private:
    //

public:
    Direct2NeuronEdge(NeuronNode* s, NeuronNode* d): NeuronEdge(s, d, Direct2Neuron) {
    }

    static inline bool classof(const Direct2NeuronEdge*)
    {
        return true;
    }

    static inline bool classof(const NeuronEdge *edge)
    {
        return edge->getEdgeKind() == Direct2Neuron;
    }

    static inline bool classof(const GenericNeuronEdgeTy *edge)
    {
        return edge->getEdgeKind() == Direct2Neuron;
    }

    virtual const std::string toString() const;
};


}

#endif // SVF_NNEDGE_H
