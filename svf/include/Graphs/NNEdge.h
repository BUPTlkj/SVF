//
// Created by ������ on 2024/2/12.
//

#ifndef SVF_NNEDGE_H
#define SVF_NNEDGE_H

#include "GenericGraph.h"


namespace SVF
{

class NeuronNode; // ��
class Direct2NeuronEdge; // �����

//GenericEdge<NeuronNode>�ǵ���GenericGraph.h
typedef GenericEdge<NeuronNode> GenericNeuronEdgeTy;
class NeuronEdge : public GenericNeuronEdgeTy {
public:
    enum NeuronEdgeK{
        Direct2Neuron //��Ԫ֮�������ߣ����FFN��CNN
    };

    typedef NeuronEdgeK NeuronGraphEdge;

public:
    //���캯��
    NeuronEdge(NeuronNode* s, NeuronNode* d, GEdgeFlag k) : GenericNeuronEdgeTy(s, d, k) {
    }

    //��������
    ~NeuronEdge(){}

    // �ж���Ԫ֮��������
    inline bool is_neuronedge() const {
        return getEdgeKind() == Direct2Neuron;
    }

    //�жϱߵ������Ƿ��������, �ڵ�ǰ��Ŀ�У���is_neuronedge��������һ��
    inline bool isDirectEdge() const{
        return getEdgeKind()== Direct2Neuron;
    }

    // ��߼���
    typedef GenericNode<NeuronNode, NeuronEdge>::GEdgeSetTy NeuronGraphEdgeSetTy;

    // ���ַ��� ���뵽�����
    friend OutStream& operator<<(OutStream& o, const NeuronEdge& edge)
    {
        o << edge.toString();
        return o;
    }

    virtual const std::string toString() const;

};

// �� Direct2Neuron��
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
