//
// Created by Kaijie Liu on 2024/2/11.
//

#ifndef SVF_NNGRAPH_H
#define SVF_NNGRAPH_H

#include "NNEdge.h"
#include "NNNode.h"
#include <memory> // ��������ָ��

// ʹ��SVF�����ռ��е�ͼ�ṹ
namespace SVF {

typedef GenericGraph<NeuronNode, NeuronEdge> GenericNeuronNetTy;
class NeuronNet : public GenericNeuronNetTy {
public:
    // id�����͵�ӳ��
    typedef OrderedMap<NodeID, NeuronNode *> NeuronNetNodeID2NodeMapTy;
    typedef NeuronEdge::NeuronGraphEdgeSetTy NeuronGraphEdgeSetTy;
    typedef NeuronNetNodeID2NodeMapTy::iterator iterator;
    typedef NeuronNetNodeID2NodeMapTy::const_iterator const_iterator;

private:
    NodeID totalNeuronNatNode;

public:
    //���캯��
    NeuronNet();

    //��������
    ~NeuronNet() override;

    // ���һ��NeuronNet��Node
    inline NeuronNode* getNeuronNetNode(NodeID id) const{
        return getGNode(id);
    }

    // �ж��Ƿ����һ����
    inline bool hasNeuronNetNode(NodeID id) const{
        return hasGNode(id);
    }

    // �ж��Ƿ����һ���ߣ�����src��dst
    NeuronEdge* hasNeuronEdge(NeuronNode* src, NeuronNode* dst, NeuronEdge::NeuronEdgeK kind);

    // ��ñ߸���src��dst
    NeuronEdge* getNeuronEdge(const NeuronNode* src, const NeuronNode* dst, NeuronEdge::NeuronEdgeK kind);

    /// Dump graph into dot file
    void dump(const std::string& file, bool simple = false);

    /// View graph from the debugger
    void view();

protected:
    // ɾ��һ���� ���������reducing �����õ�����

    //ɾ��һ���� ����Ҫ

    // �����
    inline bool addNeuronEdge(NeuronEdge* edge){
        //�����ָ��ĵ����ߣ� Ҳ���������
        bool added1 = edge->getDstNode()->addIncomingEdge(edge);
        // ������ߵ�Դ��node
        bool added2 = edge->getSrcNode()->addOutgoingEdge(edge);
        bool all_added = added1 && added2;
        assert(all_added && "NeuronEdge not added?");
        return all_added;
    }

    //�����
    virtual inline void addNeuronNode(NeuronNode* node){
        addGNode(node->getId(), node);
    }

private:
    // ReLu��
    inline ReLuNeuronNode* addReLuNeuronNode( const unsigned in_w, const unsigned in_h, const unsigned in_d){
        ReLuNeuronNode* sNode = new ReLuNeuronNode(totalNeuronNatNode++, in_w, in_h, in_d);
        //NodeID addedNodeID = totalNeuronNatNode;
        addNeuronNode(sNode);
        return sNode;
    }
    // BasicOPNeuronNode(NodeID id, const std::string op, const std::vector<Eigen::MatrixXd>& w, unsigned in_w, unsigned in_h, unsigned in_d)
    // ���Basic Op��
    inline BasicOPNeuronNode* addBasicOPNeuronNode(const std::string op, const std::vector<Eigen::MatrixXd>& w, unsigned in_w, unsigned in_h, unsigned in_d){
        BasicOPNeuronNode* sNode = new BasicOPNeuronNode(totalNeuronNatNode++, op, w, in_w, in_h, in_d);
        addNeuronNode(sNode);
        return sNode;
    }

    // MaxPool��
    inline MaxPoolNeuronNode* addMaxPoolNeuronNode(unsigned ww, unsigned wh, unsigned in_w, unsigned in_h, unsigned in_d){
        MaxPoolNeuronNode* sNode = new MaxPoolNeuronNode(totalNeuronNatNode++, ww, wh, in_w, in_h, in_d);
        addNeuronNode(sNode);
        return sNode;
    }

    // FullyCon��
    inline FullyConNeuronNode* addFullyConNeuronNode(const Eigen::MatrixXd& w, const Eigen::VectorXd& b, unsigned in_w, unsigned in_h, unsigned in_d){
        FullyConNeuronNode* sNode = new FullyConNeuronNode(totalNeuronNatNode++, w, b, in_w, in_h, in_d);
        addNeuronNode(sNode);
        return sNode;
    }

    // ConvNeuronNode��
    inline ConvNeuronNode* addConvNeuronNode(const std::vector<FilterSubNode>& fil, const std::vector<double> b, unsigned in_w, unsigned in_h){
        ConvNeuronNode* sNode = new ConvNeuronNode(totalNeuronNatNode++, fil, b, in_w, in_h);
        addNeuronNode(sNode);
        return sNode;
    }

    //Constant�� -- ���������ʼ��
    inline ConstantNeuronNode* addConstantNeuronNode(unsigned iw, unsigned ih, unsigned id){
        ConstantNeuronNode* sNode = new ConstantNeuronNode(totalNeuronNatNode++, iw, ih, id);
        addNeuronNode(sNode);
        return sNode;
    }

};

} // End namespace SVF


namespace SVF{
/* !
 * GenericGraphTraits specializations for generic graph algorithms.
 * Provide graph traits for traversing from a constraint node using standard graph traversals.
 */
template<> struct GenericGraphTraits<SVF::NeuronNode*> : public GenericGraphTraits<SVF::GenericNode<SVF::NeuronNode,SVF::NeuronEdge>*  >
{
};

/// Inverse GenericGraphTraits specializations for call graph node, it is used for inverse traversal.
//template<>
//struct GenericGraphTraits<Inverse<SVF::NeuronNode *> > : public GenericGraphTraits<Inverse<SVF::GenericNode<SVF::NeuronNode,SVF::NeuronEdge>* > >
//{
//};

template<> struct GenericGraphTraits<SVF::NeuronNet*> : public GenericGraphTraits<SVF::GenericGraph<SVF::NeuronNode,SVF::NeuronEdge>* >
{
    typedef SVF::NeuronNode *NodeRef;
};






} // End namespace SVF

#endif // SVF_NNGRAPH_H
