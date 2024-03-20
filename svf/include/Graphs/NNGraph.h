#ifndef SVF_NNGRAPH_H
#define SVF_NNGRAPH_H

#include "AE/Nnexe/NNgraphIntervalSolver.h"
#include "NNEdge.h"
#include "NNNode.h"
#include <memory>

/// using SVF namespace graph strcture
namespace SVF {

typedef GenericGraph<NeuronNode, NeuronEdge> GenericNeuronNetTy;
class NeuronNet : public GenericNeuronNetTy {

public:
    /// id to type
    typedef OrderedMap<NodeID, NeuronNode *> NeuronNetNodeID2NodeMapTy;
    typedef NeuronEdge::NeuronGraphEdgeSetTy NeuronGraphEdgeSetTy;
    typedef NeuronNetNodeID2NodeMapTy::iterator iterator;
    typedef NeuronNetNodeID2NodeMapTy::const_iterator const_iterator;

    /// Not Use currently
    NodeID totalNeuronNatNode;

public:
    /// Constructor
    NeuronNet(): totalNeuronNatNode(0) {}

    /// Destructor
    ~NeuronNet() override;

    /// get a NNGrap node
    inline NeuronNode* getNeuronNetNode(NodeID id) const{
        return getGNode(id);
    }

    /// whether it has a NNNode
    inline bool hasNeuronNetNode(NodeID id) const{
        return hasGNode(id);
    }

    /// whether it has a NNEdge
    NeuronEdge* hasNeuronEdge(NeuronNode* src, NeuronNode* dst, NeuronEdge::NeuronEdgeK kind);

    /// Get a NNGraph edge according to src and dst
    NeuronEdge* getNeuronEdge(const NeuronNode* src, const NeuronNode* dst, NeuronEdge::NeuronEdgeK kind);

    /// Dump graph into dot file
    void dump(const std::string& file, bool simple = false);

    /// View graph from the debugger
    void view();

//protected:
public:
    /// Remove an NNEdge, maybe will be used later.
    /// Remove a NNNode, maybe will be used later.

    // Add NNEdge for Node
//    NeuronEdge*  addNNEdge(NeuronNode* srcNode, NeuronNode* dstNode);


    /// Add NNEdge
    inline bool addNeuronEdge(NeuronEdge* edge){
        bool added1 = edge->getDstNode()->addIncomingEdge(edge);
        bool added2 = edge->getSrcNode()->addOutgoingEdge(edge);
        bool all_added = added1 && added2;
        assert(all_added && "NeuronEdge not added?");
        return all_added;
    }

    /// Add NNNode
    virtual inline void addNeuronNode(NeuronNode* node){
        addGNode(node->getId(), node);
    }

public:
    /// Add ReLu Node
    inline ReLuNeuronNode* addReLuNeuronNode(ReLuNeuronNode* sNode){
        addNeuronNode(sNode);
        return sNode;
    }

    /// Add Flatten Node
    inline FlattenNeuronNode* addFlattenNeuronNode(FlattenNeuronNode* sNode){
        addNeuronNode(sNode);
        return sNode;
    }

    /// get ReLu Node, Not using Currently
//    inline ReLuNeuronNode* getaddReLuNeuronNode(const NeuronNode* node){
//        const_iterator it =
//    }


    /// BasicOp layer
    inline BasicOPNeuronNode* addBasicOPNeuronNode(BasicOPNeuronNode* sNode){
        addNeuronNode(sNode);
        return sNode;
    }

    /// MaxPool layer
    inline MaxPoolNeuronNode* addMaxPoolNeuronNode(MaxPoolNeuronNode* sNode){
        addNeuronNode(sNode);
        return sNode;
    }

    /// FullyCon layer
    inline FullyConNeuronNode* addFullyConNeuronNode(FullyConNeuronNode* sNode){
        addNeuronNode(sNode);
        return sNode;
    }

    /// ConvNeuronNode layer
    inline ConvNeuronNode* addConvNeuronNode(ConvNeuronNode* sNode){
        addNeuronNode(sNode);
        return sNode;
    }

    /// Constant: Nothing to do OR Input layer
    inline ConstantNeuronNode* addConstantNeuronNode(ConstantNeuronNode* sNode){
        addNeuronNode(sNode);
        return sNode;
    }

    /// Add NNEdge
    static inline bool addDirected2NodeEdge(Direct2NeuronEdge* edge){
        bool added1 = edge->getDstNode()->addIncomingEdge(edge);
        bool added2 = edge->getSrcNode()->addOutgoingEdge(edge);
        bool all_added = added1 && added2;
        assert(all_added && "NeuronEdge not added?");
        return all_added;
    }

};

} /// End namespace SVF


namespace SVF{
/* !
 * GenericGraphTraits specializations for generic graph algorithms.
 * Provide graph traits for traversing from a constraint node using standard graph traversals.
 */
template<> struct GenericGraphTraits<NeuronNode*> : public GenericGraphTraits<GenericNode<NeuronNode,NeuronEdge>*  >
{
};

/// Inverse GenericGraphTraits specializations for call graph node, it is used for inverse traversal.
//template<>
//struct GenericGraphTraits<Inverse<NeuronNode *> > : public GenericGraphTraits<Inverse<GenericNode<NeuronNode,NeuronEdge>* > >
//{
//};

template<> struct GenericGraphTraits<NeuronNet*> : public GenericGraphTraits<GenericGraph<NeuronNode, NeuronEdge>* >
{
    typedef NeuronNode *NodeRef;
};

} // End namespace SVF

#endif // SVF_NNGRAPH_H
