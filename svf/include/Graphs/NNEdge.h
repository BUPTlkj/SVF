#ifndef SVF_NNEDGE_H
#define SVF_NNEDGE_H

#include "GenericGraph.h"


namespace SVF
{

class NeuronNode; ///  Node-Layer
class Direct2NeuronEdge; ///  Directed edge
/// GenericEdge<NeuronNode> in GenericGraph.h
typedef GenericEdge<NeuronNode> GenericNeuronEdgeTy;
class NeuronEdge : public GenericNeuronEdgeTy
{

public:
    enum NeuronEdgeK{
        Direct2Neuron ///For FNN & CNN
    };

    typedef  NeuronEdgeK NeuronGraphEdge;

public:
    /// construct only a kind edge : Direct2Neuron
    NeuronEdge(NeuronNode* s, NeuronNode* d, GEdgeFlag k) : GenericNeuronEdgeTy(s, d, k){
        std::cout<<"NNEdge is initing, SrcNode: "<<s<<" , DstNode: "<<d<<" , EdgeK: "<<k<<std::endl;
    }

    /// Destructor
    ~NeuronEdge() {}

    /// Determine whether is a NNEdge type
    inline bool is_NeuronNode() const {
        return getEdgeKind() == Direct2Neuron;
    }

    /// Determine whether is "Direct2Neuron"
    inline bool isDirect2NeuronEdge() const {
        return getEdgeKind()== Direct2Neuron;
    }

    /// Set<node,  edge>
    typedef GenericNode<NeuronNode, NeuronEdge>::GEdgeSetTy NeuronGraphEdgeSetTy;

    /// Insert into output stream
    friend OutStream& operator<<(OutStream& o, const NeuronEdge& edge)
    {
        o << edge.toString();
        return o;
    }

    virtual const std::string toString() const;

};

/// Edge Direct2Neuron
class Direct2NeuronEdge:public NeuronEdge{

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
