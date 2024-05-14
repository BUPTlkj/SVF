#ifndef SVF_NNGRAPHBUILDER_H
#define SVF_NNGRAPHBUILDER_H
#include "Graphs/NNGraph.h"
#include "AE/Nnexe/Solver.h"
#include "svf-onnx/SVFONNX.h"

namespace SVF{

class NNGraphTraversal
{

public:
    /// Constructor
    NNGraphTraversal(){};
    /// Destructor
    ~NNGraphTraversal(){};


    /// By Node
    bool checkNodeInVariant(const NeuronNode* current, const NeuronNodeVariant& dst);
    void printPath(std::vector<const NeuronNode *> &path);
    NeuronNode* getNeuronNodePtrFromVariant(const NeuronNodeVariant& variant);
    NeuronNodeVariant convertToVariant(NeuronNode* node);
    void DFS(std::set<const NeuronNode *> &visited, std::vector<const NeuronNode *> &path, const NeuronNodeVariant *src, const NeuronNodeVariant *dst, Matrices in_x);
    void IntervalDFS(std::set<const NeuronNode *> &visited, std::vector<const NeuronNode *> &path, const NeuronNodeVariant *src, const NeuronNodeVariant *dst, IntervalMatrices in_x);
    /// Retrieve all paths (a set of strings) during graph traversal
    std::set<std::string>& getPaths(){
        return paths;
    }

private:
    std::set<std::string> paths;

};

/// NNGraph Bulid
class NNGraphBuilder {
private:
    /// init nodes
    std::unordered_map<std::string, ConstantNeuronNode*> ConstantNodeIns;
    std::unordered_map<std::string, ConvNeuronNode*> ConvNodeIns;
    std::unordered_map<std::string, ReLuNeuronNode*> ReLuNodeIns;
    std::unordered_map<std::string, FlattenNeuronNode*> FlattenNodeIns;
    std::unordered_map<std::string, MaxPoolNeuronNode*> MaxPoolNodeIns;
    std::unordered_map<std::string, FullyConNeuronNode*> FullyConNodeIns;
    std::unordered_map<std::string, BasicOPNeuronNode*> BasicOPNodeIns;
    /// Node's class name for visitors
    std::vector<std::string> OrderedNodeName;
    std::vector<std::unique_ptr<NeuronNode>> Nodeins;
    /// init edges
    std::vector<std::unique_ptr<Direct2NeuronEdge>> edges;
    /// init Graph
    NeuronNet *g = new NeuronNet();
    NodeID nodeId = 0;

public:
    /// Allocate the NodeID
    // inline u32_t getNodeID(const std::string& str);

    /// Thoese operator() is designed for collecting instance
    void operator()(const ConstantNodeInfo& node);
    void operator()(const BasicNodeInfo& node);
    void operator()(const FullyconnectedInfo& node);
    void operator()(const ConvNodeInfo& node);
    void operator()(const ReluNodeInfo& node);
    void operator()(const FlattenNodeInfo& node);
    void operator()(const MaxPoolNodeInfo& node);

    NeuronNodeVariant getNodeInstanceByName(const std::string& name) const;
    NeuronNode* getNeuronNodeInstanceByName(const std::string& name) const;

    bool isValidNode(const NeuronNodeVariant& node);
    void AddEdges();
    void Traversal(Matrices& in_x);
    void IntervalTraversal(IntervalMatrices & in_x);
    inline u32_t getNodeID();
    inline void setNodeID();
};

}

#endif // SVF_NNGRAPHBUILDER_H
