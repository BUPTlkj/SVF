#include "SVF-LLVM/NNgraphBuilder.h"
#include "Util/SVFUtil.h"
#include "iomanip"
#include <cstdlib>


using namespace SVF;


/// BY NODE
void NNGraphTraversal::printPath(std::vector<const NeuronNode *> &path){
    std::string output = "START: ";
    for (size_t i = 0; i < path.size(); ++i) {
        output += std::to_string(path[i]->getId());
        if (i < path.size() - 1) {
            output += "->";
        }
    }
    paths.insert(output);
};

NeuronNodeVariant NNGraphTraversal::convertToVariant(NeuronNode* node) {
    /// Check the specific type of node and construct NeuronNodeVariant accordingly
    if (auto* constantNode = SVFUtil::dyn_cast<ConstantNeuronNode>(node)) {
        return constantNode;
    } else if (auto* basicOPNode = SVFUtil::dyn_cast<BasicOPNeuronNode>(node)) {
        return basicOPNode;
    } else if (auto* fullyConNode = SVFUtil::dyn_cast<FullyConNeuronNode>(node)) {
        return fullyConNode;
    } else if (auto* convNode = SVFUtil::dyn_cast<ConvNeuronNode>(node)) {
        return convNode;
    } else if (auto* reLuNode = SVFUtil::dyn_cast<ReLuNeuronNode>(node)) {
        return reLuNode;
    } else if(auto* flattenNode = SVFUtil::dyn_cast<FlattenNeuronNode>(node)){
        return flattenNode;
    }else if (auto* maxPoolNode = SVFUtil::dyn_cast<MaxPoolNeuronNode>(node)) {
        return maxPoolNode;
    }

    ///  If the node does not match any known type, it can return std:: monostate or throw an exception
    return std::monostate{};
}


bool NNGraphTraversal::checkNodeInVariant(const NeuronNode* current, const NeuronNodeVariant& dst) {
    return std::visit([current](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            return current == static_cast<const NeuronNode*>(arg);
        } else {
            return false;
        }
    }, dst);
}

NeuronNode* NNGraphTraversal::getNeuronNodePtrFromVariant(const NeuronNodeVariant& variant) {
    return std::visit([](auto&& arg) -> NeuronNode* {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            /// Corresponding to the situation where variant does not hold any NeuronNode pointers
            return nullptr;
        } else {
            /// Return NeuronNode' pointer
            return arg;
        }
    }, variant);
}

///  3.13
/// matrix into interval
void NNGraphTraversal::DFS(std::set<const NeuronNode *> &visited, std::vector<const NeuronNode *> &path, const NeuronNodeVariant *src, const NeuronNodeVariant *dst, Matrices in_x) {
    std::stack<std::pair<NeuronNode*, std::vector<Mat>>> stack;

    /// Ensure that src is obtained by dereferencing NeuronNode
    stack.emplace(getNeuronNodePtrFromVariant(*src), in_x);

    SolverEvaluate solver(in_x);
    int i = 0;
    Matrices IRRes;

    while (!stack.empty()) {
        std::cout<<" Node: "<<i<<std::endl;
        i++;
        auto currentPair = stack.top();
        stack.pop();

        const NeuronNode* current = currentPair.first;
        IRRes = currentPair.second;

        std::cout<<&current<<std::endl<<" STACK SIZE: "<<stack.size()<<std::endl;
        std::cout<<" The size of Matrix: "<<IRRes.size()<<std::endl;

        if (!visited.insert(current).second) {
            std::cout<<"CONTINUE: This Node "<<&current<<" has already been visited!"<<std::endl;
            continue;
        }

        path.push_back(current);

        if (checkNodeInVariant(current, *dst)) {
            printPath(path);
        }


        for (const auto& edge : current->getOutEdges()) {
            NeuronNode *neighbor = edge->getDstNode();

            NeuronNodeVariant variantNeighbor = convertToVariant(neighbor);
            std::cout<<"SrcNode: "<<current<<" ->  DSTNode: "<<neighbor<<" -> ConvertIns: "<<&variantNeighbor<<std::endl;

            std::cout<<"NodeTYpe:  "<<neighbor->get_type()<<std::endl;

            if (visited.count(neighbor) == 0) {
                /// Process IRRes based on the type of neighbor
                Matrices newIRRes; /// Copy the current IRRes to avoid modifying the original data
                if (neighbor->get_type() == 0) {
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ReLuNeuronNodeevaluate();
                    std::cout<<"FINISH RELU"<<std::endl;
                } else if (neighbor->get_type() == 1 || neighbor->get_type() == 6 || neighbor->get_type() == 7 || neighbor->get_type() == 8 || neighbor->get_type() == 9) {
                    const BasicOPNeuronNode *node = SVFUtil::dyn_cast<BasicOPNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.BasicOPNeuronNodeevaluate(node);
                    std::cout<<"FINISH BAIC"<<std::endl;
                } else if (neighbor->get_type() == 2) {
                    const MaxPoolNeuronNode *node = SVFUtil::dyn_cast<MaxPoolNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.MaxPoolNeuronNodeevaluate(node);
                    std::cout<<"FINISH MAXPOOLING"<<std::endl;
                } else if (neighbor->get_type() == 3) {
                    const ConvNeuronNode *node = static_cast<ConvNeuronNode *>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ConvNeuronNodeevaluate(node);
                    std::cout<<"FINISH Conv"<<std::endl;
                } else if (neighbor->get_type() == 4) {
                    const FullyConNeuronNode *node = SVFUtil::dyn_cast<FullyConNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    std::cout<<"*******************"<<std::endl;
                    newIRRes = solver.FullyConNeuronNodeevaluate(node);
                    std::cout<<"FINISH FullyConnected"<<std::endl;
                } else if (neighbor->get_type() == 5) {
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ConstantNeuronNodeevaluate();
                    std::cout<<"FINISH Constant"<<std::endl;
                } else if (neighbor->get_type() == 10){
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.FlattenNeuronNodeevaluate();
                    std::cout<<"FINISH Flatten"<<std::endl;
                }
                /// Push neighbor and new IRRes into the stack
                IRRes = newIRRes;
                stack.emplace(neighbor, newIRRes);
                std::cout<<"FINISH PUSHING STACK! "<<stack.size()<<std::endl<<std::endl;
            }
        }

        /// print the IRRes Matrix
        std::cout << "IRRes content after the loop iteration:" << i << std::endl;
        std::cout.precision(20);
        std::cout << std::fixed;
        for (size_t j = 0; j < IRRes.size(); ++j) {
            std::cout << "Matrix " << j << ":\n";
            std::cout << "Rows: " << IRRes[j].rows() << ", Columns: " << IRRes[j].cols() << "\n";
            std::cout << IRRes[j] << "\n\n";
        }
        visited.erase(current);
        path.pop_back(); /// Remove current node
    }
}

///  3.13
/// matrix into interval
void NNGraphTraversal::IntervalDFS(std::set<const NeuronNode *> &visited, std::vector<const NeuronNode *> &path, const NeuronNodeVariant *src, const NeuronNodeVariant *dst, IntervalMatrices in_x) {
    std::stack<std::pair<NeuronNode*, IntervalMatrices *>> stack;
    NNgraphIntervalSolver solver(in_x);

    /// Ensure that src is obtained by dereferencing NeuronNode
    stack.emplace(getNeuronNodePtrFromVariant(*src), &solver.interval_data_matrix);

    int i = 0;
    IntervalMatrices IRRes;
    IntervalMatrices newIRRes; /// Copy the current IRRes to avoid modifying the original data

    while (!stack.empty()) {
        std::cout<<" Node: "<<i<<std::endl;
        i++;
        auto currentPair = stack.top();
        stack.pop();

        const NeuronNode* current = currentPair.first;

        IRRes = *currentPair.second;
        std::cout<<"*******IRRes:"<<IRRes.size()<<", "<<IRRes[0].rows()<<", "<<IRRes[0].cols()<<std::endl;

        std::cout<<&current<<std::endl<<" STACK SIZE: "<<stack.size()<<std::endl;
        std::cout<<" The size of Matrix: "<<IRRes.size()<<std::endl;

        if (!visited.insert(current).second) {
            std::cout<<"CONTINUE: This Node "<<&current<<" has already been visited!"<<std::endl;
            continue;
        }

        path.push_back(current);

        if (checkNodeInVariant(current, *dst)) {
            printPath(path);
        }

        for (const auto& edge : current->getOutEdges()) {
            NeuronNode *neighbor = edge->getDstNode();

            NeuronNodeVariant variantNeighbor = convertToVariant(neighbor);
            std::cout<<"SrcNode: "<<current<<" ->  DSTNode: "<<neighbor<<" -> ConvertIns: "<<&variantNeighbor<<std::endl;

            std::cout<<"NodeTYpe:  "<<neighbor->get_type()<<std::endl;

            if (visited.count(neighbor) == 0) {
                /// Process IRRes based on the type of neighbor
                if (neighbor->get_type() == 0) {
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ReLuNeuronNodeevaluate();
                    std::cout<<"FINISH RELU, The Result size:  ("<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                } else if (neighbor->get_type() == 1 || neighbor->get_type() == 6 || neighbor->get_type() == 7 || neighbor->get_type() == 8 || neighbor->get_type() == 9) {
                    const BasicOPNeuronNode *node = SVFUtil::dyn_cast<BasicOPNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.BasicOPNeuronNodeevaluate(node);
                    std::cout<<"FINISH BAIC, The Result size: ("<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                } else if (neighbor->get_type() == 2) {
                    const MaxPoolNeuronNode *node = SVFUtil::dyn_cast<MaxPoolNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.MaxPoolNeuronNodeevaluate(node);
                    std::cout<<"FINISH MAXPOOLING, The Result size: ("<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                } else if (neighbor->get_type() == 3) {
                    const ConvNeuronNode *node = static_cast<ConvNeuronNode *>(neighbor);
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ConvNeuronNodeevaluate(node);
                    std::cout<<"FINISH Conv, The Result size: ("<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                } else if (neighbor->get_type() == 4) {
                    const FullyConNeuronNode *node = SVFUtil::dyn_cast<FullyConNeuronNode>(neighbor);
                    solver.setIRMatrix(IRRes);
                    std::cout<<"*******************"<<std::endl;
                    newIRRes = solver.FullyConNeuronNodeevaluate(node);
                    std::cout<<"FINISH FullyConnected, The Result size: ("<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                } else if (neighbor->get_type() == 5) {
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.ConstantNeuronNodeevaluate();
                    std::cout<<"FINISH Constant, The Result size: ("<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                }else if (neighbor->get_type() == 10) {
                    solver.setIRMatrix(IRRes);
                    newIRRes = solver.FlattenNeuronNodeevaluate();
                    std::cout<<"FINISH Flatten, The Result size: ("<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                }
                /// Push neighbor and new IRRes into the stack
                IRRes = newIRRes;
                std::cout<<"Ready to push: "<<newIRRes.size()<<", "<<newIRRes[0].rows()<<", "<<newIRRes[0].cols()<<")"<<std::endl;
                stack.emplace(neighbor, &newIRRes);
                std::cout<<"FINISH PUSHING STACK! "<<stack.size()<< ",    Number of Intervalmatrix: " << newIRRes.size() <<std::endl<<std::endl;
            }
        }

        /// print the IRRes Matrix
        std::cout << "IRRes content after the loop iteration:" << i << ",    Number of Intervalmatrix: " << IRRes.size() <<std::endl;
        std::cout.precision(20);
        std::cout << std::fixed;
        for (const auto& intervalMat : IRRes) {
            std::cout << "IntervalMatrix :\n";
            std::cout << "Rows: " << intervalMat.rows() << ", Columns: " << intervalMat.cols() << "\n";
            for (u32_t k = 0; k < intervalMat.rows(); ++k) {
                for (u32_t j = 0; j < intervalMat.cols(); ++j) {
                    std::cout<< "[ "<< intervalMat(k, j).lb().getRealNumeral()<<", "<< intervalMat(k, j).ub().getRealNumeral() <<" ]"<< "\t";
                }
                std::cout << std::endl;
            }
            std::cout<<"****************"<<std::endl;
        }

        visited.erase(current);
        path.pop_back(); /// Remove current node
    }
}

/// NNGraph Bulid
/// Allocate the NodeID
inline u32_t NNGraphBuilder::getNodeID(const std::string& str) {
    size_t underscorePos = str.find('_'); /// Find the location of "_"
    if (underscorePos == std::string::npos) {
        std::cout<<"NodeID has been not allocated!"<<std::endl;
        exit(0);
    }
    /// Extract substrings before "_"
    std::string numberStr = str.substr(0, underscorePos);

    size_t endpos = numberStr.find_last_not_of(" ");
    if (std::string::npos != endpos) {
        /// Delete all characters from endpos+1 to the end of the string
        numberStr = numberStr.substr(1, endpos + 1);
        if (numberStr.length() == 2 && numberStr[0] == '0'){
            numberStr = numberStr.substr(1, 1);
        }
    }
    u32_t number = std::stoi(numberStr);
    return number;
}

/// Thoese operator() is designed for collecting instance
void NNGraphBuilder::operator()(const ConstantNodeInfo& node) {
    NodeID id = getNodeID(node.name);
    OrderedNodeName.push_back(node.name);
    ConstantNodeIns[node.name] = new ConstantNeuronNode(id);
    g->addConstantNeuronNode(ConstantNodeIns[node.name]);
}

void NNGraphBuilder::operator()(const BasicNodeInfo& node)  {
    NodeID id = getNodeID(node.name);
    OrderedNodeName.push_back(node.name);
    BasicOPNodeIns[node.name] = new BasicOPNeuronNode(id, node.typestr, node.values, node.Intervalvalues);
    g->addBasicOPNeuronNode(BasicOPNodeIns[node.name]);
}

void NNGraphBuilder::operator()(const FullyconnectedInfo& node)  {
    NodeID id = getNodeID(node.gemmName);
    OrderedNodeName.push_back(node.gemmName);
    FullyConNodeIns[node.gemmName] = new FullyConNeuronNode(id, node.weight, node.bias, node.Intervalweight, node.Intervalbias);
    g->addFullyConNeuronNode(FullyConNodeIns[node.gemmName]);
}

void NNGraphBuilder::operator()(const ConvNodeInfo& node) {
    NodeID id = getNodeID(node.name);
    OrderedNodeName.push_back(node.name);
    ConvNodeIns[node.name] = new ConvNeuronNode(id, node.filter, node.conbias, node.pads.first, node.strides.first, node.Intervalbias);
    g->addConvNeuronNode(ConvNodeIns[node.name]);

}

void NNGraphBuilder::operator()(const ReluNodeInfo& node) {
    NodeID id = getNodeID(node.name);
    OrderedNodeName.push_back(node.name);
    ReLuNodeIns[node.name] = new ReLuNeuronNode(id);
    g->addReLuNeuronNode(ReLuNodeIns[node.name]);
}

void NNGraphBuilder::operator()(const FlattenNodeInfo& node) {
    NodeID id = getNodeID(node.name);
    OrderedNodeName.push_back(node.name);
    FlattenNodeIns[node.name] = new FlattenNeuronNode(id);
    g->addFlattenNeuronNode(FlattenNodeIns[node.name]);
}

void NNGraphBuilder::operator()(const MaxPoolNodeInfo& node) {
    NodeID id = getNodeID(node.name);
    OrderedNodeName.push_back(node.name);
    MaxPoolNodeIns[node.name] = new MaxPoolNeuronNode(id, node.windows.first, node.windows.second, node.strides.first, node.strides.second, node.pads.first, node.pads.second);
    g->addMaxPoolNeuronNode(MaxPoolNodeIns[node.name]);
}

NeuronNodeVariant NNGraphBuilder::getNodeInstanceByName(const std::string& name) const {
    if (auto it = ConstantNodeIns.find(name); it != ConstantNodeIns.end()) return it->second;
    if (auto it = ConvNodeIns.find(name); it != ConvNodeIns.end()) return it->second;
    if (auto it = ReLuNodeIns.find(name); it != ReLuNodeIns.end()) return it->second;
    if (auto it = MaxPoolNodeIns.find(name); it != MaxPoolNodeIns.end()) return it->second;
    if (auto it = FullyConNodeIns.find(name); it != FullyConNodeIns.end()) return it->second;
    if (auto it = BasicOPNodeIns.find(name); it != BasicOPNodeIns.end()) return it->second;
    if (auto it = FlattenNodeIns.find(name); it != FlattenNodeIns.end()) return it->second;

    return std::monostate{};
}

NeuronNode* NNGraphBuilder::getNeuronNodeInstanceByName(const std::string& name) const {
    if (auto it = ConstantNodeIns.find(name); it != ConstantNodeIns.end()) return it->second;
    if (auto it = ConvNodeIns.find(name); it != ConvNodeIns.end()) return it->second;
    if (auto it = ReLuNodeIns.find(name); it != ReLuNodeIns.end()) return it->second;
    if (auto it = MaxPoolNodeIns.find(name); it != MaxPoolNodeIns.end()) return it->second;
    if (auto it = FullyConNodeIns.find(name); it != FullyConNodeIns.end()) return it->second;
    if (auto it = BasicOPNodeIns.find(name); it != BasicOPNodeIns.end()) return it->second;
    if (auto it = FlattenNodeIns.find(name); it != FlattenNodeIns.end()) return it->second;
    return nullptr;
}

bool NNGraphBuilder::isValidNode(const NeuronNodeVariant& node) {
    return !std::holds_alternative<std::monostate>(node);
}

void NNGraphBuilder::AddEdges() {

    for (size_t i = 0; i < OrderedNodeName.size() - 1; ++i) {
        const auto& currentName = OrderedNodeName[i];
        const auto& nextName = OrderedNodeName[i + 1];

        NeuronNode* currentNode = getNeuronNodeInstanceByName(currentName);
        NeuronNode* nextNode = getNeuronNodeInstanceByName(nextName);

        if (currentNode && nextNode) {
            /// Ensure edge is created as a unique_ptr<Direct2NeuronEdge>
            auto edge = std::make_unique<Direct2NeuronEdge>(currentNode, nextNode);
            edges.push_back(std::move(edge)); // This should now work
        }
    }

    for (const auto& edge : edges){
        g->addDirected2NodeEdge(edge.get());
    }

}

void NNGraphBuilder::Traversal(Matrices& in_x) {

    /// Print the dataset matrix
    for(u32_t j=0; j<in_x.size();j++){
        std::cout<<"Matrix: "<<j<<std::endl;
        std::cout<<in_x[j]<<std::endl;
    }

    /// Note: Currently, visited and path store pointers to  NeuronNodeVariant
    std::set<const NeuronNode *> visited;
    std::vector<const NeuronNode *> path;
    auto *dfs = new NNGraphTraversal();


    const std::string& LastName = OrderedNodeName[OrderedNodeName.size() - 1];
    const std::string& FirstName = OrderedNodeName[0];

    /// getNodeInstanceByName() return type: NeuronNodeVariant
    NeuronNodeVariant FirstNode = getNodeInstanceByName(FirstName); /// Return NeuronNodeVariant
    NeuronNodeVariant LastNode = getNodeInstanceByName(LastName); /// Return NeuronNodeVariant

    /// Due to DFS now accepting parameters:  NeuronNodeVariant type, directly passing the addresses of FirstNode and LastNode
    dfs->DFS(visited, path, &FirstNode, &LastNode, in_x);
    std::set<std::string>& stringPath = dfs->getPaths();
    std::cout<<"GET PATH"<<stringPath.size()<<std::endl;
    u32_t i = 0;
    for (const std::string& paths : stringPath) {
        std::cout << i <<"*****"<< paths << std::endl;
        i++;
    }

    delete dfs; /// Delete allocated memory
}
void NNGraphBuilder::IntervalTraversal(IntervalMatrices & in_x) {

    /// Print the dataset matrix
    std::cout.precision(20);
    std::cout << std::fixed;
    for (const auto& intervalMat : in_x) {
        std::cout << "IntervalMatrix :\n";
        std::cout << "Rows: " << intervalMat.rows() << ", Columns: " << intervalMat.cols() << "\n";
        for (u32_t k = 0; k < intervalMat.rows(); ++k) {
            for (u32_t j = 0; j < intervalMat.cols(); ++j) {
                std::cout<< "[ "<< intervalMat(k, j).lb().getRealNumeral()<<", "<< intervalMat(k, j).ub().getRealNumeral() <<" ]"<< "\t";
            }
            std::cout << std::endl;
        }
        std::cout<<"****************"<<std::endl;
    }


    /// Note: Currently, visited and path store pointers to  NeuronNodeVariant
    std::set<const NeuronNode *> visited;
    std::vector<const NeuronNode *> path;
    auto *dfs = new NNGraphTraversal();

    const std::string& LastName = OrderedNodeName[OrderedNodeName.size() - 1];
    const std::string& FirstName = OrderedNodeName[0];

    /// getNodeInstanceByName() return type: NeuronNodeVariant
    NeuronNodeVariant FirstNode = getNodeInstanceByName(FirstName); /// Return NeuronNodeVariant
    NeuronNodeVariant LastNode = getNodeInstanceByName(LastName); /// Return NeuronNodeVariant

    /// Due to DFS now accepting parameters:  NeuronNodeVariant type, directly passing the addresses of FirstNode and LastNode
    dfs->IntervalDFS(visited, path, &FirstNode, &LastNode, in_x);
    std::set<std::string>&  stringPath = dfs->getPaths();
    std::cout<<"GET PATH"<<stringPath.size()<<std::endl;
    u32_t i = 0;
    for (const std::string& paths : stringPath) {
        std::cout << i <<"*****"<< paths << std::endl;
        i++;
    }

    delete dfs; /// Delete allocated memory
}
