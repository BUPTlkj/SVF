#ifndef SVF_ONNX_SVFONNX_H
#define SVF_ONNX_SVFONNX_H
#include "iostream"
#include "Python.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include "fstream"
#include "regex"
#include "Eigen/Dense"
#include "Graphs/1NNGraph.h"


/// Conv Node todo maybe can be more concise
struct ConvParams
{
    std::string filterName;
    std::string filterDims;
    std::string filterValue;
    std::string biasName;
    std::string biasDims;
    std::string biasValue;
};

/// input node: Nothing needs to be done
struct ConstantNodeInfo
{
    std::string name;
};

/// Basic Op Node
struct BasicNodeInfo
{
    std::string name;    /// Which Nametype
    std::string typestr; ///  Just type
    std::tuple<int, int, int, int> dimensions;
    std::vector<Eigen::MatrixXd> values;
};

/// GEMM
struct ParsedGEMMParams
{
    std::string gemmName;
    std::string weightName;
    std::string weightDimensions;
    std::string weightValues;
    std::string biasName;
    std::string biasDimensions;
    std::string biasValues;
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
};

/// Conv
struct ConvNodeInfo
{
    std::string name;
    std::vector<SVF::FilterSubNode> filter;
    std::vector<double> conbias;
    std::pair<int, int> strides;
    std::pair<int, int> pads;
};

/// ReLu
struct ReluNodeInfo
{
    std::string name;
};

/// maxpool
struct MaxPoolNodeInfo
{
    std::string name;
    std::pair<int, int> windows;
    std::pair<int, int> strides;
    std::pair<int, int> pads;
};

/// Neural Net struct
using SVFNeuralNet =
    std::variant<ConstantNodeInfo, BasicNodeInfo, ParsedGEMMParams,
                 ConvNodeInfo, ReluNodeInfo, MaxPoolNodeInfo>;


class SVFNN
{
    /// ONNX adress
    std::string onnxAdress;

    /// Ordered nodes
    std::vector<SVFNeuralNet> nodes;

public:
    /// Constructor
    SVFNN(std::string adress);

    std::vector<SVFNeuralNet> get_nodes();

    /// 1. Read Python Analysis
    std::string PyObjectToString(PyObject* pObj);
    std::map<std::string, std::string> processPythonDict(PyObject* pDict);
    std::map<std::string, std::string> callPythonFunction(
        const std::string& address, const std::string& functionName);

    /// 2. Parse function, return a string vector containing four parts of information
    std::vector<std::string> parseNodeData(const std::string& dataStr);

    std::vector<std::string> splitString(const std::string& str,
                                         const std::string& delimiter);
    std::pair<std::string, std::string> parseItem(const std::string& item);
    std::map<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>>
    parseConvItems(const std::vector<std::string>& items);
    std::map<std::string,
             std::pair<std::pair<int, int>,
                       std::pair<std::pair<int, int>, std::pair<int, int>>>>
    parseMaxPoolItems(const std::vector<std::string>& items);
    /// Parse maxpool & conv special info

    /// 3. Restoration of each part
    /// constant node op has been moved into constructor

    /// Basic OP Restore
    BasicNodeInfo parseBasicNodeString(const std::string& nodeString);

    /// GEMM (Fullycon) Restore
    std::string trim(const std::string& str,
                     const std::string& chars = "\t\n\v\f\r ");
    ParsedGEMMParams GEMMparseAndFormat(const std::string& input);
    Eigen::MatrixXd restoreGEMMWeightToMatrix(const std::string& dimensions,
                                              const std::string& values);
    Eigen::VectorXd restoreGEMMBiasMatrixFromStrings(
        const std::string& dimensionStr, const std::string& valuesStr);

    /// Conv Restore
    /// Add parse node
    ConvParams ConvparseAndFormat(const std::string& input);
    std::vector<SVF::FilterSubNode> parse_filters(const std::string& s,
                                                  unsigned num_filters,
                                                  unsigned kernel_height,
                                                  unsigned kernel_width,
                                                  unsigned kernel_depth);
    std::vector<double> parse_Convbiasvector(std::string s);

    /// ReLu's op has been moved into constructor
};



#endif //SVF_ONNX_SVFONNX_H
