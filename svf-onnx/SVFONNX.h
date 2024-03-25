#ifndef SVF_ONNX_SVFONNX_H
#define SVF_ONNX_SVFONNX_H
#include "Eigen/Dense"
#include "Graphs/NNGraph.h"
#include "Python.h"
#include "fstream"
#include "iostream"
#include "regex"
#include "iostream"
#include "map"
#include "sstream"
#include "string"


namespace SVF
{

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
    std::tuple<u32_t, u32_t, u32_t, u32_t> dimensions;
    Matrices values;
    IntervalMatrices Intervalvalues;
};

/// GEMM
//struct ParsedGEMMParams
struct FullyconnectedInfo
{
    std::string gemmName;
    std::string weightName;
    std::string weightDimensions;
    std::string weightValues;
    std::string biasName;
    std::string biasDimensions;
    std::string biasValues;
    Mat weight;
    Vector bias;
    IntervalMat Intervalweight;
    IntervalMat Intervalbias;
};

/// Conv
struct ConvNodeInfo
{
    std::string name;
    std::vector<FilterSubNode> filter;
    std::vector<double> conbias;
    std::vector<IntervalValue> Intervalbias;
    std::pair<u32_t, u32_t> strides;
    std::pair<u32_t, u32_t> pads;
};

/// ReLu
struct ReluNodeInfo
{
    std::string name;
};

/// Flatten
struct FlattenNodeInfo
{
    std::string name;
};

/// maxpool
struct MaxPoolNodeInfo
{
    std::string name;
    std::pair<u32_t, u32_t> windows;
    std::pair<u32_t, u32_t> strides;
    std::pair<u32_t, u32_t> pads;
};

/// Neural Net struct
using SVFNeuralNet =
    std::variant<ConstantNodeInfo, BasicNodeInfo, FullyconnectedInfo, ConvNodeInfo, ReluNodeInfo, MaxPoolNodeInfo, FlattenNodeInfo>;

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
    std::vector<u32_t> parseDimensions(const std::string& input);
    std::vector<std::string> parseNodeData(const std::string& dataStr);

    std::vector<std::string> splitString(const std::string& str, const std::string& delimiter);
    std::pair<std::string, std::string> parseItem(const std::string& item);
    std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>>
    parseConvItems(const std::vector<std::string>& items);
    std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>>>
    parseMaxPoolItems(const std::vector<std::string>& items);
    /// Parse maxpool & conv special info

    /// 3. IntervalValue Process
    /// Matrices -> IntervalMatrix
    IntervalMatrices convertMatricesToIntervalMatrices(const Matrices& matrices);
    IntervalMat convertMatToIntervalMat(const Mat& matrix);
    IntervalMat convertVectorXdToIntervalVector(const Vector& vec);

    std::pair<Matrices, Matrices> splitIntervalMatrices(const IntervalMatrices & intervalMatrices);

    /// 4. Restoration of each part
    /// constant node op has been moved into constructor

    /// Basic OP Restore
    BasicNodeInfo parseBasicNodeString(const std::string& nodeString);

    /// GEMM (Fullycon) Restore
    std::string trim(const std::string& str, const std::string& chars = "\t\n\v\f\r ");
    FullyconnectedInfo GEMMparseAndFormat(const std::string& input);
    Mat restoreGEMMWeightToMatrix(const std::string& dimensions, const std::string& values);
    Vector restoreGEMMBiasMatrixFromStrings(const std::string& dimensionStr, const std::string& valuesStr);

    /// Conv Restore
    /// Add parse node
    ConvParams ConvparseAndFormat(const std::string& input);
    std::vector<FilterSubNode> parse_filters(const std::string& s, u32_t num_filters, u32_t kernel_height, u32_t kernel_width, u32_t kernel_depth);
    std::vector<double> parse_Convbiasvector(std::string s);

    /// ReLu's op has been moved into constructor
};

}

#endif //SVF_ONNX_SVFONNX_H