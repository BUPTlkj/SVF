#include "SVFONNX.h"
#include "AE/Nnexe/NNgraphIntervalSolver.h"
#include "algorithm" /// For std::remove
#include "filesystem"
#include "Util/Options.h"
#include <filesystem>
namespace fs = std::filesystem;
#define CUR_DIR() (fs::path(__FILE__).parent_path())

// Parse function to extract integers from a given string and return their
/// vectors

using namespace SVF;

std::vector<u32_t> SVFNN::parseDimensions(const std::string& input) {
    std::vector<u32_t> dimensions;
    std::stringstream ss(input);
    u32_t number;
    char comma;

    while (ss >> number) {
        dimensions.push_back(number);
        ss >> comma; /// Read and ignore commas
    }
    return dimensions;
}

/// Constructor, just provide the ONNX address
SVFNN::SVFNN(std::string adress): onnxAdress{adress}{
    std::string Constant = "Constant";
    /// basci op
    std::string Sub = "Sub";
    std::string Add = "Add";
    std::string Mul = "Mul";
    std::string Div = "Div";

    std::string Gemm = "Gemm";
    std::string Conv = "Conv";
    std::string Relu = "Relu";
    std::string  Flatten = "Flatten";
    /// 22 Feb
    std::string MaxPool = "MaxPool";
    /// The information of Maxpool and conv is in SpecialInfo
    std::string SepcialInfo = "SepcialInfo";

    auto cppMapa = callPythonFunction(onnxAdress, "InteractwithCpp");

/// Put it here?? Need for optimization
    ConstantNodeInfo constantnode;
    BasicNodeInfo basicnode;
    FullyconnectedInfo gemmnode;
    ConvNodeInfo convnode;
    ReluNodeInfo relunode;
    FlattenNodeInfo flattennode;
    MaxPoolNodeInfo maxpoolnode;

    std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>> convItems;
    std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>>>  maxPoolItems;

    for(const auto& pair : cppMapa) {
        /// Key: pair.first  Value: pair.second
        std::string name = pair.first;
        auto nodeDataParts = parseNodeData(pair.second);
        for (u32_t i = 0; i < nodeDataParts.size(); ++i) {
            if (i == 3) {
                if (name.find(SepcialInfo) != std::string::npos) {
                    /// Modify the previous structure
                    std::string input = pair.second;
                    /// Preprocessing input strings to fit segmentation functions
                    /// Remove curly braces and single quotes at the beginning and end
                    input = input.substr(2, input.size() - 6);

                    /// Split String
                    auto items = splitString(input, "]], ");

                    /// Parsing items of Conv and MaxPool types
                    convItems = parseConvItems(items);
                    maxPoolItems = parseMaxPoolItems(items);
                }
            }
        }
    }

    for(const auto& pair : cppMapa){
        /// Key: pair.first  Value: pair.second
        std::string name = pair.first;
        auto nodeDataParts = parseNodeData(pair.second);

        for(u32_t i=0; i<nodeDataParts.size(); ++i){

            if(i==3){ /// The third one is specific dimensions and data
                if (name.find(Constant) != std::string::npos){
                    constantnode.name = name;
                    nodes.push_back(constantnode);
                }else if(name.find(Sub) != std::string::npos || name.find(Add) != std::string::npos || name.find(Mul) != std::string::npos || name.find(Div) != std::string::npos){
                    basicnode = parseBasicNodeString(nodeDataParts[2]);
                    basicnode.typestr = nodeDataParts[0];
                    basicnode.name = name;
                    basicnode.Intervalvalues = convertMatricesToIntervalMatrices(basicnode.values);
                    nodes.push_back(basicnode);
                }else if(name.find(Gemm) != std::string::npos){
                    gemmnode = GEMMparseAndFormat(nodeDataParts[2]);
                    gemmnode.gemmName = name;
                    Mat weight = restoreGEMMWeightToMatrix(gemmnode.weightDimensions, gemmnode.weightValues);
                    Mat bias = restoreGEMMBiasMatrixFromStrings(gemmnode.biasDimensions, gemmnode.biasValues);
                    gemmnode.weight = weight;
                    gemmnode.bias = bias;
                    gemmnode.Intervalweight = convertMatToIntervalMat(weight);
                    gemmnode.Intervalbias = convertMatToIntervalMat(bias);
                    nodes.push_back(gemmnode);
                }else if(name.find(Conv) != std::string::npos){
                    /// Refer to GEMM
                    ConvParams conf = ConvparseAndFormat(nodeDataParts[2]);
                    /// filter: filters
                    convnode.filter = parse_filters(conf.filterValue, parseDimensions(conf.filterDims)[0], parseDimensions(conf.filterDims)[1], parseDimensions(conf.filterDims)[2], parseDimensions(conf.filterDims)[3] );
                    convnode.conbias = parse_Convbiasvector(conf.biasValue);

                    for (double val : convnode.conbias) {
                        SVF::IntervalValue intervalValue(val, val);
                        convnode.Intervalbias.push_back(intervalValue);
                    }
                    convnode.name = name;
                    /// Add strides info
                    if (convItems.find(name) != convItems.end()){
                        auto stridesInfo = convItems[name];
                        convnode.pads = std::make_pair(stridesInfo.first.first, stridesInfo.first.second);
                        convnode.strides = std::make_pair(stridesInfo.second.first, stridesInfo.second.second);
                    }
                    nodes.push_back(convnode);
                }else if(name.find(Relu) != std::string::npos){
                    relunode.name = name;
                    nodes.push_back(relunode);
                }else if(name.find(MaxPool) != std::string::npos){
                    maxpoolnode.name = name;
                    /// Add windows and strides
                    if(maxPoolItems.find(name) != maxPoolItems.end()){
                        auto maxInfo = maxPoolItems[name];
                        maxpoolnode.windows = maxInfo.first;
                        maxpoolnode.pads = maxInfo.second.first;
                        maxpoolnode.strides = maxInfo.second.second;
                    }
                    nodes.push_back(maxpoolnode);
                }else if(name.find(Flatten) != std::string::npos){
                    flattennode.name = name;
                    nodes.push_back(flattennode);
                }
            }
        }
    }
}

std::vector<std::string> SVFNN::splitString(const std::string &str, const std::string &delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = 0;

    while ((end = str.find(delimiter, start)) != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    /// Add the last token
    tokens.push_back(str.substr(start));

    return tokens;
}

std::pair<std::string, std::string> SVFNN::parseItem(const std::string &item) {
    auto colonPos = item.find(':');
    if (colonPos != std::string::npos) {
        std::string key = item.substr(0, colonPos);
        std::string value = item.substr(colonPos + 1);
        return {key, value};
    }
    return {"", ""}; /// Returns an empty key value pair, indicating parsing failure
}

std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>> SVFNN::parseConvItems(const std::vector<std::string> &items) {
    std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>> result;
    for (const auto& item : items) {
        auto key_value = parseItem(item);
        if (key_value.first.find("Conv") != std::string::npos) {
            /// if value's format is correct
            u32_t px, py, px_, py_, sx, sy;
            sscanf(key_value.second.c_str(), " ['pads:', [%d, %d, %d, %d], 'strides:', [%d, %d]]", &px, &py, &px_, &py_, &sx, &sy); // �����ַ���Ϊ��������
            result[key_value.first] = std::make_pair(std::make_pair(px, py), std::make_pair(sx, sy));
        }
    }
    return result;
}

std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>>>
SVFNN::parseMaxPoolItems(const std::vector<std::string> &items) {
    std::map<std::string, std::pair<std::pair<u32_t, u32_t>, std::pair<std::pair<u32_t, u32_t>, std::pair<u32_t, u32_t>>>> result;
    for (const auto& item : items) {
        auto key_value = parseItem(item);
        if (key_value.first.find("MaxPool") != std::string::npos) {
            /// if value's format is correct
            u32_t wx, wy, px, py, px_, py_, sx, sy;
            sscanf(key_value.second.c_str(), " ['Windows:', [%d, %d], 'pads:', [%d, %d, %d, %d], 'strides:', [%d, %d]]", &wx, &wy, &px, &py,  &px_, &py_, &sx, &sy); // �����ַ���Ϊ�ĸ�����
            result[key_value.first] = std::make_pair(std::make_pair(wx, wy), std::make_pair(std::make_pair(px, py), std::make_pair(sx, sy)));
        }
    }
    return result;
}

std::vector<SVFNeuralNet> SVFNN::get_nodes(){
    return nodes;
}

IntervalMatrices SVFNN::convertMatricesToIntervalMatrices(const Matrices& matrices) {
    IntervalMatrices intervalMatrices;

    for (const auto& mat : matrices) {
        IntervalMat intervalMatrix(mat.rows(), mat.cols());
        for (u32_t i = 0; i < mat.rows(); ++i) {
            for (u32_t j = 0; j < mat.cols(); ++j) {
                /// [x, x]
                intervalMatrix(i, j) = IntervalValue(mat(i, j), mat(i, j));
            }
        }
        intervalMatrices.push_back(intervalMatrix);
    }
    return intervalMatrices;
}

IntervalMat SVFNN::convertMatToIntervalMat(const Mat& matrix){

    IntervalMat intervalMatrix(matrix.rows(), matrix.cols());
    for (u32_t i = 0; i < matrix.rows(); ++i) {
        for (u32_t j = 0; j < matrix.cols(); ++j) {
            /// [x, x]
            intervalMatrix(i, j) = IntervalValue(matrix(i, j), matrix(i, j));
        }
    }
    return intervalMatrix;
}

IntervalMat SVFNN::convertVectorXdToIntervalVector(const Vector& vec) {
    ///cols = 1
    IntervalMat intervalMat(vec.size(), 1);

    for (u32_t i = 0; i < vec.size(); ++i) {
        /// IntervalValue(double, double)
        intervalMat(i, 0) = IntervalValue(vec(i), vec(i));
    }

    return intervalMat;
}

std::string SVFNN::PyObjectToString(PyObject *pObj) {
    PyObject* pRepr = PyObject_Repr(pObj);  /// Get the printable representation of an object
    const char* s = PyUnicode_AsUTF8(pRepr);
    std::string result(s);
    Py_DECREF(pRepr);
    return result;
}

std::map<std::string, std::string> SVFNN::processPythonDict(PyObject *pDict) {
    std::map<std::string, std::string> cppMap;

    if (!PyDict_Check(pDict)) {
        std::cerr << "Provided object is not a dictionary." << std::endl;
        return cppMap;
    }

    PyObject *pKey, *pValue;
    Py_ssize_t pos = 0;

    while (PyDict_Next(pDict, &pos, &pKey, &pValue)) {
        std::string key = PyObjectToString(pKey);
        std::string value = PyObjectToString(pValue);  /// Simplify processing by directly converting values to strings
        cppMap[key] = value;
    }

    return cppMap;
}

std::map<std::string, std::string> SVFNN::callPythonFunction(const std::string& address, const std::string& functionName) {
    Py_Initialize();

    PyObject *pModule, *pFunc, *pArgs, *pValue;
    std::map<std::string, std::string> cppMap;
    std::string pathstring;

    PyRun_SimpleString("import sys");

    std::string pyscriptpath = CUR_DIR();
    std::string command = "sys.path.append('" + pyscriptpath + "')";
    PyRun_SimpleString(command.c_str());

    /// Import Python script
    pModule = PyImport_ImportModule("SVFModel_read");

    if (pModule != nullptr) {
        pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_Pack(1, PyUnicode_FromString(address.c_str()));
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != nullptr && PyDict_Check(pValue)) {
                cppMap = processPythonDict(pValue);
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
            }
        } else {
            PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
    }
    /// Release Python interpretation
    Py_Finalize();

    return cppMap;
}

std::vector<std::string> SVFNN::parseNodeData(const std::string &dataStr) {
    std::vector<std::string> parts;
    size_t pos = 0, startPos = 0, endPos = 0;

    /// Part 1: Find the first string
    startPos = dataStr.find_first_of('\'', pos) + 1; /// Skip the first quotation mark
    endPos = dataStr.find_first_of('\'', startPos);
    parts.push_back(dataStr.substr(startPos, endPos - startPos));

    /// Part 2: Find the first list
    startPos = dataStr.find_first_of('[', endPos) + 1; /// Skip the first [
    endPos = dataStr.find_first_of(']', startPos);
    parts.push_back(dataStr.substr(startPos, endPos - startPos));

    /// Part 3: Searching for a Dictionary
    startPos = dataStr.find_first_of('{', endPos) + 1; /// Skip the first {
    endPos = dataStr.find_first_of('}', startPos);
    parts.push_back(dataStr.substr(startPos, endPos - startPos));

    /// Part 4: Find the second list
    startPos = dataStr.find_first_of('[', endPos + 1) + 1; /// Skip the second [
    endPos = dataStr.find_first_of(']', startPos);
    parts.push_back(dataStr.substr(startPos, endPos - startPos));

    return parts;
}

BasicNodeInfo SVFNN::parseBasicNodeString(const std::string &nodeString) {
    std::regex pattern(R"('(\d+)': \[\((\d+), (\d+), (\d+), (\d+)\), \[\[\[\[([\d., ]+)\]\]\]\]\])");
    std::smatch matches;

    if (std::regex_search(nodeString, matches, pattern)) {
        BasicNodeInfo info;
        info.name = matches[1];
        info.dimensions = std::make_tuple(std::stoi(matches[2]), std::stoi(matches[3]), std::stoi(matches[4]),
                                          std::stoi(matches[5]));

        std::string valuesStr = matches[6];
        std::vector<std::string> splitValues;
        std::regex valuePattern(R"(([\d.]+))");
        std::sregex_token_iterator iter(valuesStr.begin(), valuesStr.end(), valuePattern);
        std::sregex_token_iterator end;

        for (; iter != end; ++iter) {
            double value = std::stod(*iter);
            info.values.push_back(Mat::Constant(1, 1, value));
        }

        return info;
    }
    return BasicNodeInfo();
}

/// Auxiliary function: Remove specific characters from both ends of a string
std::string SVFNN::trim(const std::string& str, const std::string& chars) {
    size_t start = str.find_first_not_of(chars);
    if (start == std::string::npos) return "";

    size_t end = str.find_last_not_of(chars);
    return str.substr(start, end - start + 1);
}

/// Analyze and format weight and bias information
FullyconnectedInfo SVFNN::GEMMparseAndFormat(const std::string& input) {
    FullyconnectedInfo params;
    std::regex re("'(.*?)': \\[\\((.*?)\\), \\[(.*?)\\]\\]");

    std::smatch match;
    std::string str = input;

    std::cout<<"**"<<str;

    while (std::regex_search(str, match, re)) {
        std::string name = match[1];

        if(name.find("weight")!=std::string::npos){
            params.weightName = name;
            params.weightDimensions = trim(match[2], " ()");
            params.weightValues = trim(match[3], " []");
        }

        if(name.find("bias")!=std::string::npos){
            params.biasName = name;
            params.biasDimensions = trim(match[2], " ()");
            params.biasValues = trim(match[3], " []");
        }

        str = match.suffix().str(); /// Continue to search for the next match
    }

    return params;
}

Mat SVFNN::restoreGEMMWeightToMatrix(const std::string& dimensions, const std::string& values) {
    /// Analyze dimensions
    std::istringstream dimStream(dimensions);
    u32_t rows, cols;
    char comma; /// For skipping commas
    dimStream >> rows >> comma >> cols;

    std::string processedValues = std::regex_replace(values, std::regex("\\], \\["), ", ");

    /// Parsing Numeric Strings
    std::vector<double> vals;
    std::istringstream valStream(processedValues);
    std::string val;
    while (std::getline(valStream, val, ',')) {
        vals.push_back(std::stod(val));
    }

    /// Confirm the number of values to match the size of the matrix
    if (vals.size() != rows * cols) {
        std::cerr << "Value count does not match matrix dimensions!" << std::endl;
        return Mat(0, 0); /// Empty Matrix
    }

    /// Padding matrix
    Mat matrix(rows, cols);

    for (u32_t i = 0; i < rows; ++i) {
        for (u32_t j = 0; j < cols; ++j) {
            matrix(i, j) = vals[i * cols + j];
        }
    }

    return matrix;
}

Vector SVFNN::restoreGEMMBiasMatrixFromStrings(const std::string& dimensionStr, const std::string& valuesStr) {
    /// Processing dimension strings, removing commas
    std::string dimStr = dimensionStr;
    if (!dimStr.empty() && dimStr.back() == ',') {
        dimStr.pop_back(); // Remove the last character (comma)
    }

    /// Parsing dimension strings
    u32_t rows = 1;
    u32_t cols;
    std::istringstream dimStream(dimStr);
    dimStream >> cols; // Read the number of columns

    /// Parsing value strings
    std::vector<double> values;
    std::istringstream valueStream(valuesStr);
    std::string value;
    while (std::getline(valueStream, value, ',')) {
        values.push_back(std::stod(value));
    }

    /// Ensure that the number of values matches the dimension
    if (values.size() != cols) {
        std::cerr << "Error: The number of values does not match the specified dimensions." << std::endl;
        return Mat(); // Return an empty matrix
    }

    /// Fill the matrix with parsed values
    Mat matrix(rows, cols);
    for (u32_t j = 0; j < cols; ++j) {
        matrix(0, j) = values[j];
    }
    Vector vec = Eigen::Map<Vector>(matrix.data(), matrix.size());
    return vec;
}

ConvParams SVFNN::ConvparseAndFormat(const std::string& input) {
    ConvParams params;
    std::regex re("'(.*?)': \\[\\((.*?)\\), \\[\\[\\[\\[(.*?)\\]\\]\\]\\]\\], '(.*?)': \\[\\((.*?)\\), \\[(.*?)\\]\\]");

    std::smatch match;
    std::string str = input;

    while (std::regex_search(str, match, re)) {
        std::string name = match[1];
        std::string name2 = match[4];

        if(name.find("weight")!=std::string::npos){
            params.filterName = name;
            params.filterDims = match[2];
            params.filterValue =  match[3];
            params.filterValue =  "[[[[" + params.filterValue+ "]]]]";
        }

        if(name2.find("bias")!=std::string::npos){
            params.biasName = name2;
            params.biasDims = match[5];
            params.biasValue = match[6];
            params.biasValue = "[" + params.biasValue + "]";
        }

        str = match.suffix().str(); /// Continue to search for the next match
    }

    return params;
}

std::vector<FilterSubNode> SVFNN::parse_filters(const std::string &s, u32_t num_filters,  u32_t kernel_height,
                                          u32_t kernel_width,  u32_t kernel_depth)  {
    std::vector<std::vector<std::vector<std::vector<double>>>> data(num_filters, std::vector<std::vector<std::vector<double>>>(kernel_depth, std::vector<std::vector<double>>(kernel_height, std::vector<double>(kernel_width))));

    std::stringstream ss(s);
    char ch;
    /// Skip all characters before the first digit
    while (ss >> ch && !isdigit(ch) && ch != '-' && ch != '+') {
        /// Intentionally empty; just to find the start
    }
    ss.unget(); /// Add the non-number into stream

    for (u32_t n = 0; n < num_filters; ++n) {
        for (u32_t d = 0; d < kernel_depth; ++d) {
            for (u32_t h = 0; h < kernel_height; ++h) {
                for (u32_t w = 0; w < kernel_width; ++w) {
                    if (!(ss >> data[n][d][h][w])) {
                        throw std::runtime_error("Error parsing input string: unexpected format");
                    }
                    /// Skip all characters from the numerical value until the next numerical value
                    while (ss >> ch && !isdigit(ch) && ch != '-' && ch != '+') {
                        /// Intentionally empty
                    }
                    ss.unget(); /// Put non numeric characters into a reflow to prepare for reading the next number
                }
            }
        }
    }
    std::vector<FilterSubNode> filters;
    for (u32_t n = 0; n < num_filters; ++n) {
        Matrices matrices;
        for (u32_t d = 0; d < kernel_depth; ++d) {
            Mat mat(kernel_height, kernel_width);
            for (u32_t h = 0; h < kernel_height; ++h) {
                for (u32_t w = 0; w < kernel_width; ++w) {
                    mat(h, w) = data[n][d][h][w];
                }
            }
            matrices.push_back(mat);
        }
        IntervalMatrices intervalmatrices = convertMatricesToIntervalMatrices(matrices);
        filters.emplace_back(matrices, intervalmatrices);
    }
    return filters;
}

std::vector<double> SVFNN::parse_Convbiasvector(std::string s) {
    /// Check and remove the [] around the vector
    if (s.front() == '[') {
        s.erase(0, 1);
    }
    if (s.back() == ']') {
        s.erase(s.end() - 1, s.end());
    }

    std::stringstream ss(s);
    std::string tok;
    std::vector<double> elems;

    /// Split s on commas
    while (getline(ss, tok, ',')) {
        /// Remove spaces
        tok.erase(remove_if(tok.begin(), tok.end(), isspace), tok.end());

        try {
            double val = std::stod(tok);
            elems.push_back(val);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << '\n';
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << '\n';
        }
    }

    Vector b(elems.size());
    for (size_t i = 0; i < elems.size(); i++) {
        b(i) = elems[i];
    }
    return elems;
}





