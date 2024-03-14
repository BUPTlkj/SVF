#include "SVFONNX.h"
#include "CheckModels.h"
#include "algorithm" /// For std::remove
/// Parse function to extract integers from a given string and return their
/// vectors


std::vector<int> parseDimensions(const std::string& input) {
    std::vector<int> dimensions;
    std::stringstream ss(input);
    int number;
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
    /// 22 Feb
    std::string MaxPool = "MaxPool";
    /// The information of Maxpool and conv is in SpecialInfo
    std::string SepcialInfo = "SepcialInfo";

    auto cppMapa = callPythonFunction(onnxAdress, "InteractwithCpp");

/// Put it here?? Need for optimization todo
    ConstantNodeInfo constantnode;
    BasicNodeInfo basicnode;
    ParsedGEMMParams gemmnode;
    ConvNodeInfo convnode;
    ReluNodeInfo relunode;
    MaxPoolNodeInfo maxpoolnode;

    std::map<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>> convItems;
    std::map<std::string, std::pair<std::pair<int, int>, std::pair<std::pair<int, int>, std::pair<int, int>>>>  maxPoolItems;

    for(const auto& pair : cppMapa) {
        /// Key: pair.first  Value: pair.second
        std::string name = pair.first;
        auto nodeDataParts = parseNodeData(pair.second);
        for (size_t i = 0; i < nodeDataParts.size(); ++i) {
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

        for(size_t i=0; i<nodeDataParts.size(); ++i){

            if(i==3){ /// The third one is specific dimensions and data
                if (name.find(Constant) != std::string::npos){
                    constantnode.name = name;
                    nodes.push_back(constantnode);
                }else if(name.find(Sub) != std::string::npos || name.find(Add) != std::string::npos || name.find(Mul) != std::string::npos || name.find(Div) != std::string::npos){
                    basicnode = parseBasicNodeString(nodeDataParts[2]);
                    basicnode.typestr = nodeDataParts[0];
                    basicnode.name = name;
                    nodes.push_back(basicnode);
                }else if(name.find(Gemm) != std::string::npos){
                    gemmnode = GEMMparseAndFormat(nodeDataParts[2]);
                    gemmnode.gemmName = name;
                    Eigen::MatrixXd weight = restoreGEMMWeightToMatrix(gemmnode.weightDimensions, gemmnode.weightValues);
                    Eigen::MatrixXd bias = restoreGEMMBiasMatrixFromStrings(gemmnode.biasDimensions, gemmnode.biasValues);
                    gemmnode.weight = weight;
                    gemmnode.bias = bias;
                    nodes.push_back(gemmnode);
                }else if(name.find(Conv) != std::string::npos){
                    /// Refer to GEMM
                    ConvParams conf = ConvparseAndFormat(nodeDataParts[2]);
                    /// filter: filters,tmr todo dims
                    std::cout<<"**********"<<conf.filterDims<<std::endl<<conf.filterValue<<std::endl;
                    convnode.filter = parse_filters(conf.filterValue, parseDimensions(conf.filterDims)[0], parseDimensions(conf.filterDims)[1], parseDimensions(conf.filterDims)[2], parseDimensions(conf.filterDims)[3] );
                    convnode.conbias = parse_Convbiasvector(conf.biasValue);
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

std::map<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>> SVFNN::parseConvItems(const std::vector<std::string> &items) {
    std::map<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>> result;
    for (const auto& item : items) {
        auto key_value = parseItem(item);
        if (key_value.first.find("Conv") != std::string::npos) {
            /// if value's format is correct
            int px, py, px_, py_, sx, sy;
            sscanf(key_value.second.c_str(), " ['pads:', [%d, %d, %d, %d], 'strides:', [%d, %d]]", &px, &py, &px_, &py_, &sx, &sy); // 解析字符串为两个整数
            result[key_value.first] = std::make_pair(std::make_pair(px, py), std::make_pair(sx, sy));
        }
    }
    return result;
}

std::map<std::string, std::pair<std::pair<int, int>, std::pair<std::pair<int, int>, std::pair<int, int>>>>
SVFNN::parseMaxPoolItems(const std::vector<std::string> &items) {
    std::map<std::string, std::pair<std::pair<int, int>, std::pair<std::pair<int, int>, std::pair<int, int>>>> result;
    for (const auto& item : items) {
        auto key_value = parseItem(item);
        if (key_value.first.find("MaxPool") != std::string::npos) {
            /// if value's format is correct
            int wx, wy, px, py, px_, py_, sx, sy;
            sscanf(key_value.second.c_str(), " ['Windows:', [%d, %d], 'pads:', [%d, %d, %d, %d], 'strides:', [%d, %d]]", &wx, &wy, &px, &py,  &px_, &py_, &sx, &sy); // 解析字符串为四个整数
            result[key_value.first] = std::make_pair(std::make_pair(wx, wy), std::make_pair(std::make_pair(px, py), std::make_pair(sx, sy)));
        }
    }
    return result;
}

std::vector<SVFNeuralNet> SVFNN::get_nodes(){
    return nodes;
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

    PyRun_SimpleString("import os, sys");
//    PyRun_SimpleString("current_file_path = os.path.dirname(os.path.abspath(__file__))");
//    PyRun_SimpleString("PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))");
//    PyRun_SimpleString("current_file_path = os.path.dirname(PROJECT_ROOT)");
//    PyRun_SimpleString("sys.path.append(current_file_path)");
    // todo
    PyRun_SimpleString("sys.path.append('/Users/liukaijie/CLionProjects/ReadPython')");

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
    startPos = dataStr.find_first_of('\'', pos) + 1; // Skip the first quotation mark
    endPos = dataStr.find_first_of('\'', startPos);
    parts.push_back(dataStr.substr(startPos, endPos - startPos));

    /// Part 2: Find the first list
    startPos = dataStr.find_first_of('[', endPos) + 1; // Skip the first [
    endPos = dataStr.find_first_of(']', startPos);
    parts.push_back(dataStr.substr(startPos, endPos - startPos));

    /// Part 3: Searching for a Dictionary
    startPos = dataStr.find_first_of('{', endPos) + 1; // Skip the first {
    endPos = dataStr.find_first_of('}', startPos);
    parts.push_back(dataStr.substr(startPos, endPos - startPos));

    /// Part 4: Find the second list
    startPos = dataStr.find_first_of('[', endPos + 1) + 1; // Skip the second [
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
            info.values.push_back(Eigen::MatrixXd::Constant(1, 1, value));
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
ParsedGEMMParams SVFNN::GEMMparseAndFormat(const std::string& input) {
    ParsedGEMMParams params;
    std::regex re("'(.*?)': \\[\\((.*?)\\), \\[(.*?)\\]\\]");

    std::smatch match;
    std::string str = input;

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

Eigen::MatrixXd SVFNN::restoreGEMMWeightToMatrix(const std::string& dimensions, const std::string& values) {
    /// Analyze dimensions
    std::istringstream dimStream(dimensions);
    int rows, cols;
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
        return Eigen::MatrixXd(0, 0); // 返回一个空矩阵
    }

    /// Padding matrix
    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = vals[i * cols + j];
        }
    }

    return matrix;
}

Eigen::VectorXd SVFNN::restoreGEMMBiasMatrixFromStrings(const std::string& dimensionStr, const std::string& valuesStr) {
    /// Processing dimension strings, removing commas
    std::string dimStr = dimensionStr;
    if (!dimStr.empty() && dimStr.back() == ',') {
        dimStr.pop_back(); // Remove the last character (comma)
    }

    /// Parsing dimension strings
    int rows = 1;
    int cols;
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
        return Eigen::MatrixXd(); // Return an empty matrix
    }

    /// Fill the matrix with parsed values
    Eigen::MatrixXd matrix(rows, cols);
    for (int j = 0; j < cols; ++j) {
        matrix(0, j) = values[j];
    }
    Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(matrix.data(), matrix.size());
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

std::vector<SVF::FilterSubNode> SVFNN::parse_filters(const std::string &s, unsigned int num_filters, unsigned int kernel_height,
                                         unsigned int kernel_width, unsigned int kernel_depth)  {
    std::vector<std::vector<std::vector<std::vector<double>>>> data(num_filters, std::vector<std::vector<std::vector<double>>>(kernel_depth, std::vector<std::vector<double>>(kernel_height, std::vector<double>(kernel_width))));

    std::stringstream ss(s);
    char ch;
    /// Skip all characters before the first digit
    while (ss >> ch && !isdigit(ch) && ch != '-' && ch != '+') {
        /// Intentionally empty; just to find the start
    }
    ss.unget(); /// Add the non-number into stream

    for (unsigned n = 0; n < num_filters; ++n) {
        for (unsigned d = 0; d < kernel_depth; ++d) {
            for (unsigned h = 0; h < kernel_height; ++h) {
                for (unsigned w = 0; w < kernel_width; ++w) {
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

    std::vector<SVF::FilterSubNode> filters;
    for (unsigned n = 0; n < num_filters; ++n) {
        std::vector<Eigen::MatrixXd> matrices;
        for (unsigned d = 0; d < kernel_depth; ++d) {
            Eigen::MatrixXd mat(kernel_height, kernel_width);
            for (unsigned h = 0; h < kernel_height; ++h) {
                for (unsigned w = 0; w < kernel_width; ++w) {
                    mat(h, w) = data[n][d][h][w];
                }
            }
            matrices.push_back(mat);
        }
        filters.emplace_back(matrices);
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

    Eigen::VectorXd b(elems.size());
    for (size_t i = 0; i < elems.size(); i++) {
        b(i) = elems[i];
    }

    return elems;
}


/// NNGraph Bulid
class NNGraphBuilder {
private:
    /// init nodes
    std::unordered_map<std::string, SVF::ConstantNeuronNode*> ConstantNodeIns;
    std::unordered_map<std::string, SVF::ConvNeuronNode*> ConvNodeIns;
    std::unordered_map<std::string, SVF::ReLuNeuronNode*> ReLuNodeIns;
    std::unordered_map<std::string, SVF::MaxPoolNeuronNode*> MaxPoolNodeIns;
    std::unordered_map<std::string, SVF::FullyConNeuronNode*> FullyConNodeIns;
    std::unordered_map<std::string, SVF::BasicOPNeuronNode*> BasicOPNodeIns;
    /// Node's class name for visitors
    std::vector<std::string> OrderedNodeName;
    std::vector<std::unique_ptr<SVF::NeuronNode>> Nodeins;
    /// init edges
    std::vector<std::unique_ptr<SVF::Direct2NeuronEdge>> edges;
    /// init Graph
    SVF::NeuronNet *g = new SVF::NeuronNet();

public:
    /// Allocate the NodeID
    inline unsigned getNodeID(const std::string& str) {
        size_t underscorePos = str.find('_'); /// Find the location of "_"
        if (underscorePos == std::string::npos) {
            throw std::invalid_argument("NodeID has been not allocated!");
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
        unsigned number = std::stoi(numberStr);
        return number;
    }

    /// Thoese operator() is designed for collecting instance
    void operator()(const ConstantNodeInfo& node) {
        unsigned id = getNodeID(node.name);
        OrderedNodeName.push_back(node.name);
        ConstantNodeIns[node.name] = new SVF::ConstantNeuronNode(id);
        g->addConstantNeuronNode(ConstantNodeIns[node.name]);
    }

    void operator()(const BasicNodeInfo& node)  {
        auto id = getNodeID(node.name);
        OrderedNodeName.push_back(node.name);
        BasicOPNodeIns[node.name] = new SVF::BasicOPNeuronNode(id, node.typestr, node.values);
        g->addBasicOPNeuronNode(BasicOPNodeIns[node.name]);
    }

    void operator()(const ParsedGEMMParams& node)  {
        auto id = getNodeID(node.gemmName);
        OrderedNodeName.push_back(node.gemmName);
        FullyConNodeIns[node.gemmName] = new SVF::FullyConNeuronNode(id, node.weight, node.bias);
        g->addFullyConNeuronNode(FullyConNodeIns[node.gemmName]);
    }

    void operator()(const ConvNodeInfo& node) {
        auto id = getNodeID(node.name);
        OrderedNodeName.push_back(node.name);
        ConvNodeIns[node.name] = new SVF::ConvNeuronNode(id, node.filter, node.conbias, node.pads.first, node.strides.first);
        g->addConvNeuronNode(ConvNodeIns[node.name]);

        for(size_t ip = 0; ip < node.filter.size(); ++ip) {
            /// Using filter[i]
            const SVF::FilterSubNode &subNode = node.filter[ip];
            for(size_t ipp = 0; ipp < subNode.value.size(); ipp++){
                std::cout<<"Filter: "<<ip<<" - Matrix: "<<ipp<<std::endl;
                std::cout<<subNode.value[ipp]<<std::endl;
            }
        }

    }

    void operator()(const ReluNodeInfo& node) {
        auto id = getNodeID(node.name);
        OrderedNodeName.push_back(node.name);
        ReLuNodeIns[node.name] = new SVF::ReLuNeuronNode(id);
        g->addReLuNeuronNode(ReLuNodeIns[node.name]);
    }

    void operator()(const MaxPoolNodeInfo& node) {
        auto id = getNodeID(node.name);
        OrderedNodeName.push_back(node.name);
        MaxPoolNodeIns[node.name] = new SVF::MaxPoolNeuronNode(id, node.windows.first, node.windows.second, node.strides.first, node.strides.second, node.pads.first, node.pads.second);
        g->addMaxPoolNeuronNode(MaxPoolNodeIns[node.name]);
    }

    SVF::NeuronNodeVariant getNodeInstanceByName(const std::string& name) const {
        if (auto it = ConstantNodeIns.find(name); it != ConstantNodeIns.end()) return it->second;
        if (auto it = ConvNodeIns.find(name); it != ConvNodeIns.end()) return it->second;
        if (auto it = ReLuNodeIns.find(name); it != ReLuNodeIns.end()) return it->second;
        if (auto it = MaxPoolNodeIns.find(name); it != MaxPoolNodeIns.end()) return it->second;
        if (auto it = FullyConNodeIns.find(name); it != FullyConNodeIns.end()) return it->second;
        if (auto it = BasicOPNodeIns.find(name); it != BasicOPNodeIns.end()) return it->second;
        return std::monostate{};
    }


    SVF::NeuronNode* getNodeInstanceByName1(const std::string& name) const {
        if (auto it = ConstantNodeIns.find(name); it != ConstantNodeIns.end()) return it->second;
        if (auto it = ConvNodeIns.find(name); it != ConvNodeIns.end()) return it->second;
        if (auto it = ReLuNodeIns.find(name); it != ReLuNodeIns.end()) return it->second;
        if (auto it = MaxPoolNodeIns.find(name); it != MaxPoolNodeIns.end()) return it->second;
        if (auto it = FullyConNodeIns.find(name); it != FullyConNodeIns.end()) return it->second;
        if (auto it = BasicOPNodeIns.find(name); it != BasicOPNodeIns.end()) return it->second;
        return nullptr;
    }

    bool isValidNode(const SVF::NeuronNodeVariant& node) {
        return !std::holds_alternative<std::monostate>(node);
    }

    void AddEdges() {

        for (size_t i = 0; i < OrderedNodeName.size() - 1; ++i) {
            const auto& currentName = OrderedNodeName[i];
            const auto& nextName = OrderedNodeName[i + 1];

            SVF::NeuronNode* currentNode = getNodeInstanceByName1(currentName);
            SVF::NeuronNode* nextNode = getNodeInstanceByName1(nextName);

            if (currentNode && nextNode) {
                /// Ensure edge is created as a unique_ptr<SVF::Direct2NeuronEdge>
                auto edge = std::make_unique<SVF::Direct2NeuronEdge>(currentNode, nextNode);
                edges.push_back(std::move(edge)); // This should now work
            }
        }

        for (const auto& edge : edges){
            g->addDirected2NodeEdge(edge.get());
        }

    }

    void Traversal(std::vector<Eigen::MatrixXd>& in_x) {

        /// Print the dataset matrix
        for(int j=0; j<in_x.size();j++){
            std::cout<<"Matrix: "<<j<<std::endl;
            std::cout<<in_x[j]<<std::endl;
        }

        /// Note: Currently, visited and path store pointers to SVF:: NeuronNodeVariant
        std::set<const SVF::NeuronNode *> visited;
        std::vector<const SVF::NeuronNode *> path;
        auto *dfs = new SVF::GraphTraversal();


        const auto& LastName = OrderedNodeName[OrderedNodeName.size() - 1];
        const auto& FirstName = OrderedNodeName[0];

        /// getNodeInstanceByName() return type: SVF::NeuronNodeVariant
        auto FirstNode = getNodeInstanceByName(FirstName); /// Return SVF::NeuronNodeVariant
        auto LastNode = getNodeInstanceByName(LastName); /// Return SVF::NeuronNodeVariant

        /// Due to DFS now accepting parameters: SVF:: NeuronNodeVariant type, directly passing the addresses of FirstNode and LastNode
        dfs->DFS(visited, path, &FirstNode, &LastNode, in_x);
        auto stringPath = dfs->getPaths();
        std::cout<<"GET PATH"<<stringPath.size()<<std::endl;
        int i = 0;
        for (const std::string& paths : stringPath) {
            std::cout << i <<"*****"<< paths << std::endl;
            i++;
        }

        delete dfs; /// Delete allocated memory
    }
};


int main(){

//    std::string address = "/Users/liukaijie/Desktop/operation-py/convSmallRELU__Point.onnx";
//    std::string address = "/Users/liukaijie/Desktop/operation-py/mnist_conv_maxpool.onnx";
    std::string address = "/Users/liukaijie/Desktop/operation-py/ffnnRELU__Point_6_500.onnx";

    /// parse onnx into svf-onnx
    SVFNN svfnn(address);
    auto nodes = svfnn.get_nodes();

    /// Init nn-graph builder
    NNGraphBuilder nngraph;

    /// Init & Add node
    for (const auto& node : nodes) {
        std::visit(nngraph, node);
    }

    /// Init & Add Edge
    nngraph.AddEdges();

    /// Load dataset: mnist or cifa-10
//    SVF::LoadData dataset("cifar");
    SVF::LoadData dataset("mnist");
    auto x = dataset.read_dataset();
    std::cout<<"Label: "<<x.first.front()<<std::endl;

    double perti = 0.001;
    auto per_x = dataset.perturbateImages(x, perti);

    /// Run abstract interpretation on NNgraph
    nngraph.Traversal(x.second.front());

    return 0;
}
