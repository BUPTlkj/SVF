

#ifndef SVF_LOADDATA_H
#define SVF_LOADDATA_H

#include "Eigen/Dense"
#include "Graphs/NNNode.h"
#include "iostream"
#include <fstream>
#include <sstream>

namespace SVF
{

/// Label
typedef std::vector<signed> LabelVector;

/// Image pixel matrix
typedef std::vector<Matrices> MatrixVector_3c;

/// Upper & lower Bound
struct LabelAndBounds
{
    signed label;
    Matrices matrix_lb;
    Matrices matrix_ub;
};

class LoadData
{
public:
    std::string dataset;

public:
    LoadData(std::string name) : dataset{name}
    {
        if (dataset == "mnist" or dataset == "cifar"){
            std::cout << "Loading " << dataset << " ......" << std::endl;
        }else{
            throw std::runtime_error(
                "UNSUPPORT DATASET, ONLY SUPPORT MNIST AND CIFAR 10");
        }
    }

    std::pair<LabelVector, MatrixVector_3c> read_dataset();

    /// Perturbation function
    std::vector<LabelAndBounds> perturbateImages( const std::pair<LabelVector, MatrixVector_3c>& labelImagePairs, double eps);

};
}



#endif // SVF_LOADDATA_H
