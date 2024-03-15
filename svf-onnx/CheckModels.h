

#ifndef SVF_CHECKMODELS_H
#define SVF_CHECKMODELS_H

#include "iostream"
#include "Eigen/Dense"
#include <fstream>
#include <sstream>


namespace SVF
{

/// Label
typedef std::vector<signed> LabelVector;

/// Image pixel matrix
typedef std::vector<std::vector<Eigen::MatrixXd>> MatrixVector_3c;

/// Upper & lower Bound
struct LabelAndBounds
{
    signed label;
    std::vector<Eigen::MatrixXd> matrix_lb;
    std::vector<Eigen::MatrixXd> matrix_ub;
};

class LoadData
{
public:
    std::string dataset;

public:
    LoadData(std::string name) : dataset{name}
    {
        if (dataset == "mnist" or dataset == "cifar")
        {
            std::cout << "Loading " << dataset << " ......" << std::endl;
        }
        else
        {
            throw std::runtime_error(
                "UNSUPPORT DATASET, ONLY SUPPORT MNIST AND CIFAR 10");
        }
    }

    std::pair<LabelVector, MatrixVector_3c> read_dataset();

    /// Perturbation function
    std::vector<LabelAndBounds> perturbateImages( const std::pair<LabelVector, MatrixVector_3c>& labelImagePairs, double eps);

};
}



#endif // SVF_CHECKMODELS_H
