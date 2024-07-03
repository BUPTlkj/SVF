

#ifndef SVF_NNLOADDATA_H
#define SVF_NNLOADDATA_H

#include "Eigen/Core"
#include "Graphs/NNNode.h"
#include "iostream"
#include "fstream"
#include "sstream"

namespace SVF
{

/// Label
typedef std::vector<u32_t> LabelVector;

/// Image pixel matrix
typedef std::vector<Matrices> MatrixVector_3c;

/// Upper & lower Bound
struct LabelAndBounds
{
    u32_t label;
    Matrices matrix_lb;
    Matrices matrix_ub;
};

class LoadData
{
public:
    std::string dataset;
    u32_t data_num;

public:
    LoadData(std::string name, u32_t num) : dataset{name}, data_num(num){
        if (dataset.find("mnist") != std::string::npos or dataset.find("cifar") != std::string::npos){
            std::cout << "Loading " << dataset << " ......" << std::endl;
        }else{
            std::cout<<"UNSUPPORT DATASET, ONLY SUPPORT MNIST AND CIFAR 10"<<std::endl;
            exit(0);
        }
    }

    std::pair<LabelVector, MatrixVector_3c> read_dataset();

    /// Perturbation function
    std::vector<LabelAndBounds> perturbateImages( const std::pair<LabelVector, MatrixVector_3c>& labelImagePairs, double eps);

    std::vector<std::pair<u32_t , IntervalMatrices>> convertLabelAndBoundsToIntervalMatrices(const std::vector<LabelAndBounds>& labelAndBoundsVec);

    std::vector<Eigen::MatrixXd> transpose_nhw_hnw(const Matrices & matrices);

    };
}


#endif // SVF_NNLOADDATA_H
