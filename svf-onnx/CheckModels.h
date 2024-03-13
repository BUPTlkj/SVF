

#ifndef SVF_CHECKMODELS_H
#define SVF_CHECKMODELS_H

#include "iostream"
#include "Eigen/Dense"
#include <fstream>
#include <sstream>


namespace SVF{


/// ±êÇ©
typedef std::vector<signed> LabelVector;
/// Í¼ÏñµÄÏñËØ¾ØÕó
typedef std::vector<Eigen::MatrixXd> MatrixVector;
typedef std::vector<std::vector<Eigen::MatrixXd>> MatrixVector_3c;

class LoadData{
public:
    std::string dataset;

public:
    LoadData(std::string name): dataset{name}{
        if(dataset == "mnist" or dataset == "cifar"){
            std::cout<<"Loading "<<dataset<<" ......"<<std::endl;
        }else{
            throw std::runtime_error("UNSUPPORT DATASET, ONLY SUPPORT MNIST AND CIFAR 10");
        }
    }

    std::pair<LabelVector, MatrixVector_3c> read_dataset();





};

}



#endif // SVF_CHECKMODELS_H
