#include "CheckModels.h"

using namespace SVF;

std::pair<LabelVector, MatrixVector_3c> LoadData::read_dataset(){

    /// mnist 1*28*28
    /// cifar-10 1*3*32*32
    LabelVector labels;
    MatrixVector_3c matrixes_3c;

    if(dataset == "mnist"){
        std::string line;
        /// THIS PATH!!! ATTENTION
        std::ifstream file_mnist("/Users/liukaijie/CLionProjects/TestforAI/dataset/mnist_test.csv");

        if(!file_mnist.is_open()){
            std::cerr << "Error opening file" << std::endl;
            assert(file_mnist.is_open());
        }

        /// Read pixel
        while(getline(file_mnist, line)){
            std::stringstream ss(line);
            std::string value;

            ///  format: label, pixel...(0-255)
            getline(ss,value,',');
            /// convert into integer
            signed label = std::stoi(value);
            /// label set
            labels.push_back(label);

            /// mnist 1*28*28
            std::vector<Eigen::MatrixXd> image_matrix(1, Eigen::MatrixXd(28, 28));
            unsigned channel = 0, row = 0, col = 0;

            /// Image Matrix
            while(getline(ss,value,',')) {
                image_matrix[channel](row, col) = std::stof(value)/(255.0);
                if(++col == 28) {
                    col = 0;
                    row++;
                    if(row==28) {
                        row = 0;
                        channel++;
                    }
                }
            }

            matrixes_3c.push_back(image_matrix);
            /// For Test
            if(matrixes_3c.size()==1){
                break;
            }
        }

        file_mnist.close();

    }else{
        std::string line;
        /// THIS PATH!!!!
        std::ifstream file_cifar("/Users/liukaijie/CLionProjects/TestforAI/dataset/cifar10_test.csv");

        if(!file_cifar.is_open()){
            std::cerr << "Error opening file" << std::endl;
            assert(file_cifar.is_open());
        }

        while(getline(file_cifar, line)) {
            std::stringstream ss(line);
            std::string value;

            getline(ss, value, ',');
            signed label = std::stoi(value);
            labels.push_back(label);

            /// cifar10, 3*32*32
            std::vector<Eigen::MatrixXd> image_matrix(3, Eigen::MatrixXd(32, 32));
            unsigned channel = 0, row = 0, col = 0;

            while(getline(ss,value,',')) {
                image_matrix[channel](row, col) = std::stof(value)/(255.0);
                if(++col == 32) {
                    col = 0;
                    row++;
                    if(row==32) {
                        row = 0;
                        channel++;
                    }
                }
            }
            matrixes_3c.push_back(image_matrix);
            /// For Test
            if(matrixes_3c.size()==1){
                break;
            }
        }
        file_cifar.close();

    }
    return std::make_pair(labels, matrixes_3c);
}

std::vector<LabelAndBounds> LoadData::perturbateImages(
    const std::pair<LabelVector, MatrixVector_3c>& labelImagePairs,
    double eps) {

    std::vector<LabelAndBounds> result;

    for (size_t i = 0; i < labelImagePairs.first.size(); ++i) {
        signed label = labelImagePairs.first[i];
        const std::vector<Eigen::MatrixXd>& originalMatrix = labelImagePairs.second[i];

        std::vector<Eigen::MatrixXd> matrix_lb;
        std::vector<Eigen::MatrixXd> matrix_ub;

        for (const auto& matrix : originalMatrix) {
            Eigen::MatrixXd lb = matrix.array() - eps;
            Eigen::MatrixXd ub = matrix.array() + eps;

            lb = lb.cwiseMax(0.0).cwiseMin(1.0);
            ub = ub.cwiseMax(0.0).cwiseMin(1.0);

            matrix_lb.push_back(lb);
            matrix_ub.push_back(ub);
        }

        result.push_back({label, matrix_lb, matrix_ub});
    }
    return result;
}