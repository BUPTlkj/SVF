#include "NNLoaddata.h"

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
            u32_t label = std::stoi(value);
            /// label set
            labels.push_back(label);

            /// mnist 1*28*28
            Matrices image_matrix(1, Mat(28, 28));
            u32_t channel = 0, row = 0, col = 0;

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
            /// For Test, control the number of data
            if(matrixes_3c.size()==data_num){
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
            u32_t label = std::stoi(value);
            labels.push_back(label);

            /// cifar 10, 3*32*32
            Matrices image_matrix(3, Mat(32, 32));
            u32_t channel = 0, row = 0, col = 0;

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
            /// For Test, control the number of data
            if(matrixes_3c.size()==data_num){
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

    for (u32_t i = 0; i < labelImagePairs.first.size(); ++i) {
        u32_t label = labelImagePairs.first[i];
        const Matrices& originalMatrix = labelImagePairs.second[i];

        Matrices matrix_lb;
        Matrices matrix_ub;

        for (const auto& matrix : originalMatrix) {
            Mat lb = matrix.array() - eps;
            Mat ub = matrix.array() + eps;

            lb = lb.cwiseMax(0.0).cwiseMin(1.0);
            ub = ub.cwiseMax(0.0).cwiseMin(1.0);

            matrix_lb.push_back(lb);
            matrix_ub.push_back(ub);
        }

        result.push_back({label, matrix_lb, matrix_ub});
    }
    return result;
}

std::vector<std::pair<u32_t , IntervalMatrices>> LoadData::convertLabelAndBoundsToIntervalMatrices(const std::vector<LabelAndBounds>& labelAndBoundsVec) {
    std::vector<std::pair<u32_t , IntervalMatrices>> result;

    for (const auto& labelAndBounds : labelAndBoundsVec) {
        IntervalMatrices intervalMatrices;

        /// ensure each LabelAndBounds: matrix_lb & matrix_ub have the same size
        assert(labelAndBounds.matrix_lb.size() == labelAndBounds.matrix_ub.size());

        /// Traversal matrix_lb & matrix_ub£¬create an pad IntervalMat
        for (u32_t i = 0; i < labelAndBounds.matrix_lb.size(); ++i) {
            const Mat& lb = labelAndBounds.matrix_lb[i];
            const Mat& ub = labelAndBounds.matrix_ub[i];

            IntervalMat intervalMat(lb.rows(), lb.cols());

            for (u32_t r = 0; r < lb.rows(); ++r) {
                for (u32_t c = 0; c < lb.cols(); ++c) {
                    intervalMat(r, c) = IntervalValue(lb(r, c), ub(r, c));
                }
            }

            intervalMatrices.push_back(intervalMat);
        }

        result.emplace_back(labelAndBounds.label, intervalMatrices);
    }

    return result;
}
