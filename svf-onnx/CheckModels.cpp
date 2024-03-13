////
//// Created by 刘凯杰 on 2024/3/7.
////
//
#include "CheckModels.h"

std::pair<SVF::LabelVector, SVF::MatrixVector_3c> SVF::LoadData::read_dataset(){

    // mnist 1*28*28
    // cifar10 1*3*32*32
    LabelVector labels;
    MatrixVector_3c matrixes_3c;

    if(dataset == "mnist"){
        // 测试集中的每一个数据
        std::string line;
        // 注意这个路径！！！！！
        std::ifstream file_mnist("/Users/liukaijie/CLionProjects/TestforAI/dataset/mnist_test.csv");
        // 判断是否能正常的打开数据集
        if(!file_mnist.is_open()){
            std::cerr << "Error opening file" << std::endl;
            assert(file_mnist.is_open());
        }

        //读取像素数据
        // 没有限制行数 todo
        while(getline(file_mnist, line)){
            std::stringstream ss(line);
            std::string value;

            // 处理标签 format: label, pixel...(0-255)
            getline(ss,value,',');
            // convert into integer
            signed label = std::stoi(value);
            // label set
            labels.push_back(label);

            /// mnist 1*28*28
            std::vector<Eigen::MatrixXd> image_matrix(1, Eigen::MatrixXd(28, 28));
            unsigned channel = 0, row = 0, col = 0;

            // 图片
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
        // 测试集中的每一个数据
        std::string line;
        // 注意这个路径
        std::ifstream file_cifar("/Users/liukaijie/CLionProjects/TestforAI/dataset/cifar10_test.csv");
        // 判断是否能正常的打开数据集
        if(!file_cifar.is_open()){
            std::cerr << "Error opening file" << std::endl;
            assert(file_cifar.is_open());
        }

        // 与mnist相同
        while(getline(file_cifar, line)) {
            std::stringstream ss(line);
            std::string value;

            getline(ss, value, ',');
            signed label = std::stoi(value);
            labels.push_back(label);

            // cifar10, 3*32*32
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






//
////
//// Created by Kaijie Liu on 2024/1/29.
//// 说明：此cpp主要是实现数据集的读取及扰动
//// 版本2024.01.29 目前只支持mnist数据集读取扰动， cifar10做了相应代码测试通过但因为张量不同暂未使用，
//// 同时注意数据集中的路径，扰动目前只针对mnist，后续需要改进->函数返回一维向量，针对一维向量扰动，最后设计函数返回像素矩阵
//// 有待思考：是否不需要返回一维矩阵，直接1dim节点分析
//// 2.1 测试通过
////
//
//#include "Eigen/Dense"
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <vector>
//#include <string>
//#include <cassert>
//#include <set>
//
///*
// * 本部分主要是对输入数据的检查：
// * input dataset: mnist or cifar 10 (CVS format)
// */
//
//bool check_input(const std::string dataset_name ){
//    if(dataset_name == "mnist" or dataset_name == "cifar"){
//        return true;
//    }else{
//        std::cout<<"UNSUPPORT DATASET, ONLY SUPPORT MNIST AND CIFAR 10"<<std::endl;
//        return false;
//    }
//}
//
///*
// * input: dataset csv
// * output: one-dim vector or Matrix
// * todo 结合Layer操作，目前只支持黑白图像mnist数据集，后续需要修改Layer.h, layer.cpp,使用一维向量处理
// */
//// 标签
//typedef std::vector<signed> LabelVector;
//// 图像的像素矩阵
//typedef std::vector<Eigen::MatrixXd> MatrixVector;
//typedef std::vector<std::vector<Eigen::MatrixXd>> MatrixVector_3c;
//
//// 处理数据集，获得前100张数据
////std::pair<LabelVector, MatrixVector> read_dataset(const std::string dataset_name){
//std::pair<LabelVector, MatrixVector_3c> read_dataset(const std::string dataset_name){
//    // mnist 1*28*28
//    // cifar10 1*3*32*32
//    LabelVector labels;
//    MatrixVector matrices;
//    MatrixVector_3c matrixes_3c;
//
//    if(dataset_name == "mnist"){
//        // 测试集中的每一个数据
//        std::string line;
//        // 注意这个路径！！！！！
//        std::ifstream file_mnist("/Users/liukaijie/CLionProjects/TestforAI/dataset/mnist_test.csv");
//        // 判断是否能正常的打开数据集
//        if(!file_mnist.is_open()){
//            std::cerr << "Error opening file" << std::endl;
//            assert(file_mnist.is_open());
//        }
//
//        //读取像素数据
//        // 没有限制行数 todo
//        while(getline(file_mnist, line)){
//            std::stringstream ss(line);
//            std::string value;
//
//            // 处理标签 format: label, pixel...(0-255)
//            getline(ss,value,',');
//            // convert into integer
//            signed label = std::stoi(value);
//            // label set
//            labels.push_back(label);
//
//            // 图片
//            Eigen::MatrixXd matrix(28, 28);
//            for(unsigned row=0;row<28;++row){
//                for(unsigned col=0;col<28;col++){
//                    getline(ss,value,',');
//                    // 像素处理
//                    matrix(row,col)=std::stof(value)/(255.0);
//                }
//            }
//            matrices.push_back(matrix);
//        }
//
//        file_mnist.close();
//
//        //        return{labels, matrices};
//
//    }else{
//        // 测试集中的每一个数据
//        std::string line;
//        // 注意这个路径
//        std::ifstream file_cifar("/Users/liukaijie/CLionProjects/TestforAI/dataset/cifar10_test.csv");
//        // 判断是否能正常的打开数据集
//        if(!file_cifar.is_open()){
//            std::cerr << "Error opening file" << std::endl;
//            assert(file_cifar.is_open());
//        }
//
//        // 与mnist相同
//        while(getline(file_cifar, line)) {
//            std::stringstream ss(line);
//            std::string value;
//
//            getline(ss, value, ',');
//            signed label = std::stoi(value);
//            labels.push_back(label);
//
//            // cifar10, 3*32*32
//            std::vector<Eigen::MatrixXd> image_matrix(3, Eigen::MatrixXd(32, 32));
//            unsigned channel = 0, row = 0, col = 0;
//
//            while(getline(ss,value,',')) {
//                image_matrix[channel](row, col) = std::stof(value)/(255.0);
//                if(++col == 32) {
//                    col = 0;
//                    row++;
//                    if(row==32) {
//                        row = 0;
//                        channel++;
//                    }
//                }
//            }
//            matrixes_3c.push_back(image_matrix);
//        }
//        file_cifar.close();
//        return std::make_pair(labels, matrixes_3c);
//        ///todo 细分到两个函数，或者延展成一维的向量， 目前只做mnist
//    }
//    // 因为未启用cifar10，此处防止编译警告
//    return {};
//}
//
///*
// * 扰动图像的每一个像素, 获得上下界
// */
//
//struct LabelAndBounds {
//    signed label;
//    Eigen::MatrixXd matrix_lb;
//    Eigen::MatrixXd matrix_ub;
//};
//
//std::vector<LabelAndBounds> perturbateImages(
//    const std::pair<LabelVector, MatrixVector>& labelImagePairs,
//    double eps) {
//    //存放扰动结果
//    std::vector<LabelAndBounds> result;
//
//    for (size_t i = 0; i < labelImagePairs.first.size(); ++i) {
//        // 标签
//        signed label = labelImagePairs.first[i];
//        // 像素矩阵
//        const Eigen::MatrixXd& originalMatrix = labelImagePairs.second[i];
//
//        //添加扰动
//        Eigen::MatrixXd matrix_lb = originalMatrix.array() - eps;
//        Eigen::MatrixXd matrix_ub = originalMatrix.array() + eps;
//
//        // 限定范围 [0, 1]
//        matrix_lb = matrix_lb.cwiseMax(0.0).cwiseMin(1.0);
//        matrix_ub = matrix_ub.cwiseMax(0.0).cwiseMin(1.0);
//
//        result.push_back({label, matrix_lb, matrix_ub});
//    }
//    // 构成待验证的上下界
//    return result;
//}