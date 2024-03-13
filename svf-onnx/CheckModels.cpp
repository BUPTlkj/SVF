////
//// Created by ������ on 2024/3/7.
////
//
#include "CheckModels.h"

std::pair<SVF::LabelVector, SVF::MatrixVector_3c> SVF::LoadData::read_dataset(){

    // mnist 1*28*28
    // cifar10 1*3*32*32
    LabelVector labels;
    MatrixVector_3c matrixes_3c;

    if(dataset == "mnist"){
        // ���Լ��е�ÿһ������
        std::string line;
        // ע�����·������������
        std::ifstream file_mnist("/Users/liukaijie/CLionProjects/TestforAI/dataset/mnist_test.csv");
        // �ж��Ƿ��������Ĵ����ݼ�
        if(!file_mnist.is_open()){
            std::cerr << "Error opening file" << std::endl;
            assert(file_mnist.is_open());
        }

        //��ȡ��������
        // û���������� todo
        while(getline(file_mnist, line)){
            std::stringstream ss(line);
            std::string value;

            // �����ǩ format: label, pixel...(0-255)
            getline(ss,value,',');
            // convert into integer
            signed label = std::stoi(value);
            // label set
            labels.push_back(label);

            /// mnist 1*28*28
            std::vector<Eigen::MatrixXd> image_matrix(1, Eigen::MatrixXd(28, 28));
            unsigned channel = 0, row = 0, col = 0;

            // ͼƬ
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
        // ���Լ��е�ÿһ������
        std::string line;
        // ע�����·��
        std::ifstream file_cifar("/Users/liukaijie/CLionProjects/TestforAI/dataset/cifar10_test.csv");
        // �ж��Ƿ��������Ĵ����ݼ�
        if(!file_cifar.is_open()){
            std::cerr << "Error opening file" << std::endl;
            assert(file_cifar.is_open());
        }

        // ��mnist��ͬ
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
//// ˵������cpp��Ҫ��ʵ�����ݼ��Ķ�ȡ���Ŷ�
//// �汾2024.01.29 Ŀǰֻ֧��mnist���ݼ���ȡ�Ŷ��� cifar10������Ӧ�������ͨ������Ϊ������ͬ��δʹ�ã�
//// ͬʱע�����ݼ��е�·�����Ŷ�Ŀǰֻ���mnist��������Ҫ�Ľ�->��������һά���������һά�����Ŷ��������ƺ����������ؾ���
//// �д�˼�����Ƿ���Ҫ����һά����ֱ��1dim�ڵ����
//// 2.1 ����ͨ��
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
// * ��������Ҫ�Ƕ��������ݵļ�飺
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
// * todo ���Layer������Ŀǰֻ֧�ֺڰ�ͼ��mnist���ݼ���������Ҫ�޸�Layer.h, layer.cpp,ʹ��һά��������
// */
//// ��ǩ
//typedef std::vector<signed> LabelVector;
//// ͼ������ؾ���
//typedef std::vector<Eigen::MatrixXd> MatrixVector;
//typedef std::vector<std::vector<Eigen::MatrixXd>> MatrixVector_3c;
//
//// �������ݼ������ǰ100������
////std::pair<LabelVector, MatrixVector> read_dataset(const std::string dataset_name){
//std::pair<LabelVector, MatrixVector_3c> read_dataset(const std::string dataset_name){
//    // mnist 1*28*28
//    // cifar10 1*3*32*32
//    LabelVector labels;
//    MatrixVector matrices;
//    MatrixVector_3c matrixes_3c;
//
//    if(dataset_name == "mnist"){
//        // ���Լ��е�ÿһ������
//        std::string line;
//        // ע�����·������������
//        std::ifstream file_mnist("/Users/liukaijie/CLionProjects/TestforAI/dataset/mnist_test.csv");
//        // �ж��Ƿ��������Ĵ����ݼ�
//        if(!file_mnist.is_open()){
//            std::cerr << "Error opening file" << std::endl;
//            assert(file_mnist.is_open());
//        }
//
//        //��ȡ��������
//        // û���������� todo
//        while(getline(file_mnist, line)){
//            std::stringstream ss(line);
//            std::string value;
//
//            // �����ǩ format: label, pixel...(0-255)
//            getline(ss,value,',');
//            // convert into integer
//            signed label = std::stoi(value);
//            // label set
//            labels.push_back(label);
//
//            // ͼƬ
//            Eigen::MatrixXd matrix(28, 28);
//            for(unsigned row=0;row<28;++row){
//                for(unsigned col=0;col<28;col++){
//                    getline(ss,value,',');
//                    // ���ش���
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
//        // ���Լ��е�ÿһ������
//        std::string line;
//        // ע�����·��
//        std::ifstream file_cifar("/Users/liukaijie/CLionProjects/TestforAI/dataset/cifar10_test.csv");
//        // �ж��Ƿ��������Ĵ����ݼ�
//        if(!file_cifar.is_open()){
//            std::cerr << "Error opening file" << std::endl;
//            assert(file_cifar.is_open());
//        }
//
//        // ��mnist��ͬ
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
//        ///todo ϸ�ֵ�����������������չ��һά�������� Ŀǰֻ��mnist
//    }
//    // ��Ϊδ����cifar10���˴���ֹ���뾯��
//    return {};
//}
//
///*
// * �Ŷ�ͼ���ÿһ������, ������½�
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
//    //����Ŷ����
//    std::vector<LabelAndBounds> result;
//
//    for (size_t i = 0; i < labelImagePairs.first.size(); ++i) {
//        // ��ǩ
//        signed label = labelImagePairs.first[i];
//        // ���ؾ���
//        const Eigen::MatrixXd& originalMatrix = labelImagePairs.second[i];
//
//        //����Ŷ�
//        Eigen::MatrixXd matrix_lb = originalMatrix.array() - eps;
//        Eigen::MatrixXd matrix_ub = originalMatrix.array() + eps;
//
//        // �޶���Χ [0, 1]
//        matrix_lb = matrix_lb.cwiseMax(0.0).cwiseMin(1.0);
//        matrix_ub = matrix_ub.cwiseMax(0.0).cwiseMin(1.0);
//
//        result.push_back({label, matrix_lb, matrix_ub});
//    }
//    // ���ɴ���֤�����½�
//    return result;
//}