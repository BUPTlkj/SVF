#ifndef SVF_INTERVALSOLVER_H
#define SVF_INTERVALSOLVER_H
#include "../svf/include/AE/Core/IntervalValue.h"
#include "Eigen/Dense"
#include "../svf/include/Graphs//NNNode.h"

/// 1.��Ҫ���������õ�weight��bias��filterȫ��ת��ΪInterval����
/// 2.��Ҫ��ConvNode��MaxPoolNode��BasicNode��ConstantNode��FilterNode��ReluNode������ȫ����ΪInterval�����������get�����ع�
/// 3.���ڻ�����NeuronNode����Ҫ�ع���������
/// 4.����ͼ�ļ�����Ҫ�ع��м��������Ĵ���

///*********************
/// * 1.����matrix->IntervalMatrix
/// * 2.�����������м�ֵ��std::vector<Eigen::Matrix> IRRes -> IntervalMatrix IRRes
/// 3.ÿ���ڵ㴦��Ĺ����н��ڵ��std::vector<Eigen::Matrix> -> IntervalMatrix
/// 4.��solver�������



namespace SVF
{


class IntervalSolver
{
public:

    /// ��������ؾ���
    std::vector<Eigen::MatrixXd> data_matrix;
    /// ת��ΪInterval�����ؾ���
    IntervalMatrices interval_data_matrix;
    /// ÿ��solver���������
    IntervalMatrices in_x;

    /// ���캯���� ����������ؾ���, ת��Ϊ IntervalMatrix
    IntervalSolver(const std::vector<Eigen::MatrixXd>& in): data_matrix(in){
              interval_data_matrix = convertMatricesToIntervalMatrices(data_matrix);
          };
    IntervalSolver(){};

    inline IntervalMatrices get_in_x(){
              return in_x;
    }

    /// �����ú���
    void initializeMatrix();

    /// ʵ�ֽ�һ��std::vector<Eigen::MatrixXd> -> IntervalMatrix�� �������ش����ӵ�����Ĺ̶���Ϣ��weight��bias��filterת��ΪIntervalMatrix
    IntervalMatrices convertMatricesToIntervalMatrices(const std::vector<Eigen::MatrixXd>& matrices);
    IntervalMat convertMatToIntervalMat(const Eigen::MatrixXd& matrix);
    IntervalMat convertVectorXdToIntervalVector(const Eigen::VectorXd& vec);

    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> splitIntervalMatrices(const std::vector<Eigen::Matrix<IntervalValue, Eigen::Dynamic, Eigen::Dynamic>>& intervalMatrices);


    /// ÿ��solverǰ��������ֵ
    inline void setIRMatrix(IntervalMatrices x)
    {
        in_x = x;
        std::cout<<"COPYING X to IN_X: x("<<x.size()<<", "<<x[0].rows()<<", "<<x[0].cols()<<")"<<std::endl;
        std::cout<<"COPYING X to IN_X: in_x("<<in_x.size()<<", "<<in_x[0].rows()<<", "<<in_x[0].cols()<<")"<<std::endl;
    }

    IntervalMatrices ReLuNeuronNodeevaluate() const;

    IntervalMatrices BasicOPNeuronNodeevaluate(
        const BasicOPNeuronNode* basic);

    IntervalMatrices MaxPoolNeuronNodeevaluate(
        const MaxPoolNeuronNode* maxpool);

    IntervalMatrices FullyConNeuronNodeevaluate(
        const FullyConNeuronNode* fully);

    IntervalMatrices ConvNeuronNodeevaluate(
        const ConvNeuronNode* conv);

    IntervalMatrices ConstantNeuronNodeevaluate();

    /// ���������������ֵ
    //  todo
};

}/// SVF namespace

#endif // SVF_INTERVALSOLVER_H
