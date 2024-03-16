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
typedef Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic> IntervalEigen;
typedef std::vector<IntervalEigen> IntervalMatrix;
typedef Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, 1> IntervalVector;


class IntervalSolver
{
public:

    /// ��������ؾ���
    std::vector<Eigen::MatrixXd> data_matrix;
    /// ת��ΪInterval�����ؾ���
    IntervalMatrix interval_data_matrix;
    /// ÿ��solver���������
    IntervalMatrix in_x;

    /// ���캯���� ����������ؾ���, ת��Ϊ IntervalMatrix
    IntervalSolver(const std::vector<Eigen::MatrixXd>& in): data_matrix(in){
              interval_data_matrix = convertMatricesToIntervalMatrices(data_matrix);
          };

    inline IntervalMatrix get_in_x(){
              return in_x;
    }

    /// �����ú���
    void initializeMatrix();

    /// ʵ�ֽ�һ��std::vector<Eigen::MatrixXd> -> IntervalMatrix�� �������ش����ӵ�����Ĺ̶���Ϣ��weight��bias��filterת��ΪIntervalMatrix
    IntervalMatrix convertMatricesToIntervalMatrices(const std::vector<Eigen::MatrixXd>& matrices);
    IntervalEigen convertEigenToIntervalEigen(const Eigen::MatrixXd& matrix);
    IntervalEigen convertVectorXdToIntervalVector(const Eigen::VectorXd& vec);

    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> splitIntervalMatrices(const std::vector<Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic>>& intervalMatrices);


    /// ÿ��solverǰ��������ֵ
    inline void setIRMatrix(IntervalMatrix x)
    {
        in_x = x;
    }

    IntervalMatrix ReLuNeuronNodeevaluate() const;

    IntervalMatrix BasicOPNeuronNodeevaluate(
        const SVF::BasicOPNeuronNode* basic);

    IntervalMatrix MaxPoolNeuronNodeevaluate(
        const SVF::MaxPoolNeuronNode* maxpool);

    IntervalMatrix FullyConNeuronNodeevaluate(
        const SVF::FullyConNeuronNode* fully);

    IntervalMatrix ConvNeuronNodeevaluate(
        const SVF::ConvNeuronNode* conv) const;

    IntervalMatrix ConstantNeuronNodeevaluate() const;

    /// ���������������ֵ
    //  todo
};

}/// SVF namespace

#endif // SVF_INTERVALSOLVER_H
