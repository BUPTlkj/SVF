#ifndef SVF_INTERVALSOLVER_H
#define SVF_INTERVALSOLVER_H
#include "../svf/include/AE/Core/IntervalValue.h"
#include "Eigen/Dense"
#include "../svf/include/Graphs//NNNode.h"

/// 1.需要将分析后获得的weight、bias、filter全部转化为Interval类型
/// 2.需要将ConvNode，MaxPoolNode，BasicNode，ConstantNode，FilterNode，ReluNode的属性全部换为Interval，并将里面的get方法重构
/// 3.对于基础点NeuronNode，需要重构基本属性
/// 4.对于图的计算需要重构中间运算结果的处理

///*********************
/// * 1.传入matrix->IntervalMatrix
/// * 2.正常遍历：中间值的std::vector<Eigen::Matrix> IRRes -> IntervalMatrix IRRes
/// 3.每个节点处理的过程中将节点的std::vector<Eigen::Matrix> -> IntervalMatrix
/// 4.换solver计算过程



namespace SVF
{
typedef Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic> IntervalEigen;
typedef std::vector<IntervalEigen> IntervalMatrix;
typedef Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, 1> IntervalVector;


class IntervalSolver
{
public:

    /// 输入的像素矩阵
    std::vector<Eigen::MatrixXd> data_matrix;
    /// 转化为Interval的像素矩阵
    IntervalMatrix interval_data_matrix;
    /// 每次solver计算的输入
    IntervalMatrix in_x;

    /// 构造函数， 仅需接受像素矩阵, 转化为 IntervalMatrix
    IntervalSolver(const std::vector<Eigen::MatrixXd>& in): data_matrix(in){
              interval_data_matrix = convertMatricesToIntervalMatrices(data_matrix);
          };

    inline IntervalMatrix get_in_x(){
              return in_x;
    }

    /// 测试用函数
    void initializeMatrix();

    /// 实现将一个std::vector<Eigen::MatrixXd> -> IntervalMatrix， 仅对像素处理，从点里面的固定信息如weight，bias，filter转化为IntervalMatrix
    IntervalMatrix convertMatricesToIntervalMatrices(const std::vector<Eigen::MatrixXd>& matrices);
    IntervalEigen convertEigenToIntervalEigen(const Eigen::MatrixXd& matrix);
    IntervalEigen convertVectorXdToIntervalVector(const Eigen::VectorXd& vec);

    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> splitIntervalMatrices(const std::vector<Eigen::Matrix<SVF::IntervalValue, Eigen::Dynamic, Eigen::Dynamic>>& intervalMatrices);


    /// 每次solver前设置输入值
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

    /// 解析分析最后的输出值
    //  todo
};

}/// SVF namespace

#endif // SVF_INTERVALSOLVER_H
