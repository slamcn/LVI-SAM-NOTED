#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;

/**
* @class IntegrationBase IMU pre-integration class
* @Description 
*/
class IntegrationBase
{
  public:

    // 相邻的两个imu数据
    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian;   // 雅克比
    Eigen::Matrix<double, 15, 15> covariance; // 协方差
    Eigen::Matrix<double, 18, 18> noise;      // 测量噪声

    // 预积分量 pvq
    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    // 保存imu数据
    // saves all the IMU measurements and time difference between two image frames
    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

    IntegrationBase() = delete;

    IntegrationBase(const Eigen::Vector3d &_acc_0, 
                    const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, 
                    const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, 
          gyr_0{_gyr_0}, 
          linearized_acc{_acc_0}, 
          linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, 
          linearized_bg{_linearized_bg},
          jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, 
          covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, 
          delta_p{Eigen::Vector3d::Zero()}, 
          delta_q{Eigen::Quaterniond::Identity()}, 
          delta_v{Eigen::Vector3d::Zero()}

    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    // bias发生了改变, 重新预积分
    // after optimization, repropagate pre-integration using the updated bias
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba; // 新的bias
        linearized_bg = _linearized_bg; // 新的bias
        jacobian.setIdentity();
        covariance.setZero();
        // 遍历所有imu数据, 重新进行预积分
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    // 预积分(单个imu数据)
    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        // imu数据
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;

        // pvq
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;

        // bias
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        // 采用中值积分进行预积分
        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }

    // 采用中值积分进行预积分 (单个imu数据)
    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");

        // 采用中值积分进行预积分 (第3讲P37)
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);              // a0
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;           // w
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2); // q
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);       // a1
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);                       // a = a0 + a1;
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt; // p
        result_delta_v = delta_v + un_acc * _dt;                             // v
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        // 预积分的误差递推公式 (第3讲P46)   
        //优化： 残差对变量（状态）的导数  变量沿着梯度负方向迭代，残差最小，变量最优

        // 为什么需要此处的雅克比矩阵，即【预积分值对各个状态误差的导数】？
        // 优化的时候需要用到当前预积分值的【协方差】，而要推导预积分量的协方差,我们需要知道 imu 噪声和预积分量之间的线性【递推】关系；【传播】
        // 其中IMU单次测量误差可以由标定的参数得到，而预积分是一段时间的IMU数据累积，其误差需要通过误差递推得到；
        // 误差的传递由两部分组成:当前时刻的误差传递给下一时刻,当前时刻测量噪声传递给下一时刻；

        // 由于IMU积分过程中涉及到四元数及位姿的计算，是一个【非线性过程】，我们可以基于一阶泰勒展开将递推过程线性化；
        // 预积分值由真值和误差组成，其中-->
        // --> 预积分误差是由 各个状态量误差 以及 噪声 引起的，因此我们只要推导出 【预积分量 对 各个状态量误差 的导数】，
        // 即每个状态量产生一点儿误差，会对预积分值产生多大误差；然后我们就可以根据当前的状态量误差算出当前预积分的误差，进而计算协方差；

        // (此句结论需要反复重点理解，这是求误差递推的大方向，以及为什么要求各个状态量误差的雅克比)
        // 不是残差值对状态的雅克比 是预积分值对状态的雅克比  误差传递是非线性过程  避免每次bias更新都重新积分
        //// 非线性优化需要当前预积分值的协方差（权重）
        //// 预积分是一段时间多次测量得到的结果，因此每次预积分值的误差来源于本次测量噪声和上次预积分误差的影响
        //// 测量噪声由预先标定而来
        //// 预积分误差的递推==上次的状态量误差对本次预积分值（增量）的影响，即求增量值对[P,V,Q,Ba,Bg]的导数
        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            // 反对称矩阵
            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            // 更新jacobian和covariance
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }
    }

    
    // 计算imu预积分残差, 用于imu预积分因子 (第3讲P69)
    // calculate residuals for ceres optimization, used in imu_factor.h
    // paper equation 24
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba; // bias变化量
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        // bias变化导致的预积分变化 (第3讲P70)
        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }
};