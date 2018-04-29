#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <algorithm>
#define M_PI 3.1416

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using Eigen::MatrixXd;

float gk(const MatrixXd& x, const MatrixXd& c){
  float hh = std::pow(2*1, 2); 
  auto d = (x - c).norm();
  return std::exp(-d*d/hh);
}

int main()
{
  std::mt19937 engine;
  std::normal_distribution<> dist(0, 1);

  /* Making data */
  MatrixXd px(3, 100);
  MatrixXd py(1, 100);
  for(int i=0; i < 90; i++){
    px(0, i) = dist(engine) - 2;
    px(1, i) = 2*dist(engine);
    px(2, i) = 1;
    py(i) = -1;
  }
  for(int i=90; i < 100; i++){
    px(0, i) = dist(engine) + 2;
    px(1, i) = 2*dist(engine);
    px(2, i) = 1;
    py(i) = 1;
  }
  MatrixXd tx(3, 100);
  for(int i=0; i < 10; i++){
    tx(0, i) = dist(engine) - 2;
    tx(1, i) = 2*dist(engine);
    tx(2, i) = 1;
  }
  for(int i=10; i < 100; i++){
    tx(0, i) = dist(engine) + 2;
    tx(1, i) = 2*dist(engine);
    tx(2, i) = 1;
  }

  /* Learn */
  float hat_a_11 = 0;
  int nn_11 = 0;
  float hat_a_12 = 0;
  int nn_12 = 0;
  float hat_a_22 = 0;
  int nn_22 = 0;
  for(int i=0; i < 100; i++){
    for(int j=0; j < 100; j++){
      if(py(i) == -1 && py(j) == -1){
        hat_a_11 += (px.col(i) - px.col(j)).norm();
        nn_11++;
      }else if(py(i) == -1 && py(j) == 1){
        hat_a_12 += (px.col(i) - px.col(j)).norm();
        nn_12++;
      }else if(py(i) == 1 && py(j) == 1){
        hat_a_22 += (px.col(i) - px.col(j)).norm();
        nn_22++;
      }
    }
  }
  hat_a_11 /= (float)nn_11;
  hat_a_12 /= (float)nn_12;
  hat_a_22 /= (float)nn_22;
  float hat_b_1 = 0;
  int ndn_1 = 0;
  float hat_b_2 = 0;
  int ndn_2 = 0;
  for(int i=0; i < 100; i++){
    for(int j=0; j < 100; j++){
      if(py(j) == -1){
        hat_b_1 += (tx.col(i) - px.col(j)).norm();
        ndn_1++;
      }else if(py(j) == 1){
        hat_b_2 += (tx.col(i) - px.col(j)).norm();
        ndn_2++;
      }
    }
  }
  hat_b_1 /= (float)ndn_1;
  hat_b_2 /= (float)ndn_2;

  float til_pi = (hat_a_12 - hat_a_11 - hat_b_2 + hat_b_1)/(2*hat_a_12 - hat_a_22 - hat_a_11);
  float hat_pi = std::min(1.0f, std::max(0.0f, til_pi));

  /* Learn */
  std::cout << hat_pi << std::endl;
  MatrixXd til_w(100, 100);
  for(int i=0; i < 100; i++){
    if(py(i) == 1){
      til_w(i, i) = hat_pi;
    }else{
      til_w(i, i) = 1.0 - hat_pi;
    }
  }
  MatrixXd phi = px.transpose();
  MatrixXd theta = (phi.transpose()*til_w*phi).colPivHouseholderQr().solve(phi.transpose()*til_w*py.transpose());
  MatrixXd theta2 = (phi.transpose()*phi).colPivHouseholderQr().solve(phi.transpose()*py.transpose());


  /* Output */
  std::ofstream pxfile("px");
  pxfile << px << std::endl;
  std::ofstream txfile("tx");
  txfile << tx << std::endl;
  std::ofstream thetafile("theta");
  thetafile << theta << std::endl;
  std::ofstream theta2file("theta2");
  theta2file << theta2 << std::endl;
  //std::cout << m << std::endl;
}