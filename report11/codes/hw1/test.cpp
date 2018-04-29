#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
// #define M_PI 3.1416

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::GeneralizedEigenSolver;

void save_mat(std::string name, const MatrixXd* mat){
  std::ofstream file(name);
  file << *mat << std::endl << std::endl;
}

float gk(const MatrixXd& x, const MatrixXd& c){
  float hh = std::pow(2*1, 2); 
  auto d = (x - c).norm();
  return std::exp(-d*d/hh);
}


int main()
{
  std::mt19937 engine;
  std::normal_distribution<> dist(0, 1);
  std::uniform_real_distribution<> unif(0, 1);

  /* make points */
  MatrixXd matX(2, 100);
  MatrixXd sumX(2, 1);
  MatrixXd vecMu(2, 2);
  // for(int i=0; i < 100; i++){
  //   matX(0, i) = dist(engine);
  //   if(i < 50){
  //     matX(0, i) -= 4;
  //   }else{
  //     matX(0, i) += 4;
  //   }
  //   matX(1, i) = dist(engine);
  // }
  for(int i=0; i < 100; i++){
    matX(0, i) = dist(engine);
    if(i < 25){
      matX(0, i) -= 4;
    }else if(25 <= i && i < 50){
      matX(0, i) += 4;
    }
    matX(1, i) = dist(engine);
  }

  /* centralize */
  for(int i=0; i < 100; i++){
    sumX += matX.col(i);
  }
  for(int i=0; i < 100; i++){
    matX.col(i) -= sumX/100;
  }

  vecMu(0, 0) = 0;
  vecMu(0, 1) = 0;
  vecMu(1, 0) = 0;
  vecMu(1, 1) = 0;
  std::cout << vecMu << std::endl;
  for(int i=0; i < 100; i++){
    if(i < 50){
      vecMu(0, 0) += matX(0, i);
      vecMu(1, 0) += matX(1, i);
    }else{
      vecMu(0, 1) += matX(0, i);
      vecMu(1, 1) += matX(1, i);
    }
    std::cout << i << "," << vecMu << std::endl;
  }
  // vecMu /= 50;

  MatrixXd matSw(2, 2);
  // std::cout << matSw << std::endl;
  for(int i=0; i < 100; i++){
    if(i < 50){
      matSw += (matX.col(i) - vecMu.col(0))*(matX.col(i) - vecMu.col(0)).transpose();
    }else{
      matSw += (matX.col(i) - vecMu.col(1))*(matX.col(i) - vecMu.col(1)).transpose();
    }
      // std::cout << i << "," << matSw << std::endl;
  }

  MatrixXd matSb(2, 2);
  matSb = 50*vecMu.col(0)*vecMu.col(0).transpose() + 50*vecMu.col(1)*vecMu.col(1).transpose();





  // MatrixXd matW(100, 100);
  // for(int i=0; i < 100; i++){
  //   for(int j=0; j < 100; j++){
  //     matW(i, j) = gk(matX.col(i), matX.col(j));
  //   }
  // }

  // MatrixXd matD(100, 100);
  // for(int i=0; i < 100; i++){
  //   matD(i, i) = matW.col(i).sum();
  // }

  // MatrixXd matL = matD - matW;

  GeneralizedEigenSolver<MatrixXd> ges;
  ges.compute(matSb, matSw, true);
  MatrixXd matTlpp(2, 2); //= ges.eigenvectors().alpha();
  for(int i=0; i < 2; i++){
    for(int j=0; j < 2; j++){
      matTlpp(i, j) = ges.eigenvectors()(i, j).real();
    }
  }
  MatrixXd bvec(2, 1);
  if(ges.eigenvalues()(0).real() > ges.eigenvalues()(1).real()){
    bvec = matTlpp.col(0);
  }else{
    bvec = matTlpp.col(1);
  }

  // MatrixXd matRed = matTlpp*matX;

  // // out
  // std::ofstream file("matTlpp");
  // file << matTlpp << std::endl;
  // std::ofstream filew("matW");
  // filew << matW << std::endl;
  // std::ofstream filed("matD");
  // filed << matD << std::endl;
  // std::ofstream filep("points");
  // filep << matX << std::endl;
  // std::ofstream fileeig("eig");
  // fileeig << ges.eigenvalues() << std::endl << ges.eigenvectors() << std::endl;
  // std::ofstream filered("matRed");
  // filered << matRed << std::endl;
  // std::ofstream filewvec("matX");
  // filewvec << matX << std::endl;
  // std::ofstream filebvec("bestVec");
  // filebvec << bvec << std::endl;
  save_mat(std::string("matSb"), &matSb);
  save_mat(std::string("matSw"), &matSw);
  save_mat(std::string("vecMu"), &vecMu);
  save_mat(std::string("matTlpp"), &matTlpp);
  save_mat(std::string("matX"), &matX);
  save_mat(std::string("bVec"), &bvec);
}