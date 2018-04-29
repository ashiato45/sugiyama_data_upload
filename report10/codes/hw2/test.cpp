#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#define M_PI 3.1416

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::GeneralizedEigenSolver;

void save_mat(std::string& name, const MatrixXd* mat){
  std::ofstream file(name);
  file << *mat << std::endl;
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

  // MatrixXd matX(2, 100);
  // for(int i=0; i < 100; i++){
  //   matX(0, i) = 2*dist(engine);
  //   matX(1, i) = dist(engine);
  // }
  MatrixXd matX(2, 100);
  for(int i=0; i < 100; i++){
    matX(0, i) = dist(engine);
    matX(1, i) = 2*std::round(unif(engine)) - 1 + dist(engine)/3;
  }

  MatrixXd matW(100, 100);
  for(int i=0; i < 100; i++){
    for(int j=0; j < 100; j++){
      matW(i, j) = gk(matX.col(i), matX.col(j));
    }
  }

  MatrixXd matD(100, 100);
  for(int i=0; i < 100; i++){
    matD(i, i) = matW.col(i).sum();
  }

  MatrixXd matL = matD - matW;

  GeneralizedEigenSolver<MatrixXd> ges;
  ges.compute(matX*matL*matX.transpose(), matX*matD*matX.transpose(), true);
  MatrixXd matTlpp(2, 2); //= ges.eigenvectors().alpha();
  for(int i=0; i < 2; i++){
    for(int j=0; j < 2; j++){
      matTlpp(i, j) = ges.eigenvectors()(i, j).real();
    }
  }
  MatrixXd wvec(2, 1);
  if(ges.eigenvalues()(0).real() < ges.eigenvalues()(1).real()){
    wvec = matTlpp.col(0);
  }else{
    wvec = matTlpp.col(1);
  }

  MatrixXd matRed = matTlpp*matX;

  // out
  std::ofstream file("matTlpp");
  file << matTlpp << std::endl;
  std::ofstream filew("matW");
  filew << matW << std::endl;
  std::ofstream filed("matD");
  filed << matD << std::endl;
  std::ofstream filep("points");
  filep << matX << std::endl;
  std::ofstream fileeig("eig");
  fileeig << ges.eigenvalues() << std::endl << ges.eigenvectors() << std::endl;
  std::ofstream filered("matRed");
  filered << matRed << std::endl;
  std::ofstream filewvec("worstVec");
  filewvec << wvec << std::endl;
}