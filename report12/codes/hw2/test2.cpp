#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <RedSVD/RedSVD-h>

// #define M_PI 3.1416
#define NUM_X 1000

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::GeneralizedEigenSolver;
using Eigen::SparseMatrix;

void save_mat(std::string name, const MatrixXd* mat){
  std::ofstream file(name);
  file << *mat << std::endl << std::endl;
}

void save_mat_sparse(std::string name, const SparseMatrix<double>* mat){
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
  MatrixXd matX(3, NUM_X);
  for(int i=0; i < NUM_X; i++){
    double a = 3*M_PI*dist(engine);
    matX(0, i) = a*std::cos(a);
    matX(1, i) = 30*dist(engine);
    matX(2, i) = a*std::sin(a);
  }

  /* make similarity matrix */
  SparseMatrix<double> matW(NUM_X, NUM_X);
  // MatrixXd matW(NUM_X, NUM_X);
  std::vector<std::pair<double, int>> pairs(NUM_X);
  int parK = 2;
  for(int i=0; i < NUM_X; i++){
    /* calc dist from x[i] */
    for(int j=0; j < NUM_X; j++){
      pairs[j] = std::make_pair((matX.col(i) - matX.col(j)).norm(), j);
    }
    std::sort(pairs.begin(), pairs.end());
    // matW(i, pairs[0].second) = 1;
    // matW(pairs[0].second, i) = 1;
    // matW(i, pairs[1].second) = 1;
    // matW(pairs[1].second, i) = 1;
    matW.insert(i, pairs[0].second) = 1;
    matW.insert(pairs[0].second, i) = 1;
    matW.insert(i, pairs[1].second) = 1;
    matW.insert(pairs[1].second, i) = 1;
  }
  SparseMatrix<double> matD(NUM_X, NUM_X);
  // MatrixXd matD(NUM_X, NUM_X);
  for(int i=0; i < NUM_X; i++){
    matD.insert(i, i) = matW.col(i).sum();
    // matD(i, i) = matW.col(i).sum();
  }

  std::cout << "start laplus" << std::endl;

  /* calc laplus */
  RedSVD::RedSVD<SparseMatrix<double>> svd()
  // GeneralizedEigenSolver<SparseMatrix<double>> ges;
  GeneralizedEigenSolver<MatrixXd> ges;
  ges.compute(matD - matW, matD, true);
  MatrixXd matEigen(NUM_X, NUM_X);
  for(int i=0; i < NUM_X; i++){
    for(int j=0; j < NUM_X; j++){
      matEigen(i, j) = ges.eigenvectors()(i, j).real();
    }
  }
  for(int i=0; i < NUM_X; i++){
    pairs[i] = std::make_pair(ges.eigenvalues()(i).real(), i);
  }
  std::cout << "start sort" << std::endl;
  std::sort(pairs.begin(), pairs.end());
  int cnt = 0;
  MatrixXd matPsi(2, NUM_X);
  for(int i=0; i < NUM_X; i++){
    if(pairs[i].first > 10e-8){
      for(int j=0; j < NUM_X; j++){
        matPsi(cnt, j) = matEigen(j, i);
      }
      cnt++;
    }
    if(cnt <= 2){
      break;
    }
  }

  std::cout << "start saving" << std::endl;

  // save_mat(std::string("matSb"), &matSb);
  // save_mat(std::string("matSw"), &matSw);
  // save_mat(std::string("vecMu"), &vecMu);
  // save_mat(std::string("matTlpp"), &matTlpp);
  save_mat(std::string("matX"), &matX);
  save_mat(std::string("matW"), &matW);
  save_mat(std::string("matD"), &matD);
  save_mat(std::string("matPsi"), &matPsi);
}