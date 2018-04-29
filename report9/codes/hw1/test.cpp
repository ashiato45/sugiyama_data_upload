#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#define M_PI 3.1416

#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using Eigen::MatrixXd;

void save_mat(std::string& name, const MatrixXd* mat){
  std::ofstream file(name);
  file << *mat << std::endl;
}

float k(const MatrixXd& x, const MatrixXd& c){
  float hh = std::pow(2*1, 2); 
  auto d = (x - c).norm();
  return std::exp(-d*d/hh);
}

int main()
{
  std::mt19937 engine;
  std::normal_distribution<> dist(0, 1);

  MatrixXd a(100, 1);
  for(int i=0; i < 100; i++){
    a(i) = M_PI*i/100;
  }
  MatrixXd x(2, 200);
  for(int i=0; i < 100; i++){
      x(0, i) = -10*(std::cos(a(i)) + 0.5) + dist(engine);
      x(1, i) = 10*std::sin(a(i)) + dist(engine);
      x(0, i + 100) = -10*(std::cos(a(i)) - 0.5) + dist(engine);
      x(1, i + 100) = -10*std::sin(a(i)) + dist(engine);
  }
  MatrixXd y(1, 200);
  y(0) = -1;
  y(199) = 1;
  MatrixXd yy(1, 2);
  yy(0) = -1;
  yy(1) = 1;


  /* Learn */
  MatrixXd phi(200, 200);
  for(int i=0; i < 200; i++){
    for(int j=0; j < 200; j++){
      phi(i, j) =  k(x.col(i), x.col(j));
    }
  }
  MatrixXd phit(2, 200);
  for(int j=0; j < 200; j++){
    phit(0, j) = k(x.col(0), x.col(j));
    phit(1, j) = k(x.col(199), x.col(j));
  }
  MatrixXd w = phi;
  MatrixXd d(200, 200);
  for(int i=0; i < 200; i++){
    d(i, i) = w.col(i).sum();
  }
  MatrixXd l = d - w;

  MatrixXd A = phit.transpose()*phit + MatrixXd::Identity(200, 200) + 2*phi.transpose()*l*phi;
  MatrixXd b = phit.transpose()*yy.transpose();
  MatrixXd theta = A.colPivHouseholderQr().solve(b);

  /* Output */
  MatrixXd out(100, 100);
  for(int i=0; i < 100; i++){
    for(int j=0; j < 100; j++){
      float mx = -20*(1.0 - i/100.0) + 20*(i/100.0);
      float my = -20*(1.0 - j/100.0) + 20*(j/100.0);
      for(int kk = 0; kk < 200; kk++){
        MatrixXd xx(2, 1);
        xx << mx, my;
        out(j, i) += theta(kk)*k(x.col(kk), xx);
      }
    }
  }

  std::ofstream file("x");
  file << x << std::endl;
  std::ofstream fileout("out");
  fileout << out << std::endl;
  //std::cout << m << std::endl;
}