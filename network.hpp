#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <iterator>
#include <algorithm>
#include<random>
#include <opencv2/ml.hpp>
#include<utility>

class Network {
public:

    double eta;
    Network(std::vector<int> sizes, double eta);
    ~Network();

    void initialize();
    std::vector<double> feedforward(std::vector<double> a);
    int argmax(std::vector<double>& x);
    double evaluate(std::vector<std::pair<std::vector<double>, int>>& test_data);
    void Network::SGD(std::vector<std::pair<std::vector<double>, std::vector<double>>> training_data = {}, int epochs = 0, int mini_batch_size = 0, double eta = 0.0, std::vector<std::pair<std::vector<double>, int>> test_data = {});
    void Network::mini_batch_update( std::vector<std::pair<std::vector<double>, std::vector<double>>>& mini_batch, double eta);
    void Network::backprop(std::vector<double>& x, std::vector<double>& y, std::vector<std::vector<std::vector<double>>>& nabla_w, std::vector<std::vector<double>>& nabla_b);

private:


    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<int> sizes;
    std::vector<double> Network::hadprod(std::vector<double>& x, std::vector<double>& y);
    std::vector<double> cost_derivative(std::vector<double>& x, std::vector<double>& y);
    std::vector<std::vector<double>>* Network::setgradient(std::vector<std::vector<double>> w, std::vector<double> activations, std::vector<double> delta);
    std::vector<double> sigmoid_prime(std::vector<double>& z);
    std::vector<double> vectoradd(std::vector<double>& a, std::vector<double>& b);
    std::vector<std::vector<double>> vectoradd(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b);
    std::vector<double> matrixmult(std::vector<std::vector<double>>& a, std::vector<double>& b);
    std::vector<std::vector<double>> Network::matrixmult(std::vector<std::vector<double>>& a, std::vector< std::vector<double>>& b);
    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>>& a); // Ensure this declaration is correct
    /*std::vector<std::vector<double>>* scale(double a, std::vector<std::vector<double>>& b);
    std::vector<double>* scale(double a,   std::vector<double>& b);
    double sigmoid(double z);*/
    std::vector<double> sigmoid(std::vector<double>& z);
    std::vector<std::vector<double>> Network::sigmoid(std::vector<std::vector<double>>& b);
    double Network::sigmoid(double b);
};
//free funcs
double dot(std::vector<double>& x, std::vector<double>& y) ;
std::vector<std::vector<unsigned char>> readLabelFile(const std::string& filename);
std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename);




#endif // NETWORK_H