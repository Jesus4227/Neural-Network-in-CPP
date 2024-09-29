// NeuralNetworkVS.cpp : Defines the entry point for the application.
//

//Whats left to be done:
//change functions accordingly to types and what not
//Accuracy testing-increase epochs and check overfitting

#include "network.hpp"

    // the network 
Network::Network(std::vector<int> sizes,double eta) {
    this->sizes = sizes;
    this->eta = eta;
    initialize();
}
Network::~Network() {}

//Initialize the weights and biases matrix


void Network::initialize()
{            
    std::random_device rd;
    std::mt19937 gen(rd());

    for ( int i = 1; i < static_cast<int>(sizes.size()); ++i) {
        std::vector<std::vector<double>> layerWeights(sizes[i], std::vector<double>(sizes[i-1]));
        std::vector<double> layerBiases(sizes[i]);

        std::normal_distribution<double> dist(0, 1.0 / std::sqrt(sizes[i-1]));  // Xavier initialization

        for (auto& neuronWeights : layerWeights) {
            for (auto& weight : neuronWeights) {
                weight = dist(gen);
            }
        }

        for (auto& bias :layerBiases ) {
            bias = dist(gen);
        }

        weights.push_back(layerWeights);
        biases.push_back(layerBiases);
    }
}



//Return the output of the network if "a" is input.


std::vector<double> Network::feedforward(std::vector<double> a) {
    for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
        if (i >= biases.size() || i >= weights.size()) break;
        a = sigmoid(vectoradd(matrixmult(weights[i],a), biases[i]));
    }
    return a;
}


//argmax

int Network::argmax(std::vector<double>& x){
    if (x.empty())return -1;
    double max=x[0];
    int k=0;
    for(int j=1;j<static_cast<int>(x.size());j++){
        if(x[j]>max){
            max = x[j];
            k=j;
        }

    }return k;
}
//function to evaluate number of correct test cases
double Network::evaluate(std::vector<std::pair<std::vector<double>, int>>& test_data) {
    if (test_data.empty()) {
        return 0.0;  // Return a default value if test_data is empty
    }
    int count = 0;
    for (auto& x : test_data) {
        if (argmax(feedforward(x.first)) == (x.second)) {
            count++;
        }
    }
    return static_cast<double>(count) / static_cast<double>(test_data.size()) * 100.0;
}
    //to perform stochastic gradient descent
 void Network::SGD(std::vector<std::pair<std::vector<double>, std::vector<double>>> training_data, int epochs, int mini_batch_size, double eta, std::vector < std::pair < std::vector<double>, int>> test_data) {
    int n=static_cast<int>(training_data.size());
    double decay_rate = 0.995;
    std::cout << static_cast<int>(test_data.size()) << "";
    int n_test = static_cast<int>((test_data).size());
    for(int i = 0; i < epochs; i++) {
    // Using std::random_device and std::mt19937 for randomness
        eta = eta * decay_rate;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(training_data.begin(), training_data.end(), g);
    int j = 0;
    while (j < n) {
        int batch_end = std::min(j + mini_batch_size, n);
        std::vector<std::pair<std::vector<double>, std::vector<double>>> mini_batch(training_data.begin() + j, training_data.begin() + batch_end);
        mini_batch_update(mini_batch, eta);
        j += mini_batch_size;
    }std::cout << "Epoch" << i << "finished"<<std::endl;
    
    if (!test_data.empty()) {

        std::cout << "For epoch " << i << ", the accuracy is " << evaluate(test_data) << '/' << n_test << std::endl;
    } else {
        std::cout << "Epoch " << i << " completed" << std::endl;
    }
}
}

        
    
    


    //to perform mini batch updation

void Network::mini_batch_update(std::vector<std::pair<std::vector<double>, std::vector<double>>>& mini_batch, double eta) {
    std::vector<std::vector<std::vector<double>>> nabla_w;
    std::vector<std::vector<double>> nabla_b;
    int mini_batch_size = (int)mini_batch.size();
    for (size_t i = 0; i < biases.size(); i++) {
        nabla_b.push_back(std::vector<double>((biases[i]).size()));
    }
    for (size_t j = 0; j < weights.size(); j++) {
        nabla_w.push_back(std::vector<std::vector<double>>((weights[j]).size(), std::vector<double>(((weights[j])[0]).size())));
    }
    
    for (auto x = mini_batch.begin(); x != mini_batch.end(); ++x) {

        std::vector<std::vector<std::vector<double>>> d_nabla_w(weights.size());
        std::vector<std::vector<double>> d_nabla_b(biases.size());

        for (size_t i = 0; i < biases.size(); ++i) {
            d_nabla_b[i] = std::vector<double>(biases[i].size(), 0.0);
        }
        for (size_t j = 0; j < weights.size(); ++j) {
            d_nabla_w[j] = std::vector<std::vector<double>>(weights[j].size(), std::vector<double>(weights[j][0].size(), 0.0));
        }

        backprop((*x).first, (*x).second, d_nabla_w, d_nabla_b);
        for (size_t i = 0; i < nabla_w.size(); i++) {
            for (size_t j = 0; j < nabla_w[i].size(); j++) {
                for (size_t k = 0; k < nabla_w[i][j].size(); k++) {
                    nabla_w[i][j][k] += d_nabla_w[i][j][k];
                }
            }
        }

        // Accumulate gradients for nabla_b (element-wise addition)
        for (size_t i = 0; i < nabla_b.size(); i++) {
            for (size_t j = 0; j < nabla_b[i].size(); j++) {
                nabla_b[i][j] += d_nabla_b[i][j];
            }
        }
    }for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t k = 0; k < weights[i].size(); ++k) {
            for (size_t l = 0; l < weights[i][k].size(); ++l) {
                weights[i][k][l] -= eta/mini_batch_size * nabla_w[i][k][l];
            }
        }
    }
    for (size_t i = 0; i < biases.size(); i++) {
        for (size_t k = 0; k < biases[i].size(); k++) {
            biases[i][k] -=eta /mini_batch_size * nabla_b[i][k];
        }
    }
        
}

// backpropagation algorithm
void Network::backprop(std::vector<double>& x, std::vector<double>& y, std::vector<std::vector<std::vector<double>>>& nabla_w, std::vector<std::vector<double>>& nabla_b) {
    std::vector<std::vector<double>> activations;

    std::vector<std::vector<double>> zs;
    activations.push_back(x);
    // creating list of activations
    auto i = weights.begin();
    auto j = biases.begin();
    std::vector<double> add;
    while (i != weights.end() && j != biases.end()) {
        add = vectoradd(matrixmult(*i, x), *j);
        zs.push_back(add);
        x = sigmoid(add);
        activations.push_back(x);
        i++;
        j++;
    }

    int n = static_cast<int>(activations.size());
    std::vector<double> delta = hadprod(cost_derivative(activations[n - 1], y), sigmoid_prime(zs[n - 2]));
    nabla_b[n - 2] = delta;

    for (int i = 0; i < static_cast<int>(delta.size()); ++i) {
        for (int j = 0; j < static_cast<int>(activations[n - 1].size()); ++j) {
            nabla_w[n - 2][i][j] = delta[i] * activations[n - 2][j];
        }
    }    

    

    for (int l = 2; l < n; l++) {

        std::vector<double> sp = sigmoid_prime(zs[n-l-1]);
        delta = hadprod(matrixmult(transpose(weights[n-l]), delta), sp);
        nabla_b[n-l-1] = delta;
        for (int i = 0; i < static_cast<int>(delta.size()); ++i) {
            for (int j = 0; j < static_cast<int>(activations[n-l-1].size()); ++j) {  
                nabla_w[n - l - 1][i][j] = delta[i] * activations[n - l - 1][j];
            }
        }
    }


}










//MATRIX FUNCTIONS




//hadamard product
std::vector<double> Network::hadprod(std::vector<double>& x, std::vector<double>& y) {

    if (x.size() == y.size()) {
        std::vector<double> c(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            c[i] = x[i] * y[i];
        }return c;
    }return std::vector<double>();
}

    // cost activation gradient
    std::vector<double> Network::cost_derivative(std::vector<double>& x, std::vector<double>& y) {
        std::vector<double> c(x.size());
        for (int i = 0; i < x.size();i++) {
            c[i]=x[i]-y[i] ;
        }
        return c;
    }

    //sigmoid prime

    std::vector<double> Network::sigmoid_prime(std::vector<double>& z) {
        std::vector<double> c=z;
        for (auto& x : c) {
            double l = sigmoid(x);
            x = l * (1 - l);
        }return c;
    }



//setting gradients for weight matrix
std::vector<std::vector<double>>* Network::setgradient(std::vector<std::vector<double>> w, std::vector<double> activations,std::vector<double> delta) {
    for (size_t i = 0; i <delta.size(); i++) {
        for (size_t j = 0; j <(activations).size(); j++) {
            w[i][j] = delta[i] * activations[j];
        }return &w;
    }return nullptr;
}


//transpose of matrix

std::vector<std::vector<double>> Network::transpose(std::vector<std::vector<double>>& x) {
    int rows = static_cast<int>(x.size());
    int cols = static_cast<int>(x[0].size());

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = x[i][j];
        }
    }

    return result;
}




// matrix multiplication
std::vector<double> Network::matrixmult(std::vector<std::vector<double>>& a,std::vector<double>& b) {
    if(a[0].size() != b.size()) {
        throw std::invalid_argument("Incompatible matrices for multiplication");
    }std::vector<double> c(a.size());
for (size_t i = 0; i < a.size(); i++) {
    for (size_t j = 0; j < b.size(); j++) {
        c[i] += a[i][j] * b[j];
    }
}
return c;
}
std::vector<std::vector<double>> Network::matrixmult(std::vector<std::vector<double>>& a, std::vector< std::vector<double>>& b) {
    int x = static_cast<int>(b[0].size());
    int y = static_cast<int>(a[0].size());
    int rows = static_cast<int>(a.size());
    std::vector<std::vector<double>> c(a.size(), std::vector<double>(b[0].size(), 0.0));
    double sum = 0.0;
    if (y != b.size()) {
        throw std::invalid_argument("Incompatible matrices for multiplication");// This returns an empty vector
    }
    else {

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < y; k++) { // Iterate over columns of A / rows of B
                for (int j = 0; j < x; j++) {
                    c[i][j] += a[i][k] * b[k][j]; // Correct indexing for multiplication
                }
            }
        }
        return c;
    }return std::vector<std::vector<double>>();
}





/*functions to scale matrices and vectors
std::vector<double>* Network::scale(double j, std::vector<double>& a) {
    for (auto x : a) {
        x = x * j;
    }return &a;

}
std::vector<std::vector<double>>* Network::scale(double j, std::vector<std::vector<double>>& a) {
    for (auto& x : a) {
        for (auto& y : x) {
            y = y * j;
        }
    }return &a;
}*/

//function to return the sum of two vectors
std::vector<double> Network::vectoradd(std::vector<double>& a, std::vector<double>& b) {
    if(a.size()!=b.size())
    {
        throw std::invalid_argument("Incompatible vectors for addition");
    }
        std::vector<double> c(a.size());

    for (size_t i = 0; i <a.size(); i++) {
        c[i] = a[i] + b[i];
    }return c;
}

//function to add two matrices

std::vector<std::vector<double>> Network::vectoradd(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b) {
    if((a.size()!=b.size()) || (b[0].size()!=a[0].size())){throw std::invalid_argument("Incompatible matrices for multiplication");
}
    std::vector<std::vector<double>> c(a.size(), std::vector<double>(a[0].size()));
    for (size_t i = 0; i <a.size(); i++) {
        for (size_t j = 0; j < a[i].size(); j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }return c;
}

// Activation function for vectors of vectors
std::vector<std::vector<double>> Network::sigmoid(std::vector<std::vector<double>>& b) {
    auto x=b.begin();
    while(x!=b.end()) {  // iterate by reference to modify the original vector
        *x = sigmoid(*x);
        x++; 
    }
    return b;
}

// Activation function for vectors
std::vector<double> Network::sigmoid(std::vector<double>& b) {
    for (auto& y : b) {  // iterate by reference to modify the original vector
        y = 1 / (1 + exp(-y));
    }
    return b;
}

double Network::sigmoid(double b) {
    return 1 / (1 + std::exp(-b));
}

//Function to one encode

std::vector<double> one_encode(double x){
    std::vector<double> y(10);
    y[static_cast<int>(x)]=1.0;
    return y;

}

//dot product


double dot(std::vector<double>& x, std::vector<double>& y) {
    double sum = 0.0;
    int i = 0;
    while (i < x.size()) {
        sum = sum + x.at(i) * y.at(i);
        i = i + 1;
    }
    return sum;
}

//flatten 2d vectors
std::vector<double> flatten(std::vector<std::vector<double>> a) {
    int rows = static_cast<int>(a.size());
    int cols = static_cast<int>(a[0].size());
    std::vector<double> f(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        f[i] = a[i / cols][i % cols];
    }return f;
}








// Functions to read IDX3-UBYTE files



std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | (static_cast<unsigned char>(numRowsBytes[1]) << 16) | (static_cast<unsigned char>(numRowsBytes[2]) << 8) | static_cast<unsigned char>(numRowsBytes[3]);
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | (static_cast<unsigned char>(numColsBytes[1]) << 16) | (static_cast<unsigned char>(numColsBytes[2]) << 8) | static_cast<unsigned char>(numColsBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(numRows * numCols);
        file.read((char*)(image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}

// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>> readLabelFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);
    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(1);
        file.read((char*)(image.data()), 1);

        images.push_back(image);
    }

    file.close();

    return images;
}


//main function


int main() {

    //set learning rate

    double eta = 0.025;
    Network* x = new Network({ 784,30,30,10 },eta);

    std::string filename = "C://Users//Sriyansh//Downloads//MNIST//train-images-idx3-ubyte//train-images.idx3-ubyte";
    std::string testfilename = "C://Users//Sriyansh//Downloads//MNIST//t10k-images-idx3-ubyte//t10k-images.idx3-ubyte";
    std::string testlabels="C://Users//Sriyansh//Downloads//MNIST//t10k-labels-idx1-ubyte//t10k-labels.idx1-ubyte";
    std::string label_filename = "C://Users//Sriyansh//Downloads//MNIST//train-labels-idx1-ubyte//train-labels.idx1-ubyte";

    std::vector<std::vector<unsigned char>> imagesFile = readIDX3UByteFile(filename);
    std::vector<std::vector<unsigned char>> labelsFile = readLabelFile(label_filename); 
    std::vector<std::vector<unsigned char>> testimages = readIDX3UByteFile(testfilename);
    std::vector<std::vector<unsigned char>> testlabelsFile = readLabelFile(testlabels);
    
    // Corresponding labels
    std::vector<std::pair<std::vector<double>,std::vector<double>>> training_data;
    
    std::vector<std::pair<std::vector<double>,int>> y;
    for (int imgCnt = 0; imgCnt < (int)imagesFile.size(); imgCnt++)
    {
        int rowCounter = 0;
        int colCounter = 0;

        std::vector<std::vector<double>> tempImg(28, std::vector<double>(28));
        for (int i = 0; i < (int)imagesFile[imgCnt].size(); i++) {

            tempImg[rowCounter][colCounter++] = static_cast<double>((imagesFile[imgCnt][i]) / 255.0);
            if ((i) % 28 == 0) {
                rowCounter++;
                colCounter = 0;
                if (i == 756)
                    break;
            }
        }std::pair<std::vector<double>, std::vector<double>> p;
        p.first = flatten(tempImg);
        p.second = one_encode(static_cast<double>(labelsFile[imgCnt][0]));
        training_data.push_back(p);
        
        // train the model
       
    }
    for (int imgCnt = 0; imgCnt < (int)testimages.size(); imgCnt++)
    {
        int rowCounter = 0;
        int colCounter = 0;

        std::vector<std::vector<double>> tempImg(28, std::vector<double>(28));
        for (int i = 0; i < (int)testimages[imgCnt].size(); i++) {

            tempImg[rowCounter][colCounter++] = static_cast<double>((testimages[imgCnt][i]) / 255.0);
            if ((i) % 28 == 0) {
                rowCounter++;
                colCounter = 0;
                if (i == 756)
                    break;
            }
        }std::pair<std::vector<double>, int> p;
        p.first = flatten(tempImg);
        p.second = static_cast<int>(testlabelsFile[imgCnt][0]);
        y.push_back(p);

        // create testing for the model

    }std::cout<<y.size() << "";
    
    x->SGD(training_data, 20, 32, eta, y);
    std::cout << "training done" << "";
    
    return 0;
}
