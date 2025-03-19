#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <random>
#include <algorithm>
#include "Matrix.cpp"

class NN_Data
{
    private:
        std::shared_ptr<Matrix<double>> readCSV(const std::string& path);
        
        void shuffle_dataset();

    protected:
        const std::string data_path;
        std::shared_ptr<Matrix<double>> data;
    public:
        NN_Data(){}
        NN_Data(std::shared_ptr<Matrix<double>> matrix);
        NN_Data(const std::string data_path);
        NN_Data(size_t row, size_t col, bool random_matrix = false);
        ~NN_Data();

        size_t get_row_count()
        {
            std::cout << this->data->get_rows() << std::endl;
            return (this->data)->get_rows();
        }
        size_t get_col_count()
        {
            std::cout << this->data->get_cols() << std::endl;
            return (this->data)->get_cols();
        }

        void print_dataset(std::shared_ptr<Matrix<double>> matrix = nullptr)
        {
            if (matrix == nullptr)
            {
                matrix = this->data;
            }

            for (size_t i = 0; i < (this->data)->get_rows(); i++)
            {
                for (size_t j = 0; j < (this->data)->get_cols(); j++)
                {
                    std::cout << (this->data)->at(i,j) << " ";
                }
            }
        }

        std::shared_ptr<NN_Data> section(size_t start, size_t end) const;
        std::shared_ptr<NN_Data> ReLU() const;
        std::shared_ptr<NN_Data> softmax() const;
        std::shared_ptr<NN_Data> transpose();
        std::shared_ptr<NN_Data> dot(std::shared_ptr<NN_Data> other);
        std::shared_ptr<Matrix<double>> get_matrix(){return this->data;}
        std::shared_ptr<NN_Data> operator+(std::shared_ptr<NN_Data> obj)
        {
            return std::make_shared<NN_Data>(*(this->get_matrix()) + *(obj->get_matrix()));
        }
};

NN_Data::NN_Data(const std::string datapath): data_path(datapath)
{
    this->data = this->readCSV(this->data_path);
    this->shuffle_dataset();
}

NN_Data::NN_Data(std::shared_ptr<Matrix<double>> matrix)
{
    this->data = matrix;
}

NN_Data::NN_Data(size_t row, size_t col, bool random_matrix)
{
    this->data = std::make_shared<Matrix<double>>(row,col);

    if(random_matrix)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        
        std::uniform_real_distribution<double> dist(-0.5,0.5);
    
        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                this->data->at(i,j) = dist(g);
            }
        }
    }
}

std::shared_ptr<NN_Data> NN_Data::ReLU() const
{
    auto result = std::make_shared<NN_Data>(this->data->get_rows(), this->data->get_cols());

    for (size_t i = 0; i < this->data->get_rows(); ++i) {
        for (size_t j = 0; j < this->data->get_cols(); ++j) {
            result->get_matrix()->at(i, j) = std::max(0.0, this->data->at(i, j)); // ReLU operation
        }
    }

    return result;
}

std::shared_ptr<NN_Data> NN_Data::softmax() const {
    auto result = std::make_shared<NN_Data>(this->data->get_rows(), this->data->get_cols());

    for (size_t i = 0; i < this->data->get_rows(); ++i) {
        // Compute the maximum value in the row for numerical stability
        double row_max = *std::max_element(this->data->at_row(i)->begin(), this->data->at_row(i)->end());

        // Compute exponentials and sum
        std::vector<double> exp_values(this->data->get_cols());
        double sum_exp = 0.0;

        for (size_t j = 0; j < this->data->get_cols(); ++j) {
            exp_values[j] = std::exp(this->data->at(i, j) - row_max); // Subtract row_max for numerical stability
            sum_exp += exp_values[j];
        }

        // Normalize
        for (size_t j = 0; j < this->data->get_cols(); ++j) {
            result->get_matrix()->at(i, j) = exp_values[j] / sum_exp;
        }
    }

    return result;
}


std::shared_ptr<Matrix<double>> NN_Data::readCSV(const std::string& path)
{
    auto matrix = std::vector<std::vector<double>>();
    std::ifstream file(path);

    if(!file.is_open())
    {
        std::cerr << "Unable to open file " << path << std::endl;
        auto ret_matrix = std::make_shared<std::vector<std::vector<double>>>(matrix);
        return std::make_shared<Matrix<double>>(ret_matrix);
    }

    std::string line;

    while (std::getline(file, line)){
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')){
            try{
                row.push_back(std::stod(cell));
                
            }
            catch (const std::exception& e) {
                row.push_back(0);
            }
        }

        matrix.push_back(row);
    }

    file.close();

    // matrix.pop_back();

    auto ret_matrix = std::make_shared<std::vector<std::vector<double>>>(matrix);

    return std::make_shared<Matrix<double>>(ret_matrix);
}

void NN_Data::shuffle_dataset()
{
    this->data = this->data->shuffle();
    return;
}

std::shared_ptr<NN_Data> NN_Data::transpose()
{
    auto retval = std::make_shared<NN_Data>(this->data->transpose());
    
    return retval;
}

std::shared_ptr<NN_Data> NN_Data::dot(std::shared_ptr<NN_Data> other)
{
    return std::make_shared<NN_Data>(this->data->dot(*(other->get_matrix())));
}

std::shared_ptr<NN_Data> NN_Data::section(size_t start, size_t end) const {
    auto retval = std::make_shared<NN_Data>(this->data->section(start, end));
    return retval;
}

NN_Data::~NN_Data()
{

}