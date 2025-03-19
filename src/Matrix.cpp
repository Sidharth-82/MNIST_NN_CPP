#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iomanip>
#include <random>
#include <algorithm>

template <typename T>
class Matrix {
private:
    std::shared_ptr<std::vector<std::vector<T>>> data;
    size_t rows, cols;

public:
    // Constructor
    Matrix(size_t n, size_t m, T init_val = T()) 
        : rows(n), cols(m), data(std::make_shared<std::vector<std::vector<T>>>(n, std::vector<T>(m, init_val))) {}
    
    // Constructor
    Matrix(std::shared_ptr<std::vector<std::vector<T>>> matrix) 
        : data(matrix) {
            rows = data->size();
            cols = (*data)[0].size();
        }

    // Getters for dimensions
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }

    // Accessor
    T& at(size_t i, size_t j) {
        if (i >= rows || j >= cols) 
        {
            std::cerr << "Out of bounds access: (" << i << ", " << j << ")\n";
            throw std::out_of_range("Index out of bounds");
        }
        return (*data)[i][j];
    }

    const T& at(size_t i, size_t j) const {
        if (i >= rows || j >= cols)
        {
            std::cerr << "Out of bounds access: (" << i << ", " << j << ")\n";
            throw std::out_of_range("Index out of bounds");
        }
        return (*data)[i][j];
    }

    std::shared_ptr<std::vector<T>> at_row(size_t i) {
        if (i >= rows)
        {
            std::cerr << "Out of bounds access: (" << i << ")\n";
            throw std::out_of_range("Index out of bounds");
        }
        return std::make_shared<std::vector<T>>((*data)[i]);
    }

    void push_row_back(std::vector<T> row)
    {
        (this->data)->push_back(row);
        (this->rows)++;
    }

    void push_col_back(std::vector<T> col)
    {
        if(col.size() < this->get_rows()) throw std::out_of_range("Not enough values to fill coloums");

        for(int i = 0; i < this->get_rows(); i++)
        {
            this->data->at(i).push_back(col);
        }
    }

    // Transpose function
    std::shared_ptr<Matrix<T>> transpose() const {
        std::shared_ptr<Matrix<T>> transposed = std::make_shared<Matrix<T>>(this->get_cols(), this->get_rows());
        
        for (size_t i = 0; i < this->get_rows(); i++) {
            for (size_t j = 0; j < this->get_cols(); j++) {
                // std::cout <<"i: "<< i <<" j:" << j << std::endl;
                transposed->at(j, i) = this->at(i, j);
            }
        }
        // std::cout <<"T Rows: "<< transposed->get_rows()<<" T Cols:" << transposed->get_cols() << std::endl;
        return transposed;
    }

    // Dot product (Matrix Multiplication)
    std::shared_ptr<Matrix<T>> dot(const Matrix<T>& other) const {
        if (cols != other.rows) throw std::invalid_argument("Incompatible matrix dimensions for dot product");

        auto result = std::make_shared<Matrix<T>>(rows, other.cols, T());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result->at(i, j) += at(i, k) * other.at(k, j);
                }
            }
        }
        return result;
    }

    std::shared_ptr<Matrix<T>> operator+(const Matrix<T>& other) const {
        if (cols != other.get_cols() && rows != other.get_rows()) throw std::invalid_argument("Incompatible matrix dimensions for addition product");

        auto result = std::make_shared<Matrix<T>>(rows, cols, T());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result->at(i, j) = this->at(i, j) + other.at(i, j);
            }
        }
        return result;
    }

    std::shared_ptr<Matrix<T>> shuffle() const {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(this->data->begin(), this->data->end(), g);
        return std::make_shared<Matrix<T>>(this->data);
    }

    std::shared_ptr<Matrix<T>> section(size_t start, size_t end) const {
        if (start < 0 || end > this->data->size() || start >= end) {
            throw std::out_of_range("Invalid slice range");
        }
        auto sectioned_matrix = std::make_shared<std::vector<std::vector<T>>>(this->data->begin() + start, this->data->begin() + end);
        // std::cout <<"S Rows: "<< sectioned_matrix->size()<<" S Cols:" << (*sectioned_matrix)[0].size() << std::endl;
        return std::make_shared<Matrix<T>>(sectioned_matrix);
    }

    // Print function
    void print() const {
        for (const auto& row : *data) {
            for (const auto& elem : row) {
                std::cout << std::setw(8) << elem << " ";
            }
            std::cout << '\n';
        }
    }

    
};

// int main() {
//     // Example usage
//     auto mat1 = std::make_shared<Matrix<double>>(3, 2, 0.5);
//     mat1->at(0, 1) = -0.2;
//     mat1->at(1, 0) = 0.8;
//     mat1->print();

//     std::cout << "\nTransposed:\n";
//     auto transposed = mat1->transpose();
//     transposed->print();

//     auto mat2 = std::make_shared<Matrix<double>>(2, 3, 0.1);
//     auto dot_product = mat1->dot(*mat2);
//     std::cout << "\nDot Product:\n";
//     dot_product->print();

//     return 0;
// }
