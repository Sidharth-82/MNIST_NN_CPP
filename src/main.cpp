#include "NNData.cpp"

std::shared_ptr<std::vector<NN_Data>> forward_propagation(std::shared_ptr<NN_Data> X, 
                         std::shared_ptr<NN_Data> W1,
                         std::shared_ptr<NN_Data> b1, 
                         std::shared_ptr<NN_Data> W2, 
                         std::shared_ptr<NN_Data> b2)
{
    auto Z1 = *(W1->dot(X)) + b1;
    auto A1 = Z1->ReLU();
    auto Z2 = *(W2->dot(A1)) + b2;
    auto A2 = Z2->softmax();
    std::vector<NN_Data> retval = {*Z1,*A1,*Z2,*A2};
    return std::make_shared<std::vector<NN_Data>>(retval);
}

std::shared_ptr<std::vector<NN_Data>> back_propagation(std::shared_ptr<NN_Data> Y, 
    std::shared_ptr<NN_Data> Z1,
    std::shared_ptr<NN_Data> A1, 
    std::shared_ptr<NN_Data> Z2, 
    std::shared_ptr<NN_Data> A2,
    std::shared_ptr<NN_Data> W2)
{
    
    std::vector<NN_Data> retval = {};
    return std::make_shared<std::vector<NN_Data>>(retval);
}

int main()
{

    const auto dataset = std::make_shared<NN_Data>("C:\\Users\\sidha\\OneDrive\\programming\\AI ML CV\\NN_from_scratch\\dataset\\mnist_train.csv");
    
    auto rows = dataset->get_row_count();
    auto cols = dataset->get_col_count();
    
    auto dev_set = dataset->section(0, 1000)->transpose();
    auto Y_dev = dev_set->section(0, 1);
    auto X_dev = dev_set->section(1, cols);
    
    // Y_dev->print_dataset();
    // X_dev->print_dataset();
    dev_set->get_row_count();
    dev_set->get_col_count();

    auto train_set = dataset->section(1000, rows)->transpose();
    auto Y_train = train_set->section(0,1);
    auto X_train = train_set->section(1,dataset->get_col_count());
    
    // Y_train->print_dataset();
    // X_train->print_dataset();
    train_set->get_row_count();
    train_set->get_col_count();
    

    auto W1 = std::make_shared<NN_Data>(10, 784);
    auto b1 = std::make_shared<NN_Data>(10, 1);
    auto W2 = std::make_shared<NN_Data>(10, 10);
    auto b2 = std::make_shared<NN_Data>(10, 1);

    return 1;
}