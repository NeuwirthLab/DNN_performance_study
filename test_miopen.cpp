#include <miopen/miopen.h>
#include <iostream>
int main() {
    miopenTensorDescriptor_t desc;
    miopenCreateTensorDescriptor(&desc);
    size_t sz = 0;
    miopenGetTensorNumBytes(desc, &sz);
    std::cout << "Success!" << std::endl;
    return 0;
}
