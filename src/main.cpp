#include <exception>
#include <iostream>

#include "app/Application.hpp"

int main() {
    try {
        spp::Application app(std::cin, std::cout);
        return app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Fatal error: unknown exception\n";
        return 1;
    }
}