#pragma once

#include <chrono>
#include <iostream>

class Clock {
  public:
    Clock() : m_start(std::chrono::high_resolution_clock::now()) {}

    void start() { m_start = std::chrono::high_resolution_clock::now(); }

    void reset() { m_start = std::chrono::high_resolution_clock::now(); }

    void time_us(const std::string &message) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - m_start);
        std::cout << message << " " << duration.count() << "us" << std::endl;
    }

    void time_ms(const std::string &message) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - m_start);
        std::cout << message << " " << duration.count() << "ms" << std::endl;
    }

    void time_s(const std::string &message) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end - m_start);
        std::cout << message << " " << duration.count() << "s" << std::endl;
    }

    void time_min(const std::string &message){
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(
            end - m_start);
        std::cout << message << " " << duration.count() << "min" << std::endl;
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};


