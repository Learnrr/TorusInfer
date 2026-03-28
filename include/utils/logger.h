#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <cstdlib>

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

#define LOG_INFO(message) Logger::getInstance().log(LogLevel::INFO, __FILE__, __LINE__, message)
#define LOG_ERROR(message) Logger::getInstance().log(LogLevel::ERROR, __FILE__, __LINE__, message)
#define LOG_WARNING(message) Logger::getInstance().log(LogLevel::WARNING, __FILE__, __LINE__, message)
#define LOG_DEBUG(message) Logger::getInstance().log(LogLevel::DEBUG, __FILE__, __LINE__, message)

class Logger{
    public:
        static Logger& getInstance() {
            static Logger instance;
            return instance;
        }
        ~Logger() {
            if (log_file.is_open()) {
                log_file.close();
            }
        }
        void log(
            const LogLevel level, 
            const char* file,
            size_t line, 
            const std::string& message
        ) {
            if (level >= log_level) {
                std::lock_guard<std::mutex> lock(log_mutex);

                auto now = std::chrono::system_clock::now();
                std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                std::tm tm_now;
                localtime_r(&now_time, &tm_now);
                std::ostringstream oss;
                oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
                std::string timestamp = oss.str();
                std::cout << "[" << timestamp << "] [" << LeveltoString(level) << "] " << file << ":" << line << " - " << message << std::endl;

                if (log_file.is_open()) {
                    log_file << "[" << timestamp << "] [" << LeveltoString(level) << "] " << file << ":" << line << " - " << message << std::endl;
                }
            }
        }


    private:
        Logger(bool to_file = false) {
            log_level = ResolveLogLevelFromEnv();
            if (to_file) {
                log_file.open("log.txt", std::ios::out | std::ios::app);
            }
        };
        
        LogLevel log_level;

        std::string LeveltoString(LogLevel level) {
            switch (level) {
                case LogLevel::INFO: return "INFO";
                case LogLevel::ERROR: return "ERROR";
                case LogLevel::WARNING: return "WARNING";
                case LogLevel::DEBUG: return "DEBUG";
                default: return "UNKNOWN";
            }
        }

        LogLevel ResolveLogLevelFromEnv() {
            const char* raw_level = std::getenv("LOG_LEVEL");
            if (!raw_level || raw_level[0] == '\0') {
                return LogLevel::INFO;
            }
            if (std::strcmp(raw_level, "DEBUG") == 0) {
                return LogLevel::DEBUG;
            }
            if (std::strcmp(raw_level, "INFO") == 0) {
                return LogLevel::INFO;
            }
            if (std::strcmp(raw_level, "WARNING") == 0) {
                return LogLevel::WARNING;
            }
            if (std::strcmp(raw_level, "ERROR") == 0) {
                return LogLevel::ERROR;
            }

            return LogLevel::INFO;
        }
        static Logger* instance;
        std::ofstream log_file;
        std::mutex log_mutex;
};