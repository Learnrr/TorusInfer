/*
cd tests
nvcc -std=c++17 -O2 \
    -I../ -I../include -I../include/model -I../include/utils \
    test_engine.cpp ../src/Engine.cpp ../src/Scheduler.cpp ../src/KVCacheManager.cpp ../src/Workspace.cpp \
    -o ../build/tests/test_engine.exe
./../build/tests/test_engine.exe
*/

#include "Engine.h"

#include <cassert>
#include <iostream>

namespace {

void TestEngineSingleton() {
    Engine* e1 = Engine::get_instance();
    Engine* e2 = Engine::get_instance();

    assert(e1 != nullptr);
    assert(e2 != nullptr);
    assert(e1 == e2);
}

} // namespace

int main() {
    TestEngineSingleton();
    std::cout << "test_engine passed\n";
    return 0;
}
