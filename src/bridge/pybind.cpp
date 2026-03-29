
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Engine.h"

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(cpp_engine, m) {
    py::class_<SequenceOutput>(m, "SequenceOutput")
        .def(py::init<>())
        .def_readwrite("seq_id", &SequenceOutput::seq_id)
        .def_readwrite("token_ids", &SequenceOutput::token_ids);

    py::class_<Engine>(m, "Engine")
        .def_static(
            "get_instance",
            []() -> Engine& {
                return *Engine::get_instance();
            },
            py::return_value_policy::reference,
            "Get process-wide Engine singleton"
        )
        .def("init",
             [](Engine& self, const std::string& llm_engine_config_path) {
                 self.init(const_cast<char*>(llm_engine_config_path.c_str()));
             },
             py::arg("llm_engine_config_path"),
             "Initialize engine with config path")
        .def("run", &Engine::run, "Start scheduler thread")
        .def("create_request",
             [](Engine& self, const std::vector<size_t>& token_ids) {
                 size_t request_id = 0;
                 self.create_request(token_ids, request_id);
                 return request_id;
             },
             py::arg("token_ids"),
             "Create request and return request_id")
        .def("submit_request", &Engine::submit_request, py::arg("request_id"),
             "Submit request to scheduler")
        .def("get_request_output",
             [](Engine& self, size_t request_id) {
                 SequenceOutput output{};
                 self.get_request_output(request_id, output);
                 return output;
             },
             py::arg("request_id"),
             "Get output for request")
        .def("check_request_state",
             [](Engine& self, size_t request_id) {
                 RequestStatus state;
                 self.check_request_state(request_id, state);
                 return state;
             },
             py::arg("request_id"),
             "Check request state");

    py::enum_<RequestStatus>(m, "RequestStatus")
        .value("PENDING", RequestStatus::PENDING)
        .value("IN_PROGRESS", RequestStatus::IN_PROGRESS)
        .value("COMPLETED", RequestStatus::COMPLETED)
        .value("CANCELLED", RequestStatus::CANCELLED)
        .value("FAILED", RequestStatus::FAILED);
}