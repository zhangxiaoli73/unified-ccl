#include <pybind11/pybind11.h>
#include "uccl.h"

#include <stdexcept>
#include <cstdint>
#include <string>

namespace py = pybind11;

/* pybind11 bindings for Unified-CCL.
 * Exposes the C API to Python for use with torch.Tensor. */

PYBIND11_MODULE(_uccl_bindings, m) {
    m.doc() = "Unified-CCL Python bindings";

    /* Version */
    m.def("get_version", []() {
        int version;
        ucclResult_t res = ucclGetVersion(&version);
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
        return version;
    }, "Get UCCL version number");

    /* Communicator init */
    m.def("comm_init_rank", [](int nranks, int rank) -> uintptr_t {
        ucclUniqueId id;
        if (rank == 0) {
            ucclResult_t res = ucclGetUniqueId(&id);
            if (res != ucclSuccess) {
                throw std::runtime_error(
                    std::string("GetUniqueId failed: ") +
                    ucclGetErrorString(res));
            }
        }

        /* Broadcast ID to all ranks via MPI
         * In Python usage, each process calls this independently.
         * The MPI bootstrap inside CommInitRank handles coordination. */

        ucclComm_t comm;
        ucclResult_t res = ucclCommInitRank(&comm, nranks, id, rank);
        if (res != ucclSuccess) {
            throw std::runtime_error(
                std::string("CommInitRank failed: ") +
                ucclGetErrorString(res));
        }
        return reinterpret_cast<uintptr_t>(comm);
    }, py::arg("nranks"), py::arg("rank"),
       "Initialize communicator for given rank");

    /* AllReduce */
    m.def("allreduce", [](uintptr_t comm_handle,
                          uintptr_t sendbuff,
                          uintptr_t recvbuff,
                          size_t count,
                          py::object dtype,
                          std::string op) {
        ucclComm_t comm = reinterpret_cast<ucclComm_t>(comm_handle);

        /* Map torch dtype to ucclDataType_t */
        ucclDataType_t dt = ucclFloat16; /* default */
        std::string dtStr = py::str(dtype);
        if (dtStr.find("bfloat16") != std::string::npos) {
            dt = ucclBfloat16;
        } else if (dtStr.find("float16") != std::string::npos) {
            dt = ucclFloat16;
        } else {
            throw std::runtime_error(
                "Unsupported dtype: " + dtStr +
                ". Supported: float16, bfloat16");
        }

        /* Map op string to ucclRedOp_t */
        ucclRedOp_t redOp = ucclSum;
        if (op != "sum") {
            throw std::runtime_error(
                "Unsupported reduction op: " + op +
                ". Supported: sum");
        }

        ucclResult_t res = ucclAllReduce(
            reinterpret_cast<const void*>(sendbuff),
            reinterpret_cast<void*>(recvbuff),
            count, dt, redOp, comm,
            nullptr /* use default stream */);

        if (res != ucclSuccess) {
            throw std::runtime_error(
                std::string("AllReduce failed: ") +
                ucclGetErrorString(res));
        }
    }, py::arg("comm"), py::arg("sendbuff"), py::arg("recvbuff"),
       py::arg("count"), py::arg("dtype"), py::arg("op"),
       "Perform AllReduce operation");

    /* AllGather */
    m.def("allgather", [](uintptr_t comm_handle,
                          uintptr_t sendbuff,
                          uintptr_t recvbuff,
                          size_t sendcount,
                          py::object dtype) {
        ucclComm_t comm = reinterpret_cast<ucclComm_t>(comm_handle);

        ucclDataType_t dt = ucclFloat16;
        std::string dtStr = py::str(dtype);
        if (dtStr.find("bfloat16") != std::string::npos) {
            dt = ucclBfloat16;
        } else if (dtStr.find("float16") != std::string::npos) {
            dt = ucclFloat16;
        } else {
            throw std::runtime_error(
                "Unsupported dtype: " + dtStr +
                ". Supported: float16, bfloat16");
        }

        ucclResult_t res = ucclAllGather(
            reinterpret_cast<const void*>(sendbuff),
            reinterpret_cast<void*>(recvbuff),
            sendcount, dt, comm,
            nullptr);

        if (res != ucclSuccess) {
            throw std::runtime_error(
                std::string("AllGather failed: ") +
                ucclGetErrorString(res));
        }
    }, py::arg("comm"), py::arg("sendbuff"), py::arg("recvbuff"),
       py::arg("sendcount"), py::arg("dtype"),
       "Perform AllGather operation");

    /* ReduceScatter */
    m.def("reduce_scatter", [](uintptr_t comm_handle,
                               uintptr_t sendbuff,
                               uintptr_t recvbuff,
                               size_t recvcount,
                               py::object dtype,
                               std::string op) {
        ucclComm_t comm = reinterpret_cast<ucclComm_t>(comm_handle);

        ucclDataType_t dt = ucclFloat16;
        std::string dtStr = py::str(dtype);
        if (dtStr.find("bfloat16") != std::string::npos) {
            dt = ucclBfloat16;
        } else if (dtStr.find("float16") != std::string::npos) {
            dt = ucclFloat16;
        } else {
            throw std::runtime_error(
                "Unsupported dtype: " + dtStr +
                ". Supported: float16, bfloat16");
        }

        ucclRedOp_t redOp = ucclSum;
        if (op != "sum") {
            throw std::runtime_error(
                "Unsupported reduction op: " + op +
                ". Supported: sum");
        }

        ucclResult_t res = ucclReduceScatter(
            reinterpret_cast<const void*>(sendbuff),
            reinterpret_cast<void*>(recvbuff),
            recvcount, dt, redOp, comm,
            nullptr);

        if (res != ucclSuccess) {
            throw std::runtime_error(
                std::string("ReduceScatter failed: ") +
                ucclGetErrorString(res));
        }
    }, py::arg("comm"), py::arg("sendbuff"), py::arg("recvbuff"),
       py::arg("recvcount"), py::arg("dtype"), py::arg("op"),
       "Perform ReduceScatter operation");

    /* Comm queries */
    m.def("comm_count", [](uintptr_t h) {
        int count;
        ucclResult_t res = ucclCommCount(
            reinterpret_cast<ucclComm_t>(h), &count);
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
        return count;
    }, py::arg("comm"), "Get number of ranks in communicator");

    m.def("comm_rank", [](uintptr_t h) {
        int rank;
        ucclResult_t res = ucclCommUserRank(
            reinterpret_cast<ucclComm_t>(h), &rank);
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
        return rank;
    }, py::arg("comm"), "Get rank of this communicator");

    /* Comm lifecycle */
    m.def("comm_finalize", [](uintptr_t h) {
        ucclResult_t res = ucclCommFinalize(
            reinterpret_cast<ucclComm_t>(h));
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
    }, py::arg("comm"), "Finalize communicator");

    m.def("comm_destroy", [](uintptr_t h) {
        ucclResult_t res = ucclCommDestroy(
            reinterpret_cast<ucclComm_t>(h));
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
    }, py::arg("comm"), "Destroy communicator");

    m.def("comm_abort", [](uintptr_t h) {
        ucclResult_t res = ucclCommAbort(
            reinterpret_cast<ucclComm_t>(h));
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
    }, py::arg("comm"), "Abort communicator");

    /* Group semantics */
    m.def("group_start", []() {
        ucclResult_t res = ucclGroupStart();
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
    }, "Start group operation");

    m.def("group_end", []() {
        ucclResult_t res = ucclGroupEnd();
        if (res != ucclSuccess) {
            throw std::runtime_error(ucclGetErrorString(res));
        }
    }, "End group operation");

    /* Error string */
    m.def("get_error_string", [](int result) {
        return std::string(ucclGetErrorString(
            static_cast<ucclResult_t>(result)));
    }, py::arg("result"), "Get human-readable error string");
}
