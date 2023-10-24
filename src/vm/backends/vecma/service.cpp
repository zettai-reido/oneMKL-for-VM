#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "xtypes.hpp"

namespace vecma::detail {

using Mode64T = uint64_t; // for compatibility with M
using ErrorKey = StreamIdT;

class Singleton {
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    std::string m_no_error;

    std::unordered_map<ErrorKey, std::string> m_last_cuda_error;

    std::unordered_map<StreamIdT, Mode64T> m_mode64;

    std::unordered_map<StreamIdT, Mode> m_mode;
    std::unordered_map<StreamIdT, AtomicStatusT> m_status;

    Mode m_default_mode;

    std::mutex m_state_mutex;

public:
    Singleton(): m_no_error("no_error") {
        cudaFree(0); // establish a CUDA context if not
        m_default_mode = Mode(
            Indexing::kNormal,
            GlobalReport::kQuiet
        );
    }

    static Singleton& get() { static Singleton si; return si; }

    Status get_status(StreamIdT stream_id) {
        std::lock_guard<std::mutex> lock(m_state_mutex);

        StatusT st_ret = static_cast<StatusT>(Status::kSuccess);
        auto i = m_status.find(stream_id);
        if (i != m_status.end()) { st_ret = i->second; }
        return static_cast<Status>(st_ret);
    }

    Status set_status(StreamIdT stream_id, Status new_status) {
        std::lock_guard<std::mutex> lock(m_state_mutex);

        StatusT st_ret = static_cast<StatusT>(Status::kSuccess);
        StatusT st_new = static_cast<StatusT>(new_status);

        auto i = m_status.find(stream_id);
        if (i == m_status.end()) {
            m_status.emplace(stream_id, st_new);
        } else {
            st_ret = m_status[stream_id];
            m_status[stream_id] = st_new;
        }

        return static_cast<Status>(st_ret);
    }

    Mode get_mode(StreamIdT stream_id) {
        Mode mode_ret;

        {
            std::lock_guard<std::mutex> lock(m_state_mutex);

            auto i = m_mode.find(stream_id);
            mode_ret = (i != m_mode.end()) ? i->second : Mode();
        }
        mode_ret += m_default_mode;
        return mode_ret;
    }

    Mode set_mode(StreamIdT stream_id, Mode new_mode) {
        Mode mode_ret;

        {
            std::lock_guard<std::mutex> lock(m_state_mutex);

            auto i = m_mode.find(stream_id);
            mode_ret = (i != m_mode.end()) ? i->second : m_default_mode;
            new_mode += mode_ret;
            m_mode.emplace(stream_id, new_mode);
        }
        return mode_ret;
    }

    Mode64T get_mode64(StreamIdT stream_id, Mode64T mode) {
        Mode64T mode_ret;

        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            auto i = m_mode64.find(stream_id);
            mode_ret = (i != m_mode64.end()) ? i->second : Mode64T(0);
        }
        return mode_ret;
    }

    Mode64T set_mode64(StreamIdT stream_id, Mode64T new_mode) {
        Mode64T mode_ret;

        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            auto i = m_mode64.find(stream_id);
            mode_ret = (i != m_mode64.end()) ? i->second : Mode64T(0);
            m_mode64.emplace(stream_id, new_mode);
        }

        return mode_ret;
    }

    std::string const& get_last_cuda_error(StreamIdT stream_id) {
        std::lock_guard<std::mutex> lock(m_state_mutex);

        auto i = m_last_cuda_error.find(stream_id);
        if (i != m_last_cuda_error.end()) { return i->second; }
        return m_no_error;
    }

    bool get_current_context(StreamIdT stream_id, CUcontext* ctx) {
        CUresult r = cuCtxGetCurrent(ctx);

        if (r != CUDA_SUCCESS) { 
            const char* desc = nullptr;

            cuGetErrorString(r, &desc);

            if (nullptr != desc) {
                std::lock_guard<std::mutex> lock(m_state_mutex);
                m_last_cuda_error.emplace(stream_id, desc);
            }
            return false;
        }
        return true;
    }
}; // class Singleton

Status get_status(StreamIdT stream_id) { return Singleton::get().get_status(stream_id); }
Status clear_status(StreamIdT stream_id) { return Singleton::get().set_status(stream_id, Status::kNotDefined); }
Status set_status(StreamIdT stream_id, Status new_status) { return Singleton::get().set_status(stream_id, new_status); }

std::string const& get_last_cuda_error(StreamIdT stream_id) { return Singleton::get().get_last_cuda_error(stream_id); }

} // namespace vecma::detail

