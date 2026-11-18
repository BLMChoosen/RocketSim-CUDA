#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "rocketsim_cuda/config.h"
#include "rocketsim_cuda/sim_context.cuh"
#include "rocketsim_cuda/types/arena_state.cuh"
#include "rocketsim_cuda/types/game_event_tracker.cuh"
#include "rocketsim_cuda/physics/arena_sdf.cuh"

namespace nb = nanobind;
using namespace rocketsim_cuda;

// CUDA Asynchronous Hardware Event Wrapper for non-blocking benchmarking & synchronization
class GpuEvent {
public:
    cudaEvent_t m_event = nullptr;
    bool m_enable_timing = true;

    explicit GpuEvent(bool enable_timing = true) : m_enable_timing(enable_timing) {
        unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
        cudaEventCreateWithFlags(&m_event, flags);
    }

    ~GpuEvent() {
        if (m_event) {
            cudaEventDestroy(m_event);
            m_event = nullptr;
        }
    }

    // Disable copy, allow move
    GpuEvent(const GpuEvent&) = delete;
    GpuEvent& operator=(const GpuEvent&) = delete;
    GpuEvent(GpuEvent&& o) noexcept : m_event(o.m_event), m_enable_timing(o.m_enable_timing) {
        o.m_event = nullptr;
    }
    GpuEvent& operator=(GpuEvent&& o) noexcept {
        if (this != &o) {
            if (m_event) cudaEventDestroy(m_event);
            m_event = o.m_event;
            m_enable_timing = o.m_enable_timing;
            o.m_event = nullptr;
        }
        return *this;
    }

    void record(nb::object stream = nb::none()) {
        cudaStream_t s = nullptr;
        if (!stream.is_none()) {
            if (nb::isinstance<nb::int_>(stream)) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream));
            } else if (nb::hasattr(stream, "cuda_stream")) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream.attr("cuda_stream")));
            }
        }
        cudaEventRecord(m_event, s);
    }

    void wait(nb::object stream = nb::none()) {
        cudaStream_t s = nullptr;
        if (!stream.is_none()) {
            if (nb::isinstance<nb::int_>(stream)) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream));
            } else if (nb::hasattr(stream, "cuda_stream")) {
                s = reinterpret_cast<cudaStream_t>(nb::cast<uintptr_t>(stream.attr("cuda_stream")));
            }
        }
        cudaStreamWaitEvent(s, m_event, 0);
    }

    void synchronize() {
        cudaEventSynchronize(m_event);
    }

    bool query() const {
        return cudaEventQuery(m_event) == cudaSuccess;
    }

    float elapsed_time(const GpuEvent& end) const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, m_event, end.m_event);
        return ms;
    }
};

// Zero-copy DLPack-compliant GPU Tensor View structure
struct GpuTensorView {
    void* data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::string dtype = "float32";
    int device_type = 2; // kDLCUDA
    int device_id = 0;
    nb::object owner;

    size_t get_element_size() const {
        if (dtype == "uint8") return 1;
        if (dtype == "int64") return 8;
        return 4; // float32, int32, uint32
    }

    int64_t total_elements() const {
        int64_t count = 1;
        for (auto s : shape) count *= s;
        return count;
    }

    nb::object getitem(nb::object idx_obj) const {
        std::vector<int64_t> indices;
        if (nb::isinstance<nb::tuple>(idx_obj)) {
            nb::tuple t = nb::cast<nb::tuple>(idx_obj);
            for (size_t i = 0; i < t.size(); ++i) {
                indices.push_back(nb::cast<int64_t>(t[i]));
            }
        } else if (nb::isinstance<nb::int_>(idx_obj)) {
            indices.push_back(nb::cast<int64_t>(idx_obj));
        } else {
            throw std::invalid_argument("Index must be an integer or tuple of integers");
        }

        if (indices.size() > shape.size()) {
            throw std::invalid_argument("Too many indices for tensor dimension");
        }

        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t idx = indices[i];
            if (idx < 0) idx += shape[i];
            if (idx < 0 || idx >= shape[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += idx * strides[i];
        }

        if (indices.size() == shape.size()) {
            if (dtype == "float32") {
                float val = 0.0f;
                cudaMemcpy(&val, static_cast<const float*>(data) + offset, sizeof(float), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "uint8") {
                uint8_t val = 0;
                cudaMemcpy(&val, static_cast<const uint8_t*>(data) + offset, sizeof(uint8_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "uint32") {
                uint32_t val = 0;
                cudaMemcpy(&val, static_cast<const uint32_t*>(data) + offset, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "int32") {
                int32_t val = 0;
                cudaMemcpy(&val, static_cast<const int32_t*>(data) + offset, sizeof(int32_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            } else if (dtype == "int64") {
                int64_t val = 0;
                cudaMemcpy(&val, static_cast<const int64_t*>(data) + offset, sizeof(int64_t), cudaMemcpyDeviceToHost);
                return nb::cast(val);
            }
            throw std::runtime_error("Unsupported dtype");
        } else {
            GpuTensorView sub;
            sub.data = static_cast<char*>(data) + offset * get_element_size();
            sub.shape.assign(shape.begin() + indices.size(), shape.end());
            sub.strides.assign(strides.begin() + indices.size(), strides.end());
            sub.dtype = dtype;
            sub.device_type = device_type;
            sub.device_id = device_id;
            sub.owner = owner;
            return nb::cast(sub);
        }
    }

    void setitem(nb::object idx_obj, nb::object val_obj) {
        std::vector<int64_t> indices;
        if (nb::isinstance<nb::tuple>(idx_obj)) {
            nb::tuple t = nb::cast<nb::tuple>(idx_obj);
            for (size_t i = 0; i < t.size(); ++i) {
                indices.push_back(nb::cast<int64_t>(t[i]));
            }
        } else if (nb::isinstance<nb::int_>(idx_obj)) {
            indices.push_back(nb::cast<int64_t>(idx_obj));
        } else {
            throw std::invalid_argument("Index must be an integer or tuple of integers");
        }

        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Assignment requires all dimensional indices");
        }

        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t idx = indices[i];
            if (idx < 0) idx += shape[i];
            if (idx < 0 || idx >= shape[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += idx * strides[i];
        }

        if (dtype == "float32") {
            float val = nb::cast<float>(val_obj);
            cudaMemcpy(static_cast<float*>(data) + offset, &val, sizeof(float), cudaMemcpyHostToDevice);
        } else if (dtype == "uint8") {
            uint8_t val = nb::cast<uint8_t>(val_obj);
            cudaMemcpy(static_cast<uint8_t*>(data) + offset, &val, sizeof(uint8_t), cudaMemcpyHostToDevice);
        } else if (dtype == "uint32") {
            uint32_t val = nb::cast<uint32_t>(val_obj);
            cudaMemcpy(static_cast<uint32_t*>(data) + offset, &val, sizeof(uint32_t), cudaMemcpyHostToDevice);
        } else if (dtype == "int32") {
            int32_t val = nb::cast<int32_t>(val_obj);
            cudaMemcpy(static_cast<int32_t*>(data) + offset, &val, sizeof(int32_t), cudaMemcpyHostToDevice);
        } else if (dtype == "int64") {
            int64_t val = nb::cast<int64_t>(val_obj);
            cudaMemcpy(static_cast<int64_t*>(data) + offset, &val, sizeof(int64_t), cudaMemcpyHostToDevice);
        }
    }

    void zero_() {
        int64_t total_bytes = total_elements() * get_element_size();
        cudaMemset(data, 0, total_bytes);
    }

    GpuTensorView clone() const {
        int64_t total = total_elements();
        size_t elem_size = get_element_size();
        size_t total_bytes = total * elem_size;

        void* d_new = nullptr;
        cudaMalloc(&d_new, total_bytes);
        cudaMemcpy(d_new, data, total_bytes, cudaMemcpyDeviceToDevice);

        GpuTensorView c;
        c.data = d_new;
        c.shape = shape;
        c.strides = strides;
        c.dtype = dtype;
        c.device_type = device_type;
        c.device_id = device_id;
        nb::capsule cleanup(d_new, [](void* p) noexcept {
            if (p) cudaFree(p);
        });
        c.owner = cleanup;
        return c;
    }

    nb::object to_dlpack(nb::object /* stream */ = nb::none()) const {
        std::vector<size_t> u_shape(shape.begin(), shape.end());
        nb::dlpack::dtype dt = nanobind::dtype<float>();
        if (dtype == "uint8") {
            dt = nanobind::dtype<uint8_t>();
        } else if (dtype == "int32") {
            dt = nanobind::dtype<int32_t>();
        } else if (dtype == "uint32") {
            dt = nanobind::dtype<uint32_t>();
        } else if (dtype == "int64") {
            dt = nanobind::dtype<int64_t>();
        }

        nb::ndarray<nb::device::cuda> arr(
            data,
            u_shape.size(),
            u_shape.data(),
            owner,
            strides.data(),
            dt,
            device_type,
            device_id
        );
        return nb::cast(arr);
    }
};

// State Setter & Prediction Tensor Unpacking Helpers
struct TensorBufferInfo {
    void* d_ptr = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::string dtype = "float32";
    bool is_cuda = true;
    bool needs_free = false;
};

inline TensorBufferInfo get_tensor_buffer(nb::object obj, cudaStream_t stream = nullptr) {
    TensorBufferInfo info;
    if (obj.is_none()) return info;

    if (nb::isinstance<GpuTensorView>(obj)) {
        const auto& v = nb::cast<const GpuTensorView&>(obj);
        info.d_ptr = v.data;
        info.shape = v.shape;
        info.strides = v.strides;
        info.dtype = v.dtype;
        info.is_cuda = true;
        return info;
    }

    if (nb::hasattr(obj, "data_ptr") && nb::hasattr(obj, "shape")) {
        if (nb::hasattr(obj, "is_cuda")) {
            info.is_cuda = nb::cast<bool>(obj.attr("is_cuda"));
        } else {
            info.is_cuda = true;
        }

        if (!info.is_cuda && nb::hasattr(obj, "is_contiguous")) {
            if (!nb::cast<bool>(obj.attr("is_contiguous")())) {
                obj = obj.attr("contiguous")();
            }
        }

        uintptr_t ptr = nb::cast<uintptr_t>(obj.attr("data_ptr")());
        info.d_ptr = reinterpret_cast<void*>(ptr);
        nb::object sh = nb::borrow(obj.attr("shape"));
        size_t rank = nb::len(sh);
        for (size_t i = 0; i < rank; ++i) {
            info.shape.push_back(nb::cast<int64_t>(sh[nb::int_(i)]));
        }
        if (nb::hasattr(obj, "stride")) {
            nb::object st = nb::borrow(obj.attr("stride"));
            if (nb::isinstance<nb::callable>(st)) {
                nb::object s_ret = st();
                size_t n_st = nb::len(s_ret);
                for (size_t i = 0; i < n_st; ++i) {
                    info.strides.push_back(nb::cast<int64_t>(s_ret[nb::int_(i)]));
                }
            } else if (nb::isinstance<nb::tuple>(st)) {
                size_t n_st = nb::len(st);
                for (size_t i = 0; i < n_st; ++i) {
                    info.strides.push_back(nb::cast<int64_t>(st[nb::int_(i)]));
                }
            }
        }
        if (info.strides.empty()) {
            int64_t cur = 1;
            info.strides.resize(info.shape.size());
            for (int i = static_cast<int>(info.shape.size()) - 1; i >= 0; --i) {
                info.strides[i] = cur;
                cur *= info.shape[i];
            }
        }
        if (nb::hasattr(obj, "dtype")) {
            std::string dt_str = nb::cast<std::string>(obj.attr("dtype").attr("__str__")());
            if (dt_str.find("uint8") != std::string::npos || dt_str.find("bool") != std::string::npos || dt_str.find("byte") != std::string::npos) {
                info.dtype = "uint8";
            } else if (dt_str.find("int32") != std::string::npos) {
                info.dtype = "int32";
            } else if (dt_str.find("int64") != std::string::npos || dt_str.find("long") != std::string::npos) {
                info.dtype = "int64";
            } else {
                info.dtype = "float32";
            }
        }

        if (!info.is_cuda) {
            size_t elem_size = (info.dtype == "uint8") ? 1 : ((info.dtype == "int64") ? 8 : 4);
            int64_t total = 1;
            for (auto s : info.shape) total *= s;
            size_t total_bytes = total * elem_size;
            void* d_tmp = nullptr;
            cudaMalloc(&d_tmp, total_bytes);
            cudaMemcpyAsync(d_tmp, info.d_ptr, total_bytes, cudaMemcpyHostToDevice, stream);
            info.d_ptr = d_tmp;
            info.is_cuda = true;
            info.needs_free = true;
        }
        return info;
    }

    if (nb::isinstance<nb::list>(obj)) {
        nb::list outer = nb::cast<nb::list>(obj);
        if (outer.size() > 0) {
            if (nb::isinstance<nb::list>(outer[0])) {
                int64_t rows = outer.size();
                nb::list r0 = nb::cast<nb::list>(outer[0]);
                int64_t cols = r0.size();
                std::vector<float> h_buf(rows * cols);
                for (int64_t r = 0; r < rows; ++r) {
                    nb::list row_list = nb::cast<nb::list>(outer[r]);
                    for (int64_t c = 0; c < cols; ++c) {
                        h_buf[r * cols + c] = nb::cast<float>(row_list[c]);
                    }
                }
                void* d_tmp = nullptr;
                cudaMalloc(&d_tmp, rows * cols * sizeof(float));
                cudaMemcpyAsync(d_tmp, h_buf.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream);
                info.d_ptr = d_tmp;
                info.shape = { rows, cols };
                info.strides = { cols, 1 };
                info.dtype = "float32";
                info.is_cuda = true;
                info.needs_free = true;
                return info;
            } else {
                int64_t cols = outer.size();
                std::vector<float> h_buf(cols);
                for (int64_t c = 0; c < cols; ++c) {
                    h_buf[c] = nb::cast<float>(outer[c]);
                }
                void* d_tmp = nullptr;
                cudaMalloc(&d_tmp, cols * sizeof(float));
                cudaMemcpyAsync(d_tmp, h_buf.data(), cols * sizeof(float), cudaMemcpyHostToDevice, stream);
                info.d_ptr = d_tmp;
                info.shape = { cols };
                info.strides = { 1 };
                info.dtype = "float32";
                info.is_cuda = true;
                info.needs_free = true;
                return info;
            }
        }
    }

    return info;
}

inline void release_tensor_buffer(TensorBufferInfo& info) {
    if (info.needs_free && info.d_ptr) {
        cudaFree(info.d_ptr);
        info.d_ptr = nullptr;
        info.needs_free = false;
    }
}

struct IndicesBufferInfo {
    const int32_t* d_indices = nullptr;
    uint32_t count = 0;
    bool is_provided = false;
    bool needs_free = false;
};

inline IndicesBufferInfo get_indices_buffer(nb::object indices_obj, cudaStream_t stream = nullptr) {
    IndicesBufferInfo res;
    if (indices_obj.is_none()) return res;
    res.is_provided = true;

    if (nb::isinstance<GpuTensorView>(indices_obj)) {
        const auto& v = nb::cast<const GpuTensorView&>(indices_obj);
        res.count = static_cast<uint32_t>(v.shape[0]);
        if (res.count > 0) {
            res.d_indices = reinterpret_cast<const int32_t*>(v.data);
        }
        return res;
    }

    if (nb::hasattr(indices_obj, "data_ptr") && nb::hasattr(indices_obj, "shape")) {
        bool is_cuda = nb::hasattr(indices_obj, "is_cuda") ? nb::cast<bool>(indices_obj.attr("is_cuda")) : true;
        if (!is_cuda && nb::hasattr(indices_obj, "is_contiguous")) {
            if (!nb::cast<bool>(indices_obj.attr("is_contiguous")())) {
                indices_obj = indices_obj.attr("contiguous")();
            }
        }
        uintptr_t ptr = nb::cast<uintptr_t>(indices_obj.attr("data_ptr")());
        res.count = static_cast<uint32_t>(nb::len(indices_obj));
        if (res.count == 0) return res;
        std::string dt_str = nb::hasattr(indices_obj, "dtype") ? nb::cast<std::string>(indices_obj.attr("dtype").attr("__str__")()) : "";
        bool is_int64 = (dt_str.find("int64") != std::string::npos || dt_str.find("long") != std::string::npos);

        if (is_cuda && !is_int64) {
            res.d_indices = reinterpret_cast<const int32_t*>(ptr);
            return res;
        }

        std::vector<int32_t> h_idx(res.count);
        if (!is_cuda) {
            if (is_int64) {
                const int64_t* src = reinterpret_cast<const int64_t*>(ptr);
                for (size_t i = 0; i < res.count; ++i) h_idx[i] = static_cast<int32_t>(src[i]);
            } else {
                const int32_t* src = reinterpret_cast<const int32_t*>(ptr);
                for (size_t i = 0; i < res.count; ++i) h_idx[i] = src[i];
            }
        } else {
            std::vector<int64_t> tmp64(res.count);
            cudaMemcpy(tmp64.data(), reinterpret_cast<const void*>(ptr), res.count * sizeof(int64_t), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < res.count; ++i) h_idx[i] = static_cast<int32_t>(tmp64[i]);
        }
        int32_t* d_mem = nullptr;
        cudaMalloc(&d_mem, res.count * sizeof(int32_t));
        cudaMemcpyAsync(d_mem, h_idx.data(), res.count * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        res.d_indices = d_mem;
        res.needs_free = true;
        return res;
    }

    if (nb::isinstance<nb::list>(indices_obj)) {
        nb::list l = nb::cast<nb::list>(indices_obj);
        res.count = static_cast<uint32_t>(l.size());
        if (res.count > 0) {
            std::vector<int32_t> h_idx(res.count);
            for (uint32_t i = 0; i < res.count; ++i) h_idx[i] = nb::cast<int32_t>(l[i]);
            int32_t* d_mem = nullptr;
            cudaMalloc(&d_mem, res.count * sizeof(int32_t));
            cudaMemcpyAsync(d_mem, h_idx.data(), res.count * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            res.d_indices = d_mem;
            res.needs_free = true;
        }
        return res;
    }

    return res;
}

inline void release_indices_buffer(IndicesBufferInfo& info) {
    if (info.needs_free && info.d_indices) {
        cudaFree(const_cast<int32_t*>(info.d_indices));
        info.d_indices = nullptr;
        info.needs_free = false;
    }
}

NB_MODULE(rocketsim_cuda, m) {
    m.doc() = "RocketSim-CUDA: Ultra-parallel GPU physics simulation engine with zero-copy PyTorch/DLPack tensors";

    // Global Physics & Timing Constants
    m.attr("TICK_RATE") = TICK_RATE;
    m.attr("DELTA_TIME") = DELTA_TIME;
    m.attr("DEFAULT_MAX_ENVS") = DEFAULT_MAX_ENVS;
    m.attr("MAX_CARS_PER_ENV") = MAX_CARS_PER_ENV;
    m.attr("MAX_BOOST_PADS") = MAX_BOOST_PADS;
    m.attr("CAR_MASS") = CAR_MASS;
    m.attr("BALL_MASS") = BALL_MASS;
    m.attr("BALL_RADIUS") = BALL_RADIUS;
    m.attr("BALL_REST_Z") = BALL_REST_Z;
    m.attr("BOOST_MAX") = BOOST_MAX;

    // GpuTensorView: DLPack standard representation exposing shape, strides, and __dlpack__
    nb::class_<GpuTensorView>(m, "GpuTensorView")
        .def_prop_ro("shape", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.shape) l.append(s);
            return nb::tuple(l);
        })
        .def_prop_ro("strides", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.strides) l.append(s);
            return nb::tuple(l);
        })
        .def_prop_ro("dtype", [](const GpuTensorView& v) { return v.dtype; })
        .def_prop_ro("device", [](const GpuTensorView& v) { return "cuda:0"; })
        .def_prop_ro("is_cuda", [](const GpuTensorView&) { return true; })
        .def_prop_ro("data_ptr", [](const GpuTensorView& v) { return reinterpret_cast<uintptr_t>(v.data); })
        .def("get_data_ptr", [](const GpuTensorView& v) { return reinterpret_cast<uintptr_t>(v.data); })
        .def("size", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.shape) l.append(s);
            return nb::tuple(l);
        })
        .def("stride", [](const GpuTensorView& v) {
            nb::list l;
            for (auto s : v.strides) l.append(s);
            return nb::tuple(l);
        })
        .def("dim", [](const GpuTensorView& v) { return v.shape.size(); })
        .def("__len__", [](const GpuTensorView& v) { return v.shape.empty() ? 0 : v.shape[0]; })
        .def("__getitem__", &GpuTensorView::getitem)
        .def("__setitem__", &GpuTensorView::setitem)
        .def("zero_", &GpuTensorView::zero_)
        .def("clone", &GpuTensorView::clone)
        .def("sum", [](const GpuTensorView& v) -> double {
            size_t total_elements = 1;
            for (auto s : v.shape) total_elements *= s;
            if (total_elements == 0) return 0.0;
            if (v.dtype == "uint8") {
                std::vector<uint8_t> h_buf(total_elements);
                cudaMemcpy(h_buf.data(), v.data, total_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                double sum = 0.0;
                for (auto x : h_buf) sum += x;
                return sum;
            } else if (v.dtype == "float32") {
                std::vector<float> h_buf(total_elements);
                cudaMemcpy(h_buf.data(), v.data, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
                double sum = 0.0;
                for (auto x : h_buf) sum += x;
                return sum;
            } else if (v.dtype == "int32") {
                std::vector<int32_t> h_buf(total_elements);
                cudaMemcpy(h_buf.data(), v.data, total_elements * sizeof(int32_t), cudaMemcpyDeviceToHost);
                double sum = 0.0;
                for (auto x : h_buf) sum += x;
                return sum;
            } else if (v.dtype == "uint32") {
                std::vector<uint32_t> h_buf(total_elements);
                cudaMemcpy(h_buf.data(), v.data, total_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                double sum = 0.0;
                for (auto x : h_buf) sum += x;
                return sum;
            }
            return 0.0;
        })
        .def("__dlpack__", &GpuTensorView::to_dlpack, nb::arg("stream") = nb::none())
        .def("__dlpack_device__", [](const GpuTensorView& v) {
            return std::make_pair(v.device_type, v.device_id);
        })
        .def("__repr__", [](const GpuTensorView& v) {
            std::string s = "GpuTensorView(shape=[";
            for (size_t i = 0; i < v.shape.size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(v.shape[i]);
            }
            s += "], strides=[";
            for (size_t i = 0; i < v.strides.size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(v.strides[i]);
            }
            s += "], dtype=" + v.dtype + ", device=cuda:0)";
            return s;
        });

    nb::class_<GameEventTrackerConfig>(m, "GameEventTrackerConfig")
        .def(nb::init<>())
        .def_rw("shot_min_speed", &GameEventTrackerConfig::shot_min_speed)
        .def_rw("shot_touch_min_delay", &GameEventTrackerConfig::shot_touch_min_delay)
        .def_rw("pred_score_extra_margin", &GameEventTrackerConfig::pred_score_extra_margin)
        .def_rw("shot_event_cooldown", &GameEventTrackerConfig::shot_event_cooldown)
        .def_rw("shot_min_score_time", &GameEventTrackerConfig::shot_min_score_time)
        .def_rw("goal_max_touch_time", &GameEventTrackerConfig::goal_max_touch_time)
        .def_rw("pass_max_touch_time", &GameEventTrackerConfig::pass_max_touch_time);

    nb::class_<SimContext>(m, "SimContext")
        .def(nb::init<uint32_t, uint32_t>(),
             nb::arg("num_envs") = 1,
             nb::arg("cars_per_env") = 1,
             "Initialize SimContext allocating monolithic GPU VRAM arena")
        .def_prop_ro("num_envs", &SimContext::GetNumEnvs)
        .def_prop_ro("cars_per_env", &SimContext::GetCarsPerEnv)
        .def_prop_ro("total_cars", &SimContext::GetTotalCars)
        .def_prop_ro("allocated_bytes", &SimContext::GetAllocatedBytes)
        .def_prop_ro("ball_pitch_floats", &SimContext::GetBallPitchFloats)
        .def_prop_ro("car_pitch_floats", &SimContext::GetCarPitchFloats)
        .def("get_num_envs", &SimContext::GetNumEnvs)
        .def("get_cars_per_env", &SimContext::GetCarsPerEnv)
        .def("get_total_cars", &SimContext::GetTotalCars)
        .def("get_allocated_bytes", &SimContext::GetAllocatedBytes)
        .def("get_ball_pitch_floats", &SimContext::GetBallPitchFloats)
        .def("get_car_pitch_floats", &SimContext::GetCarPitchFloats)

        // Ball State DLPack Zero-Copy View: shape [num_envs, 13], strides [1, pitch_floats]
        .def("get_ball_observations", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetBallState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), 13 };
            v.strides = { 1, static_cast<int64_t>(ctx.GetBallPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA strided tensor view of Ball State (13 rigid body floats) [num_envs, 13]")
        .def("get_ball_state_tensor", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetBallState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), 13 };
            v.strides = { 1, static_cast<int64_t>(ctx.GetBallPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Alias for get_ball_observations")
        .def("get_ball_observations_raw", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            size_t shape[2] = { ctx.GetNumEnvs(), 13 };
            int64_t strides[2] = { 1, static_cast<int64_t>(ctx.GetBallPitchFloats()) };
            return nb::ndarray<float, nb::device::cuda>(
                ctx.GetBallState().pos_x,
                2,
                shape,
                self,
                strides
            );
        }, "Get raw DLPack PyCapsule of Ball State")

        // Car State DLPack Zero-Copy View: shape [num_envs, cars_per_env, 14], strides [cars_per_env, 1, pitch_floats]
        .def("get_car_observations", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()), 14 };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1, static_cast<int64_t>(ctx.GetCarPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA strided tensor view of Car State (14 floats) [num_envs, cars_per_env, 14]")
        .def("get_car_state_tensor", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().pos_x;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()), 14 };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1, static_cast<int64_t>(ctx.GetCarPitchFloats()) };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Alias for get_car_observations")
        .def("get_car_observations_raw", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            size_t shape[3] = { ctx.GetNumEnvs(), ctx.GetCarsPerEnv(), 14 };
            int64_t strides[3] = {
                static_cast<int64_t>(ctx.GetCarsPerEnv()),
                1,
                static_cast<int64_t>(ctx.GetCarPitchFloats())
            };
            return nb::ndarray<float, nb::device::cuda>(
                ctx.GetCarState().pos_x,
                3,
                shape,
                self,
                strides
            );
        }, "Get raw DLPack PyCapsule of Car State")

        // Ball Hit State Views: shape [num_envs, cars_per_env]
        .def("get_ball_hit_is_valid", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().ball_hit_is_valid;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Ball Hit Is Valid flags [num_envs, cars_per_env]")
        .def("get_ball_touched", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().ball_touched;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Ball Touched flags [num_envs, cars_per_env]")

        // Supersonic & Demolition State Views: shape [num_envs, cars_per_env]
        .def("get_is_supersonic", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().is_supersonic;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Supersonic flags [num_envs, cars_per_env]")
        .def("get_is_demoed", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().is_demoed;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Demolished flags [num_envs, cars_per_env]")
        .def("get_demo_respawn_timer", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().demo_respawn_timer;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Demolition Respawn Timers [num_envs, cars_per_env]")

        // Flip & Jump Mechanics Views: shape [num_envs, cars_per_env]
        .def("get_has_flipped", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().has_flipped;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Has Flipped flags [num_envs, cars_per_env]")
        .def("get_has_flip", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().has_flipped;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Has Flip / Has Flipped flags [num_envs, cars_per_env]")

        // Boost & Ground Mechanics Views: shape [num_envs, cars_per_env]
        .def("get_boost", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().boost;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Car Boost amounts [num_envs, cars_per_env]")
        .def("get_is_on_ground", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().is_on_ground;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Is On Ground flags [num_envs, cars_per_env]")
        .def("get_on_ground", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetCarState().is_on_ground;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Alias for get_is_on_ground [num_envs, cars_per_env]")

        // Boost Pad States: shape [num_envs, 34], strides [34, 1]
        .def("get_pad_is_active", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().pad_is_active;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), MAX_BOOST_PADS };
            v.strides = { MAX_BOOST_PADS, 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Boost Pad Active flags [num_envs, 34]")
        .def("get_pad_cooldown", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().pad_cooldown;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), MAX_BOOST_PADS };
            v.strides = { MAX_BOOST_PADS, 1 };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Boost Pad Cooldown timers [num_envs, 34]")

        // Episode & Termination Flags (shape [num_envs])
        .def("get_is_goal", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().is_goal;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Goal flags [num_envs]")
        .def("get_scoring_team", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().scoring_team;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Scoring Team indices [num_envs]")
        .def("get_is_out_of_bounds", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().is_out_of_bounds;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Out of Bounds flags [num_envs]")
        .def("get_tick_count", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetArenaState().tick_count;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Episode Tick Counts [num_envs]")
        .def("get_rewards", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetRewards();
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "float32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Episode Rewards [num_envs, cars_per_env]")
        .def("get_terminated", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetTerminated();
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Terminated flags [num_envs]")
        .def("get_truncated", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetTruncated();
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Truncated flags [num_envs]")
        // Game Event Tracking Views
        .def("get_step_touches", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().step_touches;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Step Ball Touches [num_envs, cars_per_env]")
        .def("get_step_shots", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().step_shots;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Step Shots on Goal [num_envs, cars_per_env]")
        .def("get_step_saves", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().step_saves;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Step Saves [num_envs, cars_per_env]")
        .def("get_step_goals", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().step_goals;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Step Goals Scored [num_envs, cars_per_env]")
        .def("get_step_assists", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().step_assists;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Step Assists [num_envs, cars_per_env]")
        .def("get_total_touches", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().total_touches;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Lifetime Total Touches [num_envs, cars_per_env]")
        .def("get_total_shots", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().total_shots;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Lifetime Total Shots [num_envs, cars_per_env]")
        .def("get_total_saves", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().total_saves;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Lifetime Total Saves [num_envs, cars_per_env]")
        .def("get_total_goals", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().total_goals;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Lifetime Total Goals [num_envs, cars_per_env]")
        .def("get_total_assists", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventCarEvents().total_assists;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()), static_cast<int64_t>(ctx.GetCarsPerEnv()) };
            v.strides = { static_cast<int64_t>(ctx.GetCarsPerEnv()), 1 };
            v.dtype = "uint32";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Lifetime Total Assists [num_envs, cars_per_env]")
        .def("get_ball_shot", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventEnvState().ball_shot;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Ball Shot status [num_envs]")
        .def("get_ball_shot_goal_team", [](nb::handle self) {
            auto& ctx = nb::cast<SimContext&>(self);
            GpuTensorView v;
            v.data = ctx.GetGameEventEnvState().ball_shot_goal_team;
            v.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v.strides = { 1 };
            v.dtype = "uint8";
            v.device_type = 2;
            v.device_id = 0;
            v.owner = nb::borrow(self);
            return v;
        }, "Get DLPack CUDA tensor view of Defending Team of Ball Shot [num_envs]")
        .def("get_game_event_tracker_config", &SimContext::GetGameEventTrackerConfig)
        .def("set_game_event_tracker_config", &SimContext::SetGameEventTrackerConfig, nb::arg("config"))
        .def("is_game_event_tracking_enabled", &SimContext::IsGameEventTrackingEnabled)
        .def("set_game_event_tracking_enabled", &SimContext::SetGameEventTrackingEnabled, nb::arg("enabled"))
        .def("update_game_events", [](SimContext& ctx, float delta_time, uint32_t delta_ticks) {
            ctx.UpdateGameEvents(delta_time, delta_ticks);
        }, nb::arg("delta_time") = DELTA_TIME, nb::arg("delta_ticks") = 1, "Manually update game events")
        .def("reset_game_events", [](SimContext& ctx, int32_t env_idx) {
            ctx.ResetGameEvents(env_idx);
        }, nb::arg("env_idx") = -1, "Reset game event tracking state (-1 for all)")
        .def("reset_game_events_indexed", [](SimContext& ctx, nb::object indices) {
            auto idx_buf = get_indices_buffer(indices, ctx.GetStream());
            if (idx_buf.is_provided && idx_buf.count > 0 && idx_buf.d_indices) {
                ctx.ResetGameEventsIndexed(idx_buf.d_indices, idx_buf.count);
            }
            if (idx_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_indices_buffer(idx_buf);
        }, nb::arg("indices"), "Reset game event tracking state for specified env indices")
        .def("is_ball_probably_going_in", [](nb::handle self, float max_time, float extra_margin) {
            auto& ctx = nb::cast<SimContext&>(self);
            ctx.EvaluateIsBallProbablyGoingIn(max_time, extra_margin);

            GpuTensorView v_in;
            v_in.data = ctx.GetIsBallProbablyGoingIn();
            v_in.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v_in.strides = { 1 };
            v_in.dtype = "uint8";
            v_in.device_type = 2;
            v_in.device_id = 0;
            v_in.owner = nb::borrow(self);

            GpuTensorView v_team;
            v_team.data = ctx.GetBallProbablyGoingInGoalTeam();
            v_team.shape = { static_cast<int64_t>(ctx.GetNumEnvs()) };
            v_team.strides = { 1 };
            v_team.dtype = "int32";
            v_team.device_type = 2;
            v_team.device_id = 0;
            v_team.owner = nb::borrow(self);

            return nb::make_tuple(v_in, v_team);
        }, nb::arg("max_time") = 2.0f, nb::arg("extra_margin") = 0.0f,
        "Extrapolate ball trajectory analytically to predict if it will enter a net within max_time.")
        .def("get_stream", [](const SimContext& ctx) {
            return reinterpret_cast<uintptr_t>(ctx.GetStream());
        }, "Get raw CUDA stream pointer (uintptr_t)")

        // Simulation Step
        .def("step", [](SimContext& ctx, uint32_t batch_size) {
            ctx.Step(batch_size, nullptr);
        }, nb::arg("batch_size") = 0, "Execute simulation step using stored controls")
        .def("step", [](SimContext& ctx, nb::object actions) {
            const float* d_actions = nullptr;
            if (nb::isinstance<GpuTensorView>(actions)) {
                d_actions = reinterpret_cast<const float*>(nb::cast<const GpuTensorView&>(actions).data);
            } else if (nb::hasattr(actions, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(actions.attr("data_ptr")());
                d_actions = reinterpret_cast<const float*>(ptr);
            } else {
                auto arr = nb::cast<nb::ndarray<float, nb::device::cuda>>(actions);
                d_actions = arr.data();
            }
            ctx.Step(0, d_actions);
        }, nb::arg("actions"), "Execute simulation step consuming GPU actions tensor with zero host copies")
        .def("step_actions", [](SimContext& ctx, nb::object actions) {
            const float* d_actions = nullptr;
            if (nb::isinstance<GpuTensorView>(actions)) {
                d_actions = reinterpret_cast<const float*>(nb::cast<const GpuTensorView&>(actions).data);
            } else if (nb::hasattr(actions, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(actions.attr("data_ptr")());
                d_actions = reinterpret_cast<const float*>(ptr);
            } else {
                auto arr = nb::cast<nb::ndarray<float, nb::device::cuda>>(actions);
                d_actions = arr.data();
            }
            ctx.Step(0, d_actions);
        }, nb::arg("actions"), "Execute simulation step consuming GPU actions tensor with zero host copies")

        // Selective Resets
        .def("reset_to_default", [](SimContext& ctx) {
            ctx.ResetToDefault();
        }, "Reset all environments to default kickoff states")
        .def("reset_batch", [](SimContext& ctx, nb::object indices) {
            if (indices.is_none()) {
                ctx.ResetToDefault();
                return;
            }
            if (nb::isinstance<GpuTensorView>(indices)) {
                const auto& v = nb::cast<const GpuTensorView&>(indices);
                if (v.dtype == "int64") {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int64_t*>(v.data), static_cast<uint32_t>(v.shape[0]));
                } else {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int32_t*>(v.data), static_cast<uint32_t>(v.shape[0]));
                }
                return;
            }
            if (nb::hasattr(indices, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(indices.attr("data_ptr")());
                size_t count = nb::len(indices);
                std::string dt_str = nb::hasattr(indices, "dtype") ? nb::cast<std::string>(indices.attr("dtype").attr("__str__")()) : "";
                if (dt_str.find("int64") != std::string::npos || dt_str.find("long") != std::string::npos) {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int64_t*>(ptr), static_cast<uint32_t>(count));
                } else {
                    ctx.ResetEnvironmentsIndexed(reinterpret_cast<const int32_t*>(ptr), static_cast<uint32_t>(count));
                }
                return;
            }
            if (nb::isinstance<nb::ndarray<int32_t, nb::device::cuda>>(indices)) {
                auto arr = nb::cast<nb::ndarray<int32_t, nb::device::cuda>>(indices);
                ctx.ResetEnvironmentsIndexed(arr.data(), static_cast<uint32_t>(arr.size()));
                return;
            }
            if (nb::isinstance<nb::ndarray<int64_t, nb::device::cuda>>(indices)) {
                auto arr = nb::cast<nb::ndarray<int64_t, nb::device::cuda>>(indices);
                ctx.ResetEnvironmentsIndexed(arr.data(), static_cast<uint32_t>(arr.size()));
                return;
            }
            if (nb::isinstance<nb::list>(indices)) {
                nb::list l = nb::cast<nb::list>(indices);
                uint32_t count = static_cast<uint32_t>(l.size());
                if (count == 0) return;
                std::vector<int32_t> host_idx(count);
                for (uint32_t i = 0; i < count; ++i) {
                    host_idx[i] = nb::cast<int32_t>(l[i]);
                }
                int32_t* d_idx = nullptr;
                cudaMalloc(&d_idx, count * sizeof(int32_t));
                cudaMemcpyAsync(d_idx, host_idx.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.GetStream());
                ctx.ResetEnvironmentsIndexed(d_idx, count);
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
                cudaFree(d_idx);
                return;
            }
        }, nb::arg("indices") = nb::none(), "Selectively reset environments on GPU asynchronously")
        .def("reset_masked", [](SimContext& ctx, nb::object mask) {
            const uint8_t* d_mask = nullptr;
            if (nb::isinstance<GpuTensorView>(mask)) {
                d_mask = reinterpret_cast<const uint8_t*>(nb::cast<const GpuTensorView&>(mask).data);
            } else if (nb::hasattr(mask, "data_ptr")) {
                uintptr_t ptr = nb::cast<uintptr_t>(mask.attr("data_ptr")());
                d_mask = reinterpret_cast<const uint8_t*>(ptr);
            } else {
                auto arr = nb::cast<nb::ndarray<uint8_t, nb::device::cuda>>(mask);
                d_mask = arr.data();
            }
            ctx.ResetEnvironmentsMasked(d_mask);
        }, nb::arg("mask"), "Reset environments using a GPU boolean/uint8 mask")

        // State Setters (Batch zero-copy GPU tensor resets)
        .def("set_ball_state", [](SimContext& ctx, nb::object state, uint32_t env_start, nb::object indices) {
            auto buf = get_tensor_buffer(state, ctx.GetStream());
            if (buf.d_ptr == nullptr) {
                throw std::invalid_argument("set_ball_state: invalid tensor object");
            }
            auto idx_buf = get_indices_buffer(indices, ctx.GetStream());

            uint32_t count = 0;
            uint32_t num_cols = 13;
            int64_t stride_env = 13;
            int64_t stride_col = 1;

            if (buf.shape.size() == 1) {
                count = 1;
                num_cols = static_cast<uint32_t>(buf.shape[0]);
                stride_env = 0;
                stride_col = buf.strides.empty() ? 1 : buf.strides[0];
            } else if (buf.shape.size() >= 2) {
                count = static_cast<uint32_t>(buf.shape[0]);
                num_cols = static_cast<uint32_t>(buf.shape[1]);
                stride_env = buf.strides.empty() ? num_cols : buf.strides[0];
                stride_col = buf.strides.size() > 1 ? buf.strides[1] : 1;
            }

            if (idx_buf.is_provided) {
                count = idx_buf.count;
            }

            if (count > 0) {
                ctx.SetBallStateFromTensor(
                    static_cast<const float*>(buf.d_ptr),
                    count,
                    env_start,
                    idx_buf.d_indices,
                    num_cols,
                    stride_env,
                    stride_col
                );
            }

            if (buf.needs_free || idx_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(buf);
            release_indices_buffer(idx_buf);
        }, nb::arg("state"), nb::arg("env_start") = 0, nb::arg("indices") = nb::none(),
        "Set ball states in batch on GPU directly in VRAM with zero host copies")
        .def("set_ball_states", [](SimContext& ctx, nb::object state, uint32_t env_start, nb::object indices) {
            auto buf = get_tensor_buffer(state, ctx.GetStream());
            if (buf.d_ptr == nullptr) throw std::invalid_argument("set_ball_states: invalid tensor object");
            auto idx_buf = get_indices_buffer(indices, ctx.GetStream());
            uint32_t count = 0, num_cols = 13;
            int64_t stride_env = 13, stride_col = 1;
            if (buf.shape.size() == 1) {
                count = 1; num_cols = static_cast<uint32_t>(buf.shape[0]); stride_env = 0; stride_col = buf.strides.empty() ? 1 : buf.strides[0];
            } else if (buf.shape.size() >= 2) {
                count = static_cast<uint32_t>(buf.shape[0]); num_cols = static_cast<uint32_t>(buf.shape[1]);
                stride_env = buf.strides.empty() ? num_cols : buf.strides[0]; stride_col = buf.strides.size() > 1 ? buf.strides[1] : 1;
            }
            if (idx_buf.is_provided) count = idx_buf.count;
            if (count > 0) {
                ctx.SetBallStateFromTensor(static_cast<const float*>(buf.d_ptr), count, env_start, idx_buf.d_indices, num_cols, stride_env, stride_col);
            }
            if (buf.needs_free || idx_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(buf); release_indices_buffer(idx_buf);
        }, nb::arg("state"), nb::arg("env_start") = 0, nb::arg("indices") = nb::none(),
        "Alias for set_ball_state")

        .def("set_car_state", [](SimContext& ctx, nb::object state, uint32_t car_start, nb::object indices) {
            auto buf = get_tensor_buffer(state, ctx.GetStream());
            if (buf.d_ptr == nullptr) {
                throw std::invalid_argument("set_car_state: invalid tensor object");
            }
            auto idx_buf = get_indices_buffer(indices, ctx.GetStream());

            uint32_t count = 0;
            uint32_t num_cols = 14;
            int64_t stride_env = 0;
            int64_t stride_car = 14;
            int64_t stride_col = 1;

            if (buf.shape.size() == 1) {
                count = 1;
                num_cols = static_cast<uint32_t>(buf.shape[0]);
                stride_car = 0;
                stride_col = buf.strides.empty() ? 1 : buf.strides[0];
            } else if (buf.shape.size() == 2) {
                count = static_cast<uint32_t>(buf.shape[0]);
                num_cols = static_cast<uint32_t>(buf.shape[1]);
                stride_car = buf.strides.empty() ? num_cols : buf.strides[0];
                stride_col = buf.strides.size() > 1 ? buf.strides[1] : 1;
            } else if (buf.shape.size() == 3) {
                int64_t e = buf.shape[0];
                int64_t c = buf.shape[1];
                count = static_cast<uint32_t>(e * c);
                num_cols = static_cast<uint32_t>(buf.shape[2]);
                stride_env = buf.strides.empty() ? (c * num_cols) : buf.strides[0];
                stride_car = buf.strides.size() > 1 ? buf.strides[1] : num_cols;
                stride_col = buf.strides.size() > 2 ? buf.strides[2] : 1;
            }

            if (idx_buf.is_provided) {
                count = idx_buf.count;
            }

            if (count > 0) {
                ctx.SetCarStateFromTensor(
                    static_cast<const float*>(buf.d_ptr),
                    count,
                    car_start,
                    idx_buf.d_indices,
                    num_cols,
                    stride_env,
                    stride_car,
                    stride_col
                );
            }

            if (buf.needs_free || idx_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(buf);
            release_indices_buffer(idx_buf);
        }, nb::arg("state"), nb::arg("car_start") = 0, nb::arg("indices") = nb::none(),
        "Set car states in batch on GPU directly in VRAM with zero host copies")
        .def("set_car_states", [](SimContext& ctx, nb::object state, uint32_t car_start, nb::object indices) {
            auto buf = get_tensor_buffer(state, ctx.GetStream());
            if (buf.d_ptr == nullptr) throw std::invalid_argument("set_car_states: invalid tensor object");
            auto idx_buf = get_indices_buffer(indices, ctx.GetStream());
            uint32_t count = 0, num_cols = 14;
            int64_t stride_env = 0, stride_car = 14, stride_col = 1;
            if (buf.shape.size() == 1) {
                count = 1; num_cols = static_cast<uint32_t>(buf.shape[0]); stride_car = 0; stride_col = buf.strides.empty() ? 1 : buf.strides[0];
            } else if (buf.shape.size() == 2) {
                count = static_cast<uint32_t>(buf.shape[0]); num_cols = static_cast<uint32_t>(buf.shape[1]);
                stride_car = buf.strides.empty() ? num_cols : buf.strides[0]; stride_col = buf.strides.size() > 1 ? buf.strides[1] : 1;
            } else if (buf.shape.size() == 3) {
                count = static_cast<uint32_t>(buf.shape[0] * buf.shape[1]); num_cols = static_cast<uint32_t>(buf.shape[2]);
                stride_env = buf.strides.empty() ? (buf.shape[1] * num_cols) : buf.strides[0];
                stride_car = buf.strides.size() > 1 ? buf.strides[1] : num_cols;
                stride_col = buf.strides.size() > 2 ? buf.strides[2] : 1;
            }
            if (idx_buf.is_provided) count = idx_buf.count;
            if (count > 0) {
                ctx.SetCarStateFromTensor(static_cast<const float*>(buf.d_ptr), count, car_start, idx_buf.d_indices, num_cols, stride_env, stride_car, stride_col);
            }
            if (buf.needs_free || idx_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(buf); release_indices_buffer(idx_buf);
        }, nb::arg("state"), nb::arg("car_start") = 0, nb::arg("indices") = nb::none(),
        "Alias for set_car_state")

        .def("set_pad_state", [](SimContext& ctx, nb::object pad_is_active, nb::object pad_cooldown, uint32_t env_start, nb::object indices) {
            TensorBufferInfo active_buf;
            if (!pad_is_active.is_none()) active_buf = get_tensor_buffer(pad_is_active, ctx.GetStream());

            TensorBufferInfo cd_buf;
            if (!pad_cooldown.is_none()) cd_buf = get_tensor_buffer(pad_cooldown, ctx.GetStream());

            if (active_buf.d_ptr == nullptr && cd_buf.d_ptr == nullptr) {
                throw std::invalid_argument("set_pad_state: both pad_is_active and pad_cooldown cannot be None");
            }

            auto idx_buf = get_indices_buffer(indices, ctx.GetStream());

            uint32_t count = 0;
            if (active_buf.d_ptr) {
                count = active_buf.shape.size() > 1 ? static_cast<uint32_t>(active_buf.shape[0]) : 1;
            } else if (cd_buf.d_ptr) {
                count = cd_buf.shape.size() > 1 ? static_cast<uint32_t>(cd_buf.shape[0]) : 1;
            }
            if (idx_buf.is_provided) {
                count = idx_buf.count;
            }

            int64_t stride_env_act = active_buf.strides.empty() ? MAX_BOOST_PADS : active_buf.strides[0];
            int64_t stride_pad_act = active_buf.strides.size() > 1 ? active_buf.strides[1] : 1;

            int64_t stride_env_cd = cd_buf.strides.empty() ? MAX_BOOST_PADS : cd_buf.strides[0];
            int64_t stride_pad_cd = cd_buf.strides.size() > 1 ? cd_buf.strides[1] : 1;

            if (count > 0) {
                ctx.SetPadStateFromTensor(
                    static_cast<const uint8_t*>(active_buf.d_ptr),
                    stride_env_act,
                    stride_pad_act,
                    static_cast<const float*>(cd_buf.d_ptr),
                    stride_env_cd,
                    stride_pad_cd,
                    count,
                    env_start,
                    idx_buf.d_indices
                );
            }

            if (active_buf.needs_free || cd_buf.needs_free || idx_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(active_buf);
            release_tensor_buffer(cd_buf);
            release_indices_buffer(idx_buf);
        }, nb::arg("pad_is_active") = nb::none(), nb::arg("pad_cooldown") = nb::none(), nb::arg("env_start") = 0, nb::arg("indices") = nb::none(),
        "Set boost pad states in batch on GPU directly in VRAM with zero host copies")
        .def("set_pad_states", [](SimContext& ctx, nb::object pad_is_active, nb::object pad_cooldown, uint32_t env_start, nb::object indices) {
            TensorBufferInfo active_buf;
            if (!pad_is_active.is_none()) active_buf = get_tensor_buffer(pad_is_active, ctx.GetStream());
            TensorBufferInfo cd_buf;
            if (!pad_cooldown.is_none()) cd_buf = get_tensor_buffer(pad_cooldown, ctx.GetStream());
            if (active_buf.d_ptr == nullptr && cd_buf.d_ptr == nullptr) throw std::invalid_argument("set_pad_states: both None");
            auto idx_buf = get_indices_buffer(indices, ctx.GetStream());
            uint32_t count = active_buf.d_ptr ? (active_buf.shape.size() > 1 ? static_cast<uint32_t>(active_buf.shape[0]) : 1) :
                                                (cd_buf.shape.size() > 1 ? static_cast<uint32_t>(cd_buf.shape[0]) : 1);
            if (idx_buf.is_provided) count = idx_buf.count;
            int64_t stride_env_act = active_buf.strides.empty() ? MAX_BOOST_PADS : active_buf.strides[0];
            int64_t stride_pad_act = active_buf.strides.size() > 1 ? active_buf.strides[1] : 1;
            int64_t stride_env_cd = cd_buf.strides.empty() ? MAX_BOOST_PADS : cd_buf.strides[0];
            int64_t stride_pad_cd = cd_buf.strides.size() > 1 ? cd_buf.strides[1] : 1;
            if (count > 0) {
                ctx.SetPadStateFromTensor(static_cast<const uint8_t*>(active_buf.d_ptr), stride_env_act, stride_pad_act,
                                          static_cast<const float*>(cd_buf.d_ptr), stride_env_cd, stride_pad_cd, count, env_start, idx_buf.d_indices);
            }
            if (active_buf.needs_free || cd_buf.needs_free || idx_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(active_buf); release_tensor_buffer(cd_buf); release_indices_buffer(idx_buf);
        }, nb::arg("pad_is_active") = nb::none(), nb::arg("pad_cooldown") = nb::none(), nb::arg("env_start") = 0, nb::arg("indices") = nb::none(),
        "Alias for set_pad_state")

        .def("predict_ball_trajectory", [](SimContext& ctx, uint32_t num_ticks, nb::object initial_state, nb::object out_tensor) -> nb::object {
            if (num_ticks == 0) num_ticks = 120;

            const float* d_init = nullptr;
            uint32_t init_cols = 0;
            int64_t stride_init_env = 0;
            int64_t stride_init_col = 1;
            TensorBufferInfo init_buf;
            if (!initial_state.is_none()) {
                init_buf = get_tensor_buffer(initial_state, ctx.GetStream());
                if (init_buf.d_ptr) {
                    d_init = static_cast<const float*>(init_buf.d_ptr);
                    init_cols = init_buf.shape.empty() ? 6 : static_cast<uint32_t>(init_buf.shape.back());
                    stride_init_env = init_buf.strides.empty() ? init_cols : init_buf.strides[0];
                    stride_init_col = init_buf.strides.size() > 1 ? init_buf.strides[1] : 1;
                }
            }

            float* d_out = nullptr;
            uint32_t num_cols = 6;
            int64_t stride_out_env = 0;
            int64_t stride_out_tick = 0;
            int64_t stride_out_col = 1;
            nb::object ret_obj;

            if (!out_tensor.is_none()) {
                auto out_buf = get_tensor_buffer(out_tensor, ctx.GetStream());
                if (!out_buf.is_cuda || out_buf.d_ptr == nullptr) {
                    throw std::invalid_argument("predict_ball_trajectory: out_tensor must reside on GPU (is_cuda=True)");
                }
                d_out = static_cast<float*>(out_buf.d_ptr);
                num_cols = out_buf.shape.empty() ? 6 : static_cast<uint32_t>(out_buf.shape.back());
                stride_out_env = out_buf.strides.empty() ? static_cast<int64_t>(num_ticks * num_cols) : out_buf.strides[0];
                stride_out_tick = out_buf.strides.size() > 1 ? out_buf.strides[1] : static_cast<int64_t>(num_cols);
                stride_out_col = out_buf.strides.size() > 2 ? out_buf.strides[2] : 1;
                ret_obj = out_tensor;
            } else {
                int64_t n_envs = ctx.GetNumEnvs();
                int64_t n_ticks = num_ticks;
                int64_t cols = 6;
                size_t total_bytes = n_envs * n_ticks * cols * sizeof(float);

                void* d_new = nullptr;
                cudaMalloc(&d_new, total_bytes);
                d_out = static_cast<float*>(d_new);

                stride_out_env = n_ticks * cols;
                stride_out_tick = cols;
                stride_out_col = 1;

                GpuTensorView v;
                v.data = d_new;
                v.shape = { n_envs, n_ticks, cols };
                v.strides = { n_ticks * cols, cols, 1 };
                v.dtype = "float32";
                v.device_type = 2;
                v.device_id = 0;
                nb::capsule cleanup(d_new, [](void* p) noexcept {
                    if (p) cudaFree(p);
                });
                v.owner = cleanup;
                ret_obj = nb::cast(v);
                num_cols = 6;
            }

            ctx.PredictBallTrajectory(
                d_out,
                num_ticks,
                num_cols,
                d_init,
                init_cols,
                stride_init_env,
                stride_init_col,
                stride_out_env,
                stride_out_tick,
                stride_out_col
            );

            if (init_buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
                release_tensor_buffer(init_buf);
            }

            return ret_obj;
        }, nb::arg("num_ticks") = 120, nb::arg("initial_state") = nb::none(), nb::arg("out_tensor") = nb::none(),
        "On-demand dedicated CUDA kernel for ball trajectory prediction over N future ticks")

        // C++ Interop / Staging Setters
        .def("copy_ball_state_to_device", [](SimContext& ctx, nb::object state, uint32_t env_start) {
            auto buf = get_tensor_buffer(state, ctx.GetStream());
            if (buf.d_ptr == nullptr) throw std::invalid_argument("copy_ball_state_to_device: invalid tensor");
            uint32_t count = buf.shape.size() > 1 ? static_cast<uint32_t>(buf.shape[0]) : 1;
            uint32_t num_cols = buf.shape.size() > 1 ? static_cast<uint32_t>(buf.shape[1]) : static_cast<uint32_t>(buf.shape[0]);
            int64_t st_env = buf.strides.empty() ? num_cols : buf.strides[0];
            int64_t st_col = buf.strides.size() > 1 ? buf.strides[1] : 1;
            ctx.SetBallStateFromTensor(static_cast<const float*>(buf.d_ptr), count, env_start, nullptr, num_cols, st_env, st_col);
            if (buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(buf);
        }, nb::arg("state"), nb::arg("env_start") = 0, "Copy ball states to device")

        .def("copy_car_state_to_device", [](SimContext& ctx, nb::object state, uint32_t car_start) {
            auto buf = get_tensor_buffer(state, ctx.GetStream());
            if (buf.d_ptr == nullptr) throw std::invalid_argument("copy_car_state_to_device: invalid tensor");
            uint32_t count = 0, num_cols = 14;
            int64_t st_env = 0, st_car = 14, st_col = 1;
            if (buf.shape.size() == 1) {
                count = 1; num_cols = static_cast<uint32_t>(buf.shape[0]); st_car = 0; st_col = buf.strides.empty() ? 1 : buf.strides[0];
            } else if (buf.shape.size() == 2) {
                count = static_cast<uint32_t>(buf.shape[0]); num_cols = static_cast<uint32_t>(buf.shape[1]);
                st_car = buf.strides.empty() ? num_cols : buf.strides[0]; st_col = buf.strides.size() > 1 ? buf.strides[1] : 1;
            } else if (buf.shape.size() == 3) {
                count = static_cast<uint32_t>(buf.shape[0] * buf.shape[1]); num_cols = static_cast<uint32_t>(buf.shape[2]);
                st_env = buf.strides.empty() ? (buf.shape[1] * num_cols) : buf.strides[0];
                st_car = buf.strides.size() > 1 ? buf.strides[1] : num_cols;
                st_col = buf.strides.size() > 2 ? buf.strides[2] : 1;
            }
            ctx.SetCarStateFromTensor(static_cast<const float*>(buf.d_ptr), count, car_start, nullptr, num_cols, st_env, st_car, st_col);
            if (buf.needs_free) {
                if (ctx.GetStream()) cudaStreamSynchronize(ctx.GetStream()); else cudaDeviceSynchronize();
            }
            release_tensor_buffer(buf);
        }, nb::arg("state"), nb::arg("car_start") = 0, "Copy car states to device");

    // Asynchronous CUDA Events (for non-blocking latency/throughput benchmarking)
    nb::class_<GpuEvent>(m, "GpuEvent")
        .def(nb::init<bool>(), nb::arg("enable_timing") = true)
        .def("record", &GpuEvent::record, nb::arg("stream") = nb::none())
        .def("wait", &GpuEvent::wait, nb::arg("stream") = nb::none())
        .def("synchronize", &GpuEvent::synchronize)
        .def("query", &GpuEvent::query)
        .def("elapsed_time", &GpuEvent::elapsed_time, nb::arg("end_event"));

    m.attr("Event") = m.attr("GpuEvent");

    // Allocate zeroed GPU tensor in VRAM
    m.def("zeros", [](std::vector<int64_t> shape, const std::string& dtype) {
        int64_t total = 1;
        for (auto s : shape) total *= s;
        size_t elem_size = (dtype == "uint8") ? 1 : ((dtype == "int64") ? 8 : 4);
        size_t total_bytes = total * elem_size;

        void* d_ptr = nullptr;
        cudaMalloc(&d_ptr, total_bytes);
        cudaMemset(d_ptr, 0, total_bytes);

        std::vector<int64_t> strides(shape.size());
        int64_t st = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = st;
            st *= shape[i];
        }

        GpuTensorView v;
        v.data = d_ptr;
        v.shape = shape;
        v.strides = strides;
        v.dtype = dtype;
        v.device_type = 2;
        v.device_id = 0;
        nb::capsule cleanup(d_ptr, [](void* p) noexcept {
            if (p) cudaFree(p);
        });
        v.owner = cleanup;
        return v;
    }, nb::arg("shape"), nb::arg("dtype") = "float32", "Allocate a zeroed GPU tensor in VRAM");

    m.def("is_ball_probably_going_in", [](
        nb::sequence pos, nb::sequence vel, float max_time, float extra_margin
    ) -> nb::tuple {
        if (nb::len(pos) != 3 || nb::len(vel) != 3) {
            throw std::invalid_argument("is_ball_probably_going_in: pos and vel must each have 3 elements");
        }
        Vec3 ball_pos(
            nb::cast<float>(pos[0]),
            nb::cast<float>(pos[1]),
            nb::cast<float>(pos[2])
        );
        Vec3 ball_vel(
            nb::cast<float>(vel[0]),
            nb::cast<float>(vel[1]),
            nb::cast<float>(vel[2])
        );
        int32_t goal_team = -1;
        bool going_in = is_ball_probably_going_in(ball_pos, ball_vel, max_time, extra_margin, &goal_team);
        return nb::make_tuple(going_in, goal_team);
    }, nb::arg("pos"), nb::arg("vel"), nb::arg("max_time") = 2.0f, nb::arg("extra_margin") = 0.0f,
    "Extrapolate ball trajectory analytically to predict if it will enter a net within max_time.");

    m.def("get_vram_info", []() {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return std::make_pair(free_b, total_b);
    }, "Return tuple of (free_vram_bytes, total_vram_bytes)");
}
