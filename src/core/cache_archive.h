#pragma once
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <utility>
#include <memory>
#include <span>
#include <cstring>
#include <future>
#include <deque>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
namespace ninfer::cache_archive {
// Versioned local session format. Only owning values and payload bytes are serialized.
class Writer {
public:
    explicit Writer(const std::filesystem::path& path) : stream(path, std::ios::binary) {
        if (!stream) throw std::runtime_error("cannot create cache checkpoint");
    }
    void bytes(const void* data, std::size_t count) {
        stream.write(static_cast<const char*>(data), count);
        if (!stream) throw std::runtime_error("cache checkpoint write failed");
    }
    template<class T> void operator()(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(value));
    }
    template<class T> void operator()(const std::vector<T>& value) {
        (*this)(std::uint64_t(value.size()));
        for (const auto& item : value) (*this)(item);
    }
    void operator()(const std::string& value) {
        (*this)(std::uint64_t(value.size())); bytes(value.data(), value.size());
    }
    void close() { stream.flush(); if (!stream) throw std::runtime_error("cache flush failed"); stream.close(); if (!stream) throw std::runtime_error("cache close failed"); }
private:
    std::ofstream stream;
};
class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : stream(path, std::ios::binary) {
        if (!stream) throw std::runtime_error("cannot open cache checkpoint");
    }
    void bytes(void* data, std::size_t count) {
        stream.read(static_cast<char*>(data), count);
        if (!stream) throw std::runtime_error("truncated cache checkpoint");
    }
    template<class T> void operator()(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(value));
    }
    template<class T> void operator()(std::vector<T>& value) {
        std::uint64_t size; (*this)(size);
        if (size > 16777216) throw std::runtime_error("invalid cache vector length");
        value.resize(size); for (auto& item : value) (*this)(item);
    }
    void operator()(std::string& value) {
        std::uint64_t size; (*this)(size);
        if (size > 16777216) throw std::runtime_error("invalid cache string length");
        value.resize(size); bytes(value.data(), size);
    }
private:
    std::ifstream stream;
};
inline void sync_file(const std::filesystem::path& path) {
#ifndef _WIN32
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot sync cache checkpoint");
    int result = ::fsync(fd); ::close(fd);
    if (result) throw std::runtime_error("cache checkpoint fsync failed");
#else
    (void)path;
#endif
}
using Publications = std::vector<std::pair<std::filesystem::path,std::filesystem::path>>;
struct CaptureStats {
    std::uint64_t device_to_host_bytes = 0;
    double device_to_host_seconds = 0;
    std::uint64_t payload_bytes_written = 0;
};
struct PayloadSpan { const std::byte* data; std::size_t size; };
struct PendingPayload {
    std::filesystem::path temporary;
    std::unique_ptr<std::byte[]> bytes;
    std::size_t size;
};
struct Payloads {
    std::vector<PendingPayload> files;
    std::deque<std::future<PendingPayload>> jobs;
    void finish() {
        while (!jobs.empty()) {
            auto job=std::move(jobs.front()); jobs.pop_front();
            files.push_back(job.get());
        }
    }
    void wait() noexcept { for (auto& job : jobs) if (job.valid()) job.wait(); }
};
inline thread_local Payloads* deferred_payloads = nullptr;
inline bool stage_payload(const std::filesystem::path& temporary, std::span<const PayloadSpan> spans) {
    if (!deferred_payloads) return false;
    if (deferred_payloads->jobs.size() >= 8) {
        auto job=std::move(deferred_payloads->jobs.front()); deferred_payloads->jobs.pop_front();
        deferred_payloads->files.push_back(job.get());
    }
    std::vector<PayloadSpan> sources(spans.begin(),spans.end());
    deferred_payloads->jobs.push_back(std::async(std::launch::async,[temporary,sources=std::move(sources)] {
        std::uint64_t count = 0;
        for (auto span : sources) count += span.size;
        PendingPayload pending{temporary, std::make_unique_for_overwrite<std::byte[]>(sizeof(count)+count), sizeof(count)+count};
        std::memcpy(pending.bytes.get(), &count, sizeof(count));
        std::size_t offset = sizeof(count);
        for (auto span : sources) {
            std::memcpy(pending.bytes.get()+offset, span.data, span.size); offset += span.size;
        }
        return pending;
    }));
    return true;
}
inline void write_payloads(Payloads& payloads) {
    payloads.finish();
    for (auto& pending : payloads.files) {
        Writer writer(pending.temporary); writer.bytes(pending.bytes.get(),pending.size); writer.close();
        pending.bytes.reset();
    }
}
inline thread_local CaptureStats* capture_stats = nullptr;
inline thread_local Publications* deferred_publications = nullptr;
class DeferPublication {
public:
    explicit DeferPublication(Publications& files, CaptureStats* stats = nullptr, Payloads* payloads = nullptr)
        : previous_(deferred_publications), previous_stats_(capture_stats), previous_payloads_(deferred_payloads) {
        deferred_publications=&files; capture_stats=stats; deferred_payloads=payloads;
    }
    ~DeferPublication() { if (deferred_payloads) deferred_payloads->wait(); deferred_publications=previous_; capture_stats=previous_stats_; deferred_payloads=previous_payloads_; }
private:
    Publications* previous_;
    CaptureStats* previous_stats_;
    Payloads* previous_payloads_;
};
// Keep source views alive until all parallel RAM copies have finished, even on failure.
class PayloadCaptureBarrier {
public:
    ~PayloadCaptureBarrier() { if (deferred_payloads) deferred_payloads->wait(); }
};
inline void finish_payload_capture() { if (deferred_payloads) deferred_payloads->finish(); }
inline bool payload_exists(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) return true;
    if (!deferred_publications) return false;
    for (auto& file : *deferred_publications) if (file.second==path) return true;
    return false;
}
inline void publish(const std::filesystem::path& temporary, const std::filesystem::path& final) {
    if (deferred_publications) { deferred_publications->emplace_back(temporary,final); return; }
    sync_file(temporary); std::filesystem::rename(temporary, final); sync_file(final.parent_path());
}
}
