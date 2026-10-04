#include "core/cache_archive.h"
#include <chrono>
#include <iostream>
#include <thread>

using namespace ninfer::cache_archive;
namespace {
std::string current(const std::filesystem::path& root) {
    Reader input(root/"CURRENT"); std::string value; input(value); return value;
}
void pointer(const std::filesystem::path& root, const std::string& value) {
    Writer output(root/"CURRENT.tmp"); output(value); output.close();
    publish(root/"CURRENT.tmp",root/"CURRENT");
}
}
int main() {
    auto root=std::filesystem::temp_directory_path()/
        ("ninfer-cache-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directory(root);
        pointer(root,"previous");
        Publications pending;
        Payloads payloads;
        std::vector<std::uint32_t> expected(262144);
        for (std::size_t index=0; index<expected.size(); ++index) expected[index]=std::uint32_t(index*7919);
        {
            DeferPublication capture(pending,nullptr,&payloads);
            const PayloadSpan span{reinterpret_cast<const std::byte*>(expected.data()),expected.size()*sizeof(std::uint32_t)};
            if (!stage_payload(root/"payload.tmp",std::span(&span,1))) throw std::runtime_error("payload staging failed");
            publish(root/"payload.tmp",root/"payload.bin");
            pointer(root,"next");
        }
        // The durable root must still identify the preceding snapshot while writes are deferred.
        if (current(root)!="previous") throw std::runtime_error("published an incomplete snapshot");
        // The background writer owns RAM data and has not touched the payload file yet.
        if (std::filesystem::exists(root/"payload.tmp")) throw std::runtime_error("disk write happened during capture");
        auto original=expected; expected.clear(); expected.shrink_to_fit();
        std::exception_ptr failure;
        std::thread worker([&] {
            try { write_payloads(payloads); for (auto& [temporary,final] : pending) publish(temporary,final); }
            catch (...) { failure=std::current_exception(); }
        });
        worker.join(); if (failure) std::rethrow_exception(failure);
        if (current(root)!="next") throw std::runtime_error("snapshot was not published");
        Reader data(root/"payload.bin"); std::uint64_t count; data(count);
        if (count!=original.size()*sizeof(std::uint32_t)) throw std::runtime_error("payload size mismatch");
        std::vector<std::uint32_t> actual(original.size()); data.bytes(actual.data(),count);
        if (actual!=original) throw std::runtime_error("persisted payload changed");
        // A failed payload flush must not advance the durable root.
        pending.clear();
        {
            DeferPublication capture(pending);
            publish(root/"missing.tmp",root/"missing.bin"); pointer(root,"broken");
        }
        bool rejected=false;
        try { for (auto& [temporary,final] : pending) publish(temporary,final); }
        catch (const std::runtime_error&) { rejected=true; }
        if (!rejected || current(root)!="next") throw std::runtime_error("failed save replaced previous snapshot");
        std::filesystem::resize_file(root/"payload.bin",15);
        rejected=false;
        try { Reader truncated(root/"payload.bin"); truncated(count); truncated.bytes(actual.data(),count); }
        catch (const std::runtime_error&) { rejected=true; }
        if (!rejected) throw std::runtime_error("accepted truncated payload");
        std::filesystem::remove_all(root);
        std::cout<<"PASS: asynchronous payload ownership, atomic snapshot publication, failed save preservation, truncated payload rejection\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root); std::cerr<<error.what()<<'\n'; return 1;
    }
}
