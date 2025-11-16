#pragma once

#include "utils.hpp"
#include "common.h"

#include <functional>
#include <string>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    // TODO: move this to a virtual function once we have proper polymorphism support
    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;

struct server_http_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // reserved for future use
    std::string path; // reserved for future use
    std::string body;
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }
};

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    std::string path_prefix;
    std::string hostname;
    int port;

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop();

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;
    void get(const std::string &, handler_t);
    void post(const std::string &, handler_t);

    // for debugging
    std::string listening_address;
};

// simple HTTP client with blocking API
struct server_http_client : server_http_res {
    std::function<void()> cleanup = nullptr;
public:
    server_http_client(const std::string & method,
                       const std::string & host,
                       int port,
                       const std::string & path,
                       const std::map<std::string, std::string> & headers,
                       const std::string & body,
                       const std::function<bool()> should_stop);
    ~server_http_client() {
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
    };
    // simple implementation of a pipe
    template<typename T>
    struct pipe_t {
        std::mutex mutex;
        std::condition_variable cv;
        std::queue<T> queue;
        std::atomic<bool> writer_closed{false};
        std::atomic<bool> reader_closed{false};
        void close_write() {
            writer_closed.store(true);
            cv.notify_all();
        }
        void close_read() {
            reader_closed.store(true);
            cv.notify_all();
        }
        bool read(T & output, const std::function<bool()> & should_stop) {
            std::unique_lock<std::mutex> lk(mutex);
            constexpr auto poll_interval = std::chrono::milliseconds(500);
            while (true) {
                if (!queue.empty()) {
                    output = std::move(queue.front());
                    queue.pop();
                    return true;
                }
                if (writer_closed.load()) {
                    return false; // clean EOF
                }
                if (should_stop()) {
                    close_read(); // signal broken pipe to writer
                    return false; // cancelled / reader no longer alive
                }
                cv.wait_for(lk, poll_interval);
            }
        }
        bool write(T && data) {
            std::lock_guard<std::mutex> lk(mutex);
            if (reader_closed.load()) {
                return false; // broken pipe
            }
            queue.push(std::move(data));
            cv.notify_one();
            return true;
        }
    };
};
