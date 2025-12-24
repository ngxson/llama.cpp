#include <common.h>
#include <sstream>

struct vm_context {
    std::ostringstream out;
};

struct op_base {
    virtual ~op_base() = default;
    virtual void execute(vm_context & ctx) = 0;
};

struct op_print : public op_base {
    std::string message;
    op_print(const std::string & message) : message(message) {}
    void execute(vm_context & ctx) override {
        ctx.out << message;
    }
};

struct op_load : public op_base {
    std::string dst;
    std::string src;
    std::string value;
    op_load(const std::string & dst) : dst(dst) {}
    void execute(vm_context & ctx) override {
    }
};
