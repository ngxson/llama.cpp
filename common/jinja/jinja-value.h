#pragma once

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <sstream>


namespace jinja {

struct value_t;
using value = std::unique_ptr<value_t>;


// Helper to check the type of a value
template<typename T>
struct extract_pointee {
    using type = T;
};
template<typename U>
struct extract_pointee<std::unique_ptr<U>> {
    using type = U;
};
template<typename T>
bool is_val(const value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr.get()) != nullptr;
}
template<typename T, typename... Args>
value mk_val(Args&&... args) {
    using PointeeType = typename extract_pointee<T>::type;
    return std::make_unique<PointeeType>(std::forward<Args>(args)...);
}
template<typename T>
void ensure_val(const value & ptr) {
    if (!is_val<T>(ptr)) {
        throw std::runtime_error("Expected value of type " + std::string(typeid(T).name()));
    }
}
// End Helper


struct func_args {
    std::vector<value> args;
    void ensure_count(size_t count) const {
        if (args.size() != count) {
            throw std::runtime_error("Expected " + std::to_string(count) + " arguments, got " + std::to_string(args.size()));
        }
    }
    // utility functions
    template<typename T> void ensure_vals() const {
        ensure_count(1);
        ensure_val<T>(args[0]);
    }
    template<typename T, typename U> void ensure_vals() const {
        ensure_count(2);
        ensure_val<T>(args[0]);
        ensure_val<U>(args[1]);
    }
    template<typename T, typename U, typename V> void ensure_vals() const {
        ensure_count(3);
        ensure_val<T>(args[0]);
        ensure_val<U>(args[1]);
        ensure_val<V>(args[2]);
    }
};

using func_handler = std::function<value(const func_args &)>;
using func_builtins = std::map<std::string, func_handler>;

struct value_t {
    int64_t val_int;
    double val_flt;
    std::string val_str;
    bool val_bool;

    // array and object are stored as shared_ptr to allow reference access
    // example:
    //     my_obj = {"a": 1, "b": 2}
    //     my_arr = [my_obj]
    //     my_obj["a"] = 3
    //     print(my_arr[0]["a"])  # should print 3
    std::shared_ptr<std::vector<value>> val_arr;
    std::shared_ptr<std::map<std::string, value>> val_obj;

    func_handler val_func;

    value_t() = default;
    value_t(const value_t &) = default;
    virtual ~value_t() = default;

    virtual std::string type() const { return ""; }

    virtual int64_t as_int() const { throw std::runtime_error("Not an int value"); }
    virtual double as_float() const { throw std::runtime_error("Not a float value"); }
    virtual std::string as_string() const { throw std::runtime_error("Not a string value"); }
    virtual bool as_bool() const { throw std::runtime_error("Not a bool value"); }
    virtual const std::vector<value> & as_array() const { throw std::runtime_error("Not an array value"); }
    virtual const std::map<std::string, value> & as_object() const { throw std::runtime_error("Not an object value"); }
    virtual value invoke(const func_args &) const { throw std::runtime_error("Not a function value"); }
    virtual bool is_null() const { return false; }
    virtual bool is_undefined() const { return false; }
    virtual const func_builtins & get_builtins() const {
        throw std::runtime_error("No builtins available for type " + type());
    }

    virtual value clone() const {
        return std::make_unique<value_t>(*this);
    }

    virtual bool operator==(const value & other) const {
        // TODO
        return false;
    }
    virtual bool operator!=(const value & other) const {
        return !(*this == other);
    }
};


struct value_int_t : public value_t {
    value_int_t(int64_t v) { val_int = v; }
    virtual std::string type() const override { return "Integer"; }
    virtual int64_t as_int() const override { return val_int; }
    virtual double as_float() const override { return static_cast<double>(val_int); }
    virtual std::string as_string() const override { return std::to_string(val_int); }
    virtual value clone() const override { return std::make_unique<value_int_t>(*this); }
    virtual const func_builtins & get_builtins() const override;
};
using value_int = std::unique_ptr<value_int_t>;


struct value_float_t : public value_t {
    value_float_t(double v) { val_flt = v; }
    virtual std::string type() const override { return "Float"; }
    virtual double as_float() const override { return val_flt; }
    virtual int64_t as_int() const override { return static_cast<int64_t>(val_flt); }
    virtual std::string as_string() const override { return std::to_string(val_flt); }
    virtual value clone() const override { return std::make_unique<value_float_t>(*this); }
    virtual const func_builtins & get_builtins() const override;
};
using value_float = std::unique_ptr<value_float_t>;


struct value_string_t : public value_t {
    bool is_user_input = false; // may skip parsing special tokens if true

    value_string_t(const std::string & v) { val_str = v; }
    virtual std::string type() const override { return "String"; }
    virtual std::string as_string() const override { return val_str; }
    virtual value clone() const override { return std::make_unique<value_string_t>(*this); }
    virtual const func_builtins & get_builtins() const override;
};
using value_string = std::unique_ptr<value_string_t>;


struct value_bool_t : public value_t {
    value_bool_t(bool v) { val_bool = v; }
    virtual std::string type() const override { return "Boolean"; }
    virtual bool as_bool() const override { return val_bool; }
    virtual std::string as_string() const override { return val_bool ? "True" : "False"; }
    virtual value clone() const override { return std::make_unique<value_bool_t>(*this); }
    virtual const func_builtins & get_builtins() const override;
};
using value_bool = std::unique_ptr<value_bool_t>;


struct value_array_t : public value_t {
    value_array_t() {
        val_arr = std::make_shared<std::vector<value>>();
    }
    value_array_t(value & v) {
        // point to the same underlying data
        val_arr = v->val_arr;
    }
    value_array_t(value_array_t & other, size_t start = 0, size_t end = -1) {
        val_arr = std::make_shared<std::vector<value>>();
        size_t sz = other.val_arr->size();
        if (end == static_cast<size_t>(-1) || end > sz) {
            end = sz;
        }
        if (start > end || start >= sz) {
            return;
        }
        for (size_t i = start; i < end; i++) {
            val_arr->push_back(other.val_arr->at(i)->clone());
        }
    }
    virtual std::string type() const override { return "Array"; }
    virtual const std::vector<value> & as_array() const override { return *val_arr; }
    virtual value clone() const override {
        auto tmp = std::make_unique<value_array_t>();
        tmp->val_arr = this->val_arr;
        return tmp;
    }
    virtual std::string as_string() const override {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < val_arr->size(); i++) {
            if (i > 0) ss << ", ";
            ss << val_arr->at(i)->as_string();
        }
        ss << "]";
        return ss.str();
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_array = std::unique_ptr<value_array_t>;


struct value_object_t : public value_t {
    value_object_t() {
        val_obj = std::make_shared<std::map<std::string, value>>();
    }
    value_object_t(value & v) {
        // point to the same underlying data
        val_obj = v->val_obj;
    }
    value_object_t(const std::map<std::string, value> & obj) {
        val_obj = std::make_shared<std::map<std::string, value>>();
        for (const auto & pair : obj) {
            (*val_obj)[pair.first] = pair.second->clone();
        }
    }
    virtual std::string type() const override { return "Object"; }
    virtual const std::map<std::string, value> & as_object() const override { return *val_obj; }
    virtual value clone() const override {
        auto tmp = std::make_unique<value_object_t>();
        tmp->val_obj = this->val_obj;
        return tmp;
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_object = std::unique_ptr<value_object_t>;


struct value_func_t : public value_t {
    value_func_t(func_handler & func) {
        val_func = func;
    }
    virtual value invoke(const func_args & args) const override {
        return val_func(args);
    }
    virtual std::string type() const override { return "Function"; }
    virtual value clone() const override { return std::make_unique<value_func_t>(*this); }
};
using value_func = std::unique_ptr<value_func_t>;


struct value_null_t : public value_t {
    virtual std::string type() const override { return "Null"; }
    virtual bool is_null() const override { return true; }
    virtual value clone() const override { return std::make_unique<value_null_t>(*this); }
};
using value_null = std::unique_ptr<value_null_t>;


struct value_undefined_t : public value_t {
    virtual std::string type() const override { return "Undefined"; }
    virtual bool is_undefined() const override { return true; }
    virtual value clone() const override { return std::make_unique<value_undefined_t>(*this); }
};
using value_undefined = std::unique_ptr<value_undefined_t>;

} // namespace jinja
