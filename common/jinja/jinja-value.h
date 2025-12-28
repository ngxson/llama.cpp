#pragma once

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <sstream>

#include "jinja-string.h"

namespace jinja {

struct value_t;
using value = std::shared_ptr<value_t>;


// Helper to check the type of a value
template<typename T>
struct extract_pointee {
    using type = T;
};
template<typename U>
struct extract_pointee<std::shared_ptr<U>> {
    using type = U;
};
template<typename T>
bool is_val(const value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr.get()) != nullptr;
}
template<typename T>
bool is_val(const value_t * ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr) != nullptr;
}
template<typename T, typename... Args>
std::shared_ptr<typename extract_pointee<T>::type> mk_val(Args&&... args) {
    using PointeeType = typename extract_pointee<T>::type;
    return std::make_shared<PointeeType>(std::forward<Args>(args)...);
}
template<typename T>
const typename extract_pointee<T>::type * cast_val(const value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr.get());
}
template<typename T>
typename extract_pointee<T>::type * cast_val(value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<PointeeType*>(ptr.get());
}
template<typename T>
void ensure_val(const value & ptr) {
    if (!is_val<T>(ptr)) {
        throw std::runtime_error("Expected value of type " + std::string(typeid(T).name()));
    }
}
// End Helper


struct context; // forward declaration


// for converting from JSON to jinja values
// example input JSON:
// {
//   "messages": [
//     {"role": "user", "content": "Hello!"},
//     {"role": "assistant", "content": "Hi there!"}
//   ],
//   "bos_token": "<s>",
//   "eos_token": "</s>",
// }
//
// to mark strings as user input, wrap them in a special object:
// {
//   "messages": [
//     {
//       "role": "user",
//       "content": {"__input__": "Hello!"}  // this string is user input
//     },
//     ...
//   ],
// }
//
// marking input can be useful for tracking data provenance
// and preventing template injection attacks
//
// Note: T_JSON can be nlohmann::json or similar types
template<typename T_JSON>
void global_from_json(context & ctx, const T_JSON & json_obj);



struct func_args {
    std::vector<value> args;
    context & ctx;
    func_args(context & ctx) : ctx(ctx) {}
    void ensure_count(size_t min, size_t max = 999) const {
        if (args.size() < min || args.size() > max) {
            throw std::runtime_error("Expected between " + std::to_string(min) + " and " + std::to_string(max) + " arguments, got " + std::to_string(args.size()));
        }
    }
    value get_kwarg(const std::string & key) const;
    // utility functions
    // TODO: allow optional arguments
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

bool value_compare(const value & a, const value & b);

struct value_t {
    int64_t val_int;
    double val_flt;
    string val_str;
    bool val_bool;

    // array and object are stored as shared_ptr to allow reference access
    // example:
    //     my_obj = {"a": 1, "b": 2}
    //     my_arr = [my_obj]
    //     my_obj["a"] = 3
    //     print(my_arr[0]["a"])  # should print 3
    std::vector<value> val_arr;
    std::map<std::string, value> val_obj;

    func_handler val_func;

    value_t() = default;
    value_t(const value_t &) = default;
    virtual ~value_t() = default;

    virtual std::string type() const { return ""; }

    virtual int64_t as_int() const { throw std::runtime_error(type() + " is not an int value"); }
    virtual double as_float() const { throw std::runtime_error(type() + " is not a float value"); }
    virtual string as_string() const { throw std::runtime_error(type() + " is not a string value"); }
    virtual bool as_bool() const { throw std::runtime_error(type() + " is not a bool value"); }
    virtual const std::vector<value> & as_array() const { throw std::runtime_error(type() + " is not an array value"); }
    virtual const std::map<std::string, value> & as_object() const { throw std::runtime_error(type() + " is not an object value"); }
    virtual value invoke(const func_args &) const { throw std::runtime_error(type() + " is not a function value"); }
    virtual bool is_null() const { return false; }
    virtual bool is_undefined() const { return false; }
    virtual const func_builtins & get_builtins() const {
        throw std::runtime_error("No builtins available for type " + type());
    }

    virtual std::string as_repr() const { return as_string().str(); }
};


struct value_int_t : public value_t {
    value_int_t(int64_t v) { val_int = v; }
    virtual std::string type() const override { return "Integer"; }
    virtual int64_t as_int() const override { return val_int; }
    virtual double as_float() const override { return static_cast<double>(val_int); }
    virtual string as_string() const override { return std::to_string(val_int); }
    virtual const func_builtins & get_builtins() const override;
};
using value_int = std::shared_ptr<value_int_t>;


struct value_float_t : public value_t {
    value_float_t(double v) { val_flt = v; }
    virtual std::string type() const override { return "Float"; }
    virtual double as_float() const override { return val_flt; }
    virtual int64_t as_int() const override { return static_cast<int64_t>(val_flt); }
    virtual string as_string() const override { return std::to_string(val_flt); }
    virtual const func_builtins & get_builtins() const override;
};
using value_float = std::shared_ptr<value_float_t>;


struct value_string_t : public value_t {
    value_string_t() { val_str = string(); }
    value_string_t(const std::string & v) { val_str = string(v); }
    value_string_t(const string & v) { val_str = v; }
    virtual std::string type() const override { return "String"; }
    virtual string as_string() const override { return val_str; }
    virtual std::string as_repr() const override {
        std::ostringstream ss;
        for (const auto & part : val_str.parts) {
            ss << (part.is_input ? "INPUT: " : "TMPL:  ") << part.val << "\n";
        }
        return ss.str();
    }
    virtual bool as_bool() const override {
        return val_str.length() > 0;
    }
    virtual const func_builtins & get_builtins() const override;
    void mark_input() {
        val_str.mark_input();
    }
};
using value_string = std::shared_ptr<value_string_t>;


struct value_bool_t : public value_t {
    value_bool_t(bool v) { val_bool = v; }
    virtual std::string type() const override { return "Boolean"; }
    virtual bool as_bool() const override { return val_bool; }
    virtual string as_string() const override { return std::string(val_bool ? "True" : "False"); }
    virtual const func_builtins & get_builtins() const override;
};
using value_bool = std::shared_ptr<value_bool_t>;


struct value_array_t : public value_t {
    value_array_t() = default;
    value_array_t(value & v) {
        // point to the same underlying data
        val_arr = v->val_arr;
    }
    void push_back(const value & val) {
        val_arr.push_back(val);
    }
    virtual std::string type() const override { return "Array"; }
    virtual const std::vector<value> & as_array() const override { return val_arr; }
    virtual string as_string() const override {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < val_arr.size(); i++) {
            if (i > 0) ss << ", ";
            ss << val_arr.at(i)->as_repr();
        }
        ss << "]";
        return ss.str();
    }
    virtual bool as_bool() const override {
        return !val_arr.empty();
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_array = std::shared_ptr<value_array_t>;


struct value_object_t : public value_t {
    value_object_t() = default;
    value_object_t(value & v) {
        // point to the same underlying data
        val_obj = v->val_obj;
    }
    value_object_t(const std::map<std::string, value> & obj) {
        val_obj = std::map<std::string, value>();
        for (const auto & pair : obj) {
            val_obj[pair.first] = pair.second;
        }
    }
    void insert(const std::string & key, const value & val) {
        val_obj[key] = val;
    }
    virtual std::string type() const override { return "Object"; }
    virtual const std::map<std::string, value> & as_object() const override { return val_obj; }
    virtual bool as_bool() const override {
        return !val_obj.empty();
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_object = std::shared_ptr<value_object_t>;


struct value_func_t : public value_t {
    std::string name; // for debugging
    value arg0; // bound "this" argument, if any
    value_func_t(const func_handler & func, std::string func_name = "") {
        val_func = func;
        name = func_name;
    }
    value_func_t(const func_handler & func, const value & arg_this, std::string func_name = "") {
        val_func = func;
        name = func_name;
        arg0 = arg_this;
    }
    virtual value invoke(const func_args & args) const override {
        if (arg0) {
            func_args new_args(args.ctx);
            new_args.args.push_back(arg0);
            for (const auto & a : args.args) {
                new_args.args.push_back(a);
            }
            return val_func(new_args);
        } else {
            return val_func(args);
        }
    }
    virtual std::string type() const override { return "Function"; }
    virtual std::string as_repr() const override { return type(); }
};
using value_func = std::shared_ptr<value_func_t>;


struct value_null_t : public value_t {
    virtual std::string type() const override { return "Null"; }
    virtual bool is_null() const override { return true; }
    virtual bool as_bool() const override { return false; }
    virtual std::string as_repr() const override { return type(); }
    virtual const func_builtins & get_builtins() const override;
};
using value_null = std::shared_ptr<value_null_t>;


struct value_undefined_t : public value_t {
    std::string hint; // for debugging, to indicate where undefined came from
    value_undefined_t(const std::string & h = "") : hint(h) {}
    virtual std::string type() const override { return hint.empty() ? "Undefined" : "Undefined (hint: '" + hint + "')"; }
    virtual bool is_undefined() const override { return true; }
    virtual bool as_bool() const override { return false; }
    virtual std::string as_repr() const override { return type(); }
};
using value_undefined = std::shared_ptr<value_undefined_t>;

// special value for kwarg
struct value_kwarg_t : public value_t {
    std::string key;
    value val;
    value_kwarg_t(const std::string & k, const value & v) : key(k), val(v) {}
    virtual std::string type() const override { return "KwArg"; }
    virtual std::string as_repr() const override { return type(); }
};
using value_kwarg = std::shared_ptr<value_kwarg_t>;


const func_builtins & global_builtins();

} // namespace jinja
