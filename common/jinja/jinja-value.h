#pragma once

#include <vector>
#include <string>
#include <map>


namespace jinja {

struct value_t;
using value = std::unique_ptr<value_t>;

struct value_t {
    int64_t val_int;
    double val_flt;
    std::string val_str;
    bool val_bool;
    std::vector<value> val_arr;
    std::map<std::string, value> val_obj;

    virtual std::string type() const { return ""; }

    virtual ~value_t() = default;
    virtual int64_t as_int() const { throw std::runtime_error("Not an int value"); }
    virtual double as_float() const { throw std::runtime_error("Not a float value"); }
    virtual std::string as_string() const { throw std::runtime_error("Not a string value"); }
    virtual bool as_bool() const { throw std::runtime_error("Not a bool value"); }
    virtual const std::vector<value> & as_array() const { throw std::runtime_error("Not an array value"); }
    virtual const std::map<std::string, value> & as_object() const { throw std::runtime_error("Not an object value"); }
    virtual bool is_null() const { return false; }
    virtual bool is_undefined() const { return false; }

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
};
using value_int = std::unique_ptr<value_int_t>;

struct value_float_t : public value_t {
    value_float_t(double v) { val_flt = v; }
    virtual std::string type() const override { return "Float"; }
    virtual double as_float() const override { return val_flt; }
    virtual int64_t as_int() const override { return static_cast<int64_t>(val_flt); }
};
using value_float = std::unique_ptr<value_float_t>;

struct value_string_t : public value_t {
    value_string_t(const std::string & v) { val_str = v; }
    virtual std::string type() const override { return "String"; }
    virtual std::string as_string() const override { return val_str; }
};
using value_string = std::unique_ptr<value_string_t>;

struct value_bool_t : public value_t {
    value_bool_t(bool v) { val_bool = v; }
    virtual std::string type() const override { return "Boolean"; }
    virtual bool as_bool() const override { return val_bool; }
};
using value_bool = std::unique_ptr<value_bool_t>;

struct value_array_t : public value_t {
    value_array_t(const std::vector<value> && v) { val_arr = std::move(v); }
    virtual std::string type() const override { return "Array"; }
    virtual const std::vector<value> & as_array() const override { return val_arr; }
};
using value_array = std::unique_ptr<value_array_t>;

struct value_object_t : public value_t {
    value_object_t(const std::map<std::string, value> & v) { val_obj = v; }
    virtual std::string type() const override { return "Object"; }
    virtual const std::map<std::string, value> & as_object() const override { return val_obj; }
};
using value_object = std::unique_ptr<value_object_t>;

struct value_null_t : public value_t {
    virtual std::string type() const override { return "Null"; }
    virtual bool is_null() const override { return true; }
};
using value_null = std::unique_ptr<value_null_t>;

struct value_undefined_t : public value_t {
    virtual std::string type() const override { return "Undefined"; }
    virtual bool is_undefined() const override { return true; }
};
using value_undefined = std::unique_ptr<value_undefined_t>;

} // namespace jinja
