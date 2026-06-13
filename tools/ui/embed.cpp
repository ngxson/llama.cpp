// llama-ui-embed: generate ui.cpp / ui.h that embed UI assets as C arrays.
//
// Usage:
//   llama-ui-embed <out_cpp> <out_h> [<asset_name> <asset_path>]...

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <string>
#include <vector>
#include <cinttypes>
#include <cstdint>

// Returns true if s looks like a build hash (alphanumeric, 6-16 chars)
static bool is_hash(const std::string & s) {
    if (s.size() < 6 || s.size() > 16) return false;
    for (char c : s) {
        if (!isalnum((unsigned char)c)) return false;
    }
    return true;
}

// Returns true if s looks like a hex build hash (hex digits, 6-16 chars)
static bool is_hex_hash(const std::string & s) {
    if (s.size() < 6 || s.size() > 16) return false;
    for (char c : s) {
        if (!isxdigit((unsigned char)c)) return false;
    }
    return true;
}

// Strips the hash segment from a bare filename (no directory part):
//   bundle.HCjcCZFH.css  -> bundle.css   (name.HASH.ext)
//   workbox-12bd46aa.js  -> workbox.js   (name-HEXHASH.ext)
static std::string normalize_filename(const std::string & file) {
    size_t dot1 = file.find('.');
    if (dot1 != std::string::npos) {
        size_t dot2 = file.find('.', dot1 + 1);
        if (dot2 != std::string::npos && is_hash(file.substr(dot1 + 1, dot2 - dot1 - 1))) {
            return file.substr(0, dot1) + file.substr(dot2);
        }
    }
    size_t dot = file.rfind('.');
    if (dot != std::string::npos) {
        size_t dash = file.rfind('-', dot);
        if (dash != std::string::npos && is_hex_hash(file.substr(dash + 1, dot - dash - 1))) {
            return file.substr(0, dash) + file.substr(dot);
        }
    }
    return file;
}

// Normalizes a raw asset name from the dist directory:
//   _app/immutable/bundle.DaCibIq9.js         -> immutable/bundle.js
//   _app/immutable/assets/bundle.HCjcCZFH.css -> immutable/assets/bundle.css
//   workbox-12bd46aa.js                       -> workbox.js
static std::string normalize_asset_name(const std::string & name) {
    std::string s = name;
    if (s.size() > 5 && s.compare(0, 5, "_app/") == 0) {
        s = s.substr(5);
    }
    size_t slash = s.rfind('/');
    if (slash != std::string::npos) {
        return s.substr(0, slash + 1) + normalize_filename(s.substr(slash + 1));
    }
    return normalize_filename(s);
}

static const char * mime_from_ext(const std::string & name) {
    auto ext = name.rfind('.');
    if (ext == std::string::npos) return "application/octet-stream";
    std::string e = name.substr(ext + 1);
    if (e == "html")        return "text/html; charset=utf-8";
    if (e == "css")         return "text/css";
    if (e == "js")          return "application/javascript";
    if (e == "json")        return "application/json";
    if (e == "webmanifest") return "application/manifest+json";
    if (e == "svg")         return "image/svg+xml";
    if (e == "png")         return "image/png";
    if (e == "jpg" ||
        e == "jpeg")        return "image/jpeg";
    if (e == "ico")         return "image/x-icon";
    if (e == "woff")        return "font/woff";
    if (e == "woff2")       return "font/woff2";
    return "application/octet-stream";
}

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool read_file(const std::string & path, std::vector<unsigned char> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "embed: cannot open %s\n", path.c_str());
        return false;
    }
    const auto sz = f.tellg();
    if (sz < 0) {
        return false;
    }
    f.seekg(0);
    out.resize(static_cast<size_t>(sz));
    if (sz > 0 && !f.read(reinterpret_cast<char *>(out.data()), sz)) {
        return false;
    }
    return true;
}

static void append_bytes_hex(std::string & out, const std::vector<unsigned char> & bytes) {
    static const char hex[] = "0123456789abcdef";
    out.reserve(out.size() + bytes.size() * 5);
    for (unsigned char b : bytes) {
        out += '0';
        out += 'x';
        out += hex[b >> 4];
        out += hex[b & 0xf];
        out += ',';
    }
}

static bool write_if_different(const std::string & path, const std::string & content) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (f) {
        const auto sz = f.tellg();
        if (sz >= 0 && static_cast<size_t>(sz) == content.size()) {
            std::string existing(static_cast<size_t>(sz), '\0');
            f.seekg(0);
            if (sz == 0 || f.read(existing.data(), sz)) {
                if (existing == content) {
                    return true;
                }
            }
        }
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "embed: cannot write %s\n", path.c_str());
        return false;
    }
    if (!content.empty()) {
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    return out.good();
}

static std::string fmt(const char * pattern, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, pattern);
    const int n = vsnprintf(tmp, sizeof(tmp), pattern, ap);
    va_end(ap);
    return (n > 0) ? std::string(tmp, static_cast<size_t>(n)) : std::string();
}

int main(int argc, char ** argv) {
    if (argc < 3 || ((argc - 3) % 2) != 0) {
        fprintf(stderr, "usage: %s <out_cpp> <out_h> [<name> <path>]...\n", argv[0]);
        return 1;
    }

    const std::string out_cpp = argv[1];
    const std::string out_h   = argv[2];
    const int n_assets = (argc - 3) / 2;

    if (n_assets > 0) {
        bool has_index = false, has_bundle_js = false, has_bundle_css = false;
        for (int i = 0; i < n_assets; i++) {
            const std::string norm = normalize_asset_name(argv[3 + i * 2]);
            if (norm == "index.html")   has_index     = true;
            if (norm.find("bundle.js")  != std::string::npos) has_bundle_js  = true;
            if (norm.find("bundle.css") != std::string::npos) has_bundle_css = true;
        }
        if (!has_index || !has_bundle_js || !has_bundle_css) {
            fprintf(stderr, "embed: missing required assets (need index.html, bundle.js, bundle.css); got:\n");
            for (int i = 0; i < n_assets; i++) {
                fprintf(stderr, "  %s\n", normalize_asset_name(argv[3 + i * 2]).c_str());
            }
            return 1;
        }
    }

    std::string h;
    h += "#pragma once\n\n#include <array>\n#include <string>\n\n";
    if (n_assets > 0) {
        h += "#define LLAMA_UI_HAS_ASSETS 1\n\n";
    }
    h +=
        "struct llama_ui_asset {\n"
        "    std::string           name;\n"
        "    const unsigned char * data;\n"
        "    std::size_t           size;\n"
        "    std::string           etag;\n"
        "    std::string           type;\n"
        "};\n\n"
        "const llama_ui_asset * llama_ui_find_asset(const std::string & name);\n";
    h += fmt("const std::array<llama_ui_asset, %d> & llama_ui_get_assets();\n", n_assets);

    std::string cpp;
    cpp += "#include \"ui.h\"\n\n";

    if (n_assets > 0) {
        for (int i = 0; i < n_assets; i++) {
            const char * path = argv[3 + i * 2 + 1];
            std::vector<unsigned char> bytes;
            if (!read_file(path, bytes)) {
                return 1;
            }
            cpp += fmt("static const unsigned char asset_%d_data[] = {", i);
            append_bytes_hex(cpp, bytes);
            const auto hash = fnv_hash(bytes.data(), bytes.size());

            cpp += fmt("};\nstatic const std::size_t   asset_%d_size = %zu;\n",
                       i, bytes.size());
            cpp += fmt("static const char          asset_%d_etag[] = \"\\\"0x%016" PRIx64 "\\\"\";\n\n",
                       i, hash);
        }

        cpp += fmt("static const std::array<llama_ui_asset, %d> g_assets = {{\n", n_assets);
        for (int i = 0; i < n_assets; i++) {
            const std::string norm = normalize_asset_name(argv[3 + i * 2]);
            cpp += fmt("    { \"%s\", asset_%d_data, asset_%d_size, asset_%d_etag, \"%s\" },\n",
                       norm.c_str(), i, i, i, mime_from_ext(norm));
        }
        cpp += "}};\n\n";

        cpp +=
            "const llama_ui_asset * llama_ui_find_asset(const std::string & name) {\n"
            "    for (const auto & a : g_assets) {\n"
            "        if (a.name == name) {\n"
            "            return &a;\n"
            "        }\n"
            "    }\n"
            "    return nullptr;\n"
            "}\n";
        cpp += fmt("const std::array<llama_ui_asset, %d> & llama_ui_get_assets() {\n", n_assets);
        cpp +=     "    return g_assets;\n"
                   "}\n";
    } else {
        cpp +=
            "const llama_ui_asset * llama_ui_find_asset(const std::string &) {\n"
            "    return nullptr;\n"
            "}\n"
            "const std::array<llama_ui_asset, 0> & llama_ui_get_assets() {\n"
            "    static const std::array<llama_ui_asset, 0> empty{};\n"
            "    return empty;\n"
            "}\n";
    }

    bool ok = true;
    ok = write_if_different(out_h,   h)   && ok;
    ok = write_if_different(out_cpp, cpp) && ok;
    return ok ? 0 : 1;
}
