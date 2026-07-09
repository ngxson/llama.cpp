#include "server-tools.h"

#include <sheredom/subprocess.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <climits>
#include <algorithm>
#include <unordered_set>

namespace fs = std::filesystem;

//
// internal helpers
//

static std::vector<char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<char *> r;
    r.reserve(v.size() + 1);
    for (const auto & s : v) {
        r.push_back(const_cast<char *>(s.c_str()));
    }
    r.push_back(nullptr);
    return r;
}

struct run_proc_result {
    std::string output;
    int  exit_code = -1;
    bool timed_out = false;
};

static run_proc_result run_process(
        const std::vector<std::string> & args,
        size_t max_output,
        int timeout_secs) {
    run_proc_result res;

    subprocess_s proc;
    auto argv = to_cstr_vec(args);

    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_inherit_environment
                | subprocess_option_search_user_path;

    if (subprocess_create(argv.data(), options, &proc) != 0) {
        res.output = "failed to spawn process";
        return res;
    }

    std::atomic<bool> done{false};
    std::atomic<bool> timed_out{false};

    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out.store(true);
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    FILE * f = subprocess_stdout(&proc);
    std::string output;
    bool truncated = false;
    if (f) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            if (!truncated) {
                size_t len = strlen(buf);
                if (output.size() + len <= max_output) {
                    output.append(buf, len);
                } else {
                    output.append(buf, max_output - output.size());
                    truncated = true;
                }
            }
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) {
        timeout_thread.join();
    }

    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    res.output    = output;
    res.timed_out = timed_out.load();
    if (truncated) {
        res.output += "\n[output truncated]";
    }
    return res;
}

json server_tool::to_json() {
    return {
        {"display_name", display_name},
        {"tool", name},
        {"type", "builtin"},
        {"permissions", json{
            {"write", permission_write}
        }},
        {"definition", get_definition()},
    };
}

static constexpr size_t SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT = 8 * 1024 * 1024; // 8 MB
static constexpr int    SERVER_TOOL_GIT_LS_FILES_TIMEOUT    = 15;              // seconds

static const std::unordered_set<std::string> & server_tool_junk_dir_names() {
    static const std::unordered_set<std::string> names = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", "dist", "build", "target", ".cache", ".idea", ".vscode",
    };
    return names;
}

static std::vector<std::string> server_tool_list_files_fallback(const std::string & base) {
    std::vector<std::string> result;
    std::error_code ec;

    std::vector<std::pair<fs::path, fs::path>> stack;
    stack.emplace_back(fs::path(base), fs::path());

    while (!stack.empty()) {
        auto [dir, rel_dir] = stack.back();
        stack.pop_back();

        for (const auto & entry : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
            if (ec) break;
            std::string fname = entry.path().filename().string();
            std::error_code tec;
            if (entry.is_directory(tec)) {
                if (server_tool_junk_dir_names().count(fname) > 0) continue;
                stack.emplace_back(entry.path(), rel_dir / fname);
            } else if (entry.is_regular_file(tec)) {
                std::string rel = (rel_dir / fname).string();
                std::replace(rel.begin(), rel.end(), '\\', '/');
                result.push_back(rel);
            }
        }
    }

    return result;
}

// returns file paths relative to `base` (using '/' separators)
// sets `err` and returns empty if `base` doesn't exist / isn't a directory
static std::vector<std::string> server_tool_list_files(const std::string & base, std::string & err) {
    err.clear();
    std::error_code ec;
    if (!fs::is_directory(base, ec) || ec) {
        err = "path does not exist or is not a directory: " + base;
        return {};
    }

    auto res = run_process(
        {"git", "-C", base, "ls-files", "--cached", "--others", "--exclude-standard"},
        SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT, SERVER_TOOL_GIT_LS_FILES_TIMEOUT);

    if (res.exit_code == 0 && !res.timed_out) {
        std::vector<std::string> result;
        std::istringstream iss(res.output);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::replace(line.begin(), line.end(), '\\', '/');
            std::error_code fec;
            if (fs::is_regular_file(fs::path(base) / line, fec)) {
                result.push_back(line);
            }
        }
        return result;
    }

    return server_tool_list_files_fallback(base);
}

// bare pattern (no '/') matches the basename at any depth
// pattern containing '/' matches the full relative path, auto-anchored with "**/"
// unless it's already anchored (mirrors "fd --glob" vs "fd --glob --full-path")
static bool server_tool_path_glob_match(const std::string & pattern, const std::string & rel_path) {
    if (pattern.find('/') == std::string::npos) {
        return glob_match(pattern, fs::path(rel_path).filename().string());
    }
    if (pattern == "**" || pattern.rfind("**/", 0) == 0 || pattern.rfind('/', 0) == 0) {
        return glob_match(pattern, rel_path);
    }
    return glob_match("**/" + pattern, rel_path);
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
    server_tool_read_file() {
        name = "read_file";
        display_name = "Read file";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the contents of a file. Optionally specify a 1-based line range. "
                                "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192 ...\")."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                        {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                        {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                        {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path  = params.at("path").get<std::string>();
        int  start_line   = json_value(params, "start_line", 1);
        int  end_line     = json_value(params, "end_line",  -1); // -1 = no limit
        bool append_loc   = json_value(params, "append_loc", false);

        std::error_code ec;
        uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return {{"error", "cannot stat file: " + ec.message()}};
        }
        if (file_size > SERVER_TOOL_READ_FILE_MAX_SIZE && end_line == -1) {
            return {{"error", string_format(
                "file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                (size_t)file_size, SERVER_TOOL_READ_FILE_MAX_SIZE)}};
        }

        std::ifstream f(path);
        if (!f) {
            return {{"error", "failed to open file: " + path}};
        }

        std::string result;
        std::string line;
        int lineno = 0;

        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\u2192 " + line + "\n";
            } else {
                out_line = line + "\n";
            }

            if (result.size() + out_line.size() > SERVER_TOOL_READ_FILE_MAX_SIZE) {
                result += "[output truncated]";
                break;
            }
            result += out_line;
        }

        return {{"plain_text_response", result}};
    }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100;

struct server_tool_file_glob_search : server_tool {
    server_tool_file_glob_search() {
        name = "file_glob_search";
        display_name = "File search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Recursively search for files matching a glob pattern under a directory. "
                    "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
                    "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
                    "A pattern with no '/' (e.g. \"*.cpp\") matches the file's basename at any depth. "
                    "A pattern containing '/' matches the full relative path; unless already anchored with "
                    "\"**/\" or a leading '/', it is automatically prefixed with \"**/\"."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Base directory to search in"}}},
                        {"include", {{"type", "string"}, {"description", "Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **"}}},
                        {"exclude", {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string base    = params.at("path").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));

        std::string err;
        auto files = server_tool_list_files(base, err);
        if (!err.empty()) {
            return {{"error", err}};
        }

        std::vector<std::string> matches;
        for (const auto & rel : files) {
            if (!server_tool_path_glob_match(include, rel)) continue;
            if (!exclude.empty() && server_tool_path_glob_match(exclude, rel)) continue;
            matches.push_back(rel);
        }

        size_t total = matches.size();
        size_t shown = std::min(total, SERVER_TOOL_FILE_SEARCH_MAX_RESULTS);

        std::ostringstream output_text;
        for (size_t i = 0; i < shown; i++) {
            output_text << matches[i] << "\n";
        }

        output_text << "\n---\nTotal matches: " << total << "\n";
        if (total > shown) {
            output_text << string_format(
                "[%zu results limit reached (%zu total matches). Refine the glob pattern to narrow the search.]\n",
                shown, total);
        }

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100;

struct server_tool_grep_search : server_tool {
    server_tool_grep_search() {
        name = "grep_search";
        display_name = "Grep search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Search for a pattern in files under a path. Returns matching lines with file paths "
                    "(and, unless searching a single file, paths relative to the given directory). "
                    "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
                    "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
                    "include/exclude: a pattern with no '/' matches the basename at any depth; a pattern "
                    "containing '/' matches the full relative path (auto-anchored with \"**/\" unless already anchored)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                        {"pattern",             {{"type", "string"},  {"description", "Pattern to search for (regular expression unless literal is true)"}}},
                        {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                        {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                        {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                        {"literal",             {{"type", "boolean"}, {"description", "Treat pattern as a literal string instead of a regular expression (default: false)"}}},
                        {"ignore_case",         {{"type", "boolean"}, {"description", "Case-insensitive search (default: false)"}}},
                        {"context_lines",       {{"type", "integer"}, {"description", "Number of lines of context to show before and after each match (default: 0)"}}},
                    }},
                    {"required", json::array({"path", "pattern"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path        = params.at("path").get<std::string>();
        std::string pat_str     = params.at("pattern").get<std::string>();
        std::string include     = json_value(params, "include", std::string("**"));
        std::string exclude     = json_value(params, "exclude", std::string(""));
        bool        show_lineno = json_value(params, "return_line_numbers", false);
        bool        literal     = json_value(params, "literal", false);
        bool        ignore_case = json_value(params, "ignore_case", false);
        int         ctx_lines   = std::max(0, json_value(params, "context_lines", 0));

        std::string pattern_src = pat_str;
        if (literal) {
            static const std::string specials = "\\^$.|?*+()[]{}";
            std::string escaped;
            escaped.reserve(pat_str.size() * 2);
            for (char c : pat_str) {
                if (specials.find(c) != std::string::npos) escaped += '\\';
                escaped += c;
            }
            pattern_src = escaped;
        }

        std::regex pattern;
        try {
            auto flags = std::regex::ECMAScript;
            if (ignore_case) flags |= std::regex::icase;
            pattern = std::regex(pattern_src, flags);
        } catch (const std::regex_error & e) {
            return {{"error", std::string("invalid regex: ") + e.what()}};
        }

        // collect (absolute_path, display_path) pairs to search
        std::vector<std::pair<std::string, std::string>> files;

        std::error_code ec;
        if (fs::is_regular_file(path, ec)) {
            files.emplace_back(path, path);
        } else if (fs::is_directory(path, ec)) {
            std::string err;
            auto candidates = server_tool_list_files(path, err);
            if (!err.empty()) {
                return {{"error", err}};
            }
            for (const auto & rel : candidates) {
                if (!server_tool_path_glob_match(include, rel)) continue;
                if (!exclude.empty() && server_tool_path_glob_match(exclude, rel)) continue;
                files.emplace_back((fs::path(path) / rel).string(), rel);
            }
        } else {
            return {{"error", "path does not exist: " + path}};
        }

        std::ostringstream output_text;
        size_t total = 0;
        bool limit_reached = false;
        bool show_num = show_lineno || ctx_lines > 0;

        for (const auto & file_entry : files) {
            if (limit_reached) break;
            const std::string & fpath        = file_entry.first;
            const std::string & display_path = file_entry.second;

            std::ifstream f(fpath);
            if (!f) continue;
            std::vector<std::string> lines;
            {
                std::string line;
                while (std::getline(f, line)) lines.push_back(line);
            }

            for (size_t i = 0; i < lines.size(); i++) {
                if (total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) {
                    limit_reached = true;
                    break;
                }
                if (!std::regex_search(lines[i], pattern)) continue;

                long ctx_start = ctx_lines > 0 ? std::max<long>(0, (long) i - ctx_lines) : (long) i;
                long ctx_end   = ctx_lines > 0 ? std::min<long>((long) lines.size() - 1, (long) i + ctx_lines) : (long) i;

                for (long j = ctx_start; j <= ctx_end; j++) {
                    bool is_match = (j == (long) i);
                    output_text << display_path << (is_match ? ':' : '-');
                    if (show_num) {
                        output_text << (j + 1) << (is_match ? ':' : '-');
                    }
                    output_text << lines[j] << "\n";
                }
                if (ctx_lines > 0) {
                    output_text << "--\n";
                }
                total++;
            }
        }

        output_text << "\n---\nTotal matches: " << total << "\n";
        if (limit_reached) {
            output_text << string_format(
                "[%zu matches limit reached. Narrow the path/pattern/include to see more.]\n",
                SERVER_TOOL_GREP_SEARCH_MAX_RESULTS);
        }

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE = 16 * 1024; // 16 KB
static constexpr int    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT     = 60;        // seconds

struct server_tool_exec_shell_command : server_tool {
    server_tool_exec_shell_command() {
        name = "exec_shell_command";
        display_name = "Execute shell command";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Execute a shell command and return its output (stdout and stderr combined)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command",         {{"type", "string"},  {"description", "Shell command to execute"}}},
                        {"timeout",         {{"type", "integer"}, {"description", string_format("Timeout in seconds (default 10, max %d)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT)}}},
                        {"max_output_size", {{"type", "integer"}, {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE)}}},
                    }},
                    {"required", json::array({"command"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string command   = params.at("command").get<std::string>();
        int    timeout        = json_value(params, "timeout",         10);
        size_t max_output     = (size_t) json_value(params, "max_output_size", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

        timeout    = std::min(timeout,    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
        max_output = std::min(max_output, SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif

        auto res = run_process(args, max_output, timeout);

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }

        return {{"plain_text_response", text_output}};
    }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
    server_tool_write_file() {
        name = "write_file";
        display_name = "Write file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                        {"content", {{"type", "string"}, {"description", "Content to write"}}},
                    }},
                    {"required", json::array({"path", "content"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string content = params.at("content").get<std::string>();

        std::error_code ec;
        fs::path fpath(path);
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) {
                return {{"error", "failed to create directories: " + ec.message()}};
            }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        f << content;
        if (!f) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file written successfully"}, {"path", path}, {"bytes", content.size()}};
    }
};

//
// edit_file: edit file content via line-based changes
//

struct server_tool_edit_file : server_tool {
    server_tool_edit_file() {
        name = "edit_file";
        display_name = "Edit file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Edit a file by applying a list of line-based changes. "
                    "Each change targets a 1-based inclusive line range and has a mode: "
                    "\"replace\" (replace lines with content), "
                    "\"delete\" (remove lines, content must be empty string), "
                    "\"append\" (insert content after line_end). "
                    "Set line_start to -1 to target the end of file (line_end is ignored in that case). "
                    "Changes must not overlap. They are applied in reverse line order automatically."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path to the file to edit"}}},
                        {"changes", {
                            {"type", "array"},
                            {"description", "List of changes to apply"},
                            {"items", {
                                {"type", "object"},
                                {"properties", {
                                    {"mode",       {{"type", "string"},  {"description", "\"replace\", \"delete\", or \"append\""}}},
                                    {"line_start", {{"type", "integer"}, {"description", "First line of the range (1-based); use -1 for end of file"}}},
                                    {"line_end",   {{"type", "integer"}, {"description", "Last line of the range (1-based, inclusive); ignored when line_start is -1"}}},
                                    {"content",    {{"type", "string"},  {"description", "Content to insert; must be empty string for delete mode"}}},
                                }},
                                {"required", json::array({"mode", "line_start", "line_end", "content"})},
                            }},
                        }},
                    }},
                    {"required", json::array({"path", "changes"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path = params.at("path").get<std::string>();
        const json & changes = params.at("changes");

        if (!changes.is_array()) {
            return {{"error", "\"changes\" must be an array"}};
        }

        // read file into lines
        std::ifstream fin(path);
        if (!fin) {
            return {{"error", "failed to open file: " + path}};
        }
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(fin, line)) {
                lines.push_back(line);
            }
        }
        fin.close();

        // validate and collect changes, then sort descending by line_start
        struct change_entry {
            std::string mode;
            int line_start; // 1-based
            int line_end;   // 1-based inclusive
            std::string content;
        };
        std::vector<change_entry> entries;
        entries.reserve(changes.size());

        for (const auto & ch : changes) {
            change_entry e;
            e.mode       = ch.at("mode").get<std::string>();
            e.line_start = ch.at("line_start").get<int>();
            e.line_end   = ch.at("line_end").get<int>();
            e.content    = ch.at("content").get<std::string>();

            if (e.mode != "replace" && e.mode != "delete" && e.mode != "append") {
                return {{"error", "invalid mode \"" + e.mode + "\"; must be replace, delete, or append"}};
            }
            if (e.mode == "delete" && !e.content.empty()) {
                return {{"error", "content must be empty string for delete mode"}};
            }
            int n = (int) lines.size();
            if (e.line_start == -1) {
                // -1 targets end of file -> valid for append only; line_end is ignored
                if (e.mode != "append") {
                    return {{"error", "line_start -1 (end of file) is only valid for append mode"}};
                }
                // append at end of file: insert position is the current line count
                e.line_start = n;
                e.line_end   = n;
            } else {
                if (e.line_start < 1 || e.line_end < e.line_start) {
                    return {{"error", string_format("invalid line range [%d, %d]", e.line_start, e.line_end)}};
                }
                if (e.line_end > n) {
                    return {{"error", string_format("line_end %d exceeds file length %d", e.line_end, n)}};
                }
            }
            entries.push_back(std::move(e));
        }

        // sort descending so earlier-indexed changes don't shift later ones
        std::sort(entries.begin(), entries.end(), [](const change_entry & a, const change_entry & b) {
            return a.line_start > b.line_start;
        });

        // apply changes (0-based indices internally)
        for (const auto & e : entries) {
            int idx_start = e.line_start - 1; // 0-based
            int idx_end   = e.line_end   - 1; // 0-based inclusive

            // split content into lines (preserve trailing newline awareness)
            std::vector<std::string> new_lines;
            if (!e.content.empty()) {
                std::istringstream ss(e.content);
                std::string ln;
                while (std::getline(ss, ln)) {
                    new_lines.push_back(ln);
                }
                // if content ends with \n, getline consumed it — no extra empty line needed
                // if content does NOT end with \n, last line is still captured correctly
            }

            if (e.mode == "replace") {
                // erase [idx_start, idx_end] and insert new_lines
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
                lines.insert(lines.begin() + idx_start, new_lines.begin(), new_lines.end());
            } else if (e.mode == "delete") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
            } else { // append
                // insert after idx_end; idx_end + 1 == lines.size() for end-of-file append
                lines.insert(lines.begin() + (idx_end + 1), new_lines.begin(), new_lines.end());
            }
        }

        // write file back
        std::ofstream fout(path, std::ios::binary);
        if (!fout) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        for (size_t i = 0; i < lines.size(); i++) {
            fout << lines[i];
            if (i + 1 < lines.size()) {
                fout << "\n";
            }
        }
        if (!lines.empty()) {
            fout << "\n";
        }
        if (!fout) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file edited successfully"}, {"path", path}, {"lines", (int) lines.size()}};
    }
};

//
// get_datetime: returns the current date and time
//

struct server_tool_get_datetime : server_tool {
    server_tool_get_datetime() {
        name = "get_datetime";
        display_name = "Get Date & Time";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Returns the current date and time"},
            }},
        };
    }

    json invoke(json) override {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        return {{"result", std::ctime(&time)}};
    }
};

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> build_tools() {
    std::vector<std::unique_ptr<server_tool>> tools;
    tools.push_back(std::make_unique<server_tool_read_file>());
    tools.push_back(std::make_unique<server_tool_file_glob_search>());
    tools.push_back(std::make_unique<server_tool_grep_search>());
    tools.push_back(std::make_unique<server_tool_exec_shell_command>());
    tools.push_back(std::make_unique<server_tool_write_file>());
    tools.push_back(std::make_unique<server_tool_edit_file>());
    tools.push_back(std::make_unique<server_tool_get_datetime>());
    return tools;
}

void server_tools::setup(const std::vector<std::string> & enabled_tools) {
    if (!enabled_tools.empty()) {
        std::unordered_set<std::string> enabled_set(enabled_tools.begin(), enabled_tools.end());
        auto all_tools = build_tools();

        // collect all known tool names for validation
        std::vector<std::string> known_names;
        known_names.reserve(all_tools.size());
        for (const auto & t : all_tools) {
            known_names.push_back(t->name);
        }

        // validate that every requested tool is known
        for (const auto & name : enabled_tools) {
            if (name == "all") continue;
            if (std::find(known_names.begin(), known_names.end(), name) == known_names.end()) {
                throw std::runtime_error(string_format(
                    "unknown tool \"%s\". available tools: %s",
                    name.c_str(),
                    string_join(known_names, ", ").c_str()));
            }
        }

        tools.clear();
        for (auto & t : all_tools) {
            if (enabled_set.count(t->name) > 0 || enabled_set.count("all") > 0) {
                tools.push_back(std::move(t));
            }
        }
    }

    handle_get = [this](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json result = json::array();
            for (const auto & t : tools) {
                result.push_back(t->to_json());
            }
            res->data = safe_json_to_str(result);
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    handle_post = [this](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json body = json::parse(req.body);
            std::string tool_name = body.at("tool").get<std::string>();
            json params = body.value("params", json::object());
            json result = invoke(tool_name, params);
            res->data   = safe_json_to_str(result);
        } catch (const json::exception & e) {
            res->status = 400;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };
}

json server_tools::invoke(const std::string & name, const json & params) {
    for (auto & t : tools) {
        if (t->name == name) {
            return t->invoke(params);
        }
    }
    return {{"error", "unknown tool: " + name}};
}
