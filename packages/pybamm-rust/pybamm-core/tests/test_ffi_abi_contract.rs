//! ABI drift test: the C++ IDAKLU consumer resolves Rust FFI symbols by name via `dlsym`, so a
//! signature change would be silent UB unless both sides are checked against each other here.

use std::collections::HashMap;
use std::fs;

use pybamm_core::ffi;

const FFI_RS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi.rs");
const CONSUMER_HEADER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../pybammsolvers/src/pybammsolvers/idaklu_source/Expressions/Rust/pybamm_rust_ffi.h"
);
const CONSUMER_IMPL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../pybammsolvers/src/pybammsolvers/idaklu_source/Expressions/Rust/RustFunctions.hpp"
);

/// Map a Rust or C type spelling to a canonical token, panicking on an unknown spelling.
fn canon(raw: &str) -> String {
    let t = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let canonical = match t.as_str() {
        "f64" | "double" => "f64",
        "u32" | "uint32_t" => "u32",
        "c_int" | "int" => "i32",
        "*const f64" | "const double*" | "const double *" => "*const f64",
        "*mut f64" | "double*" | "double *" => "*mut f64",
        "*const c_void" | "const void*" | "const void *" => "*const void",
        "*mut c_void" | "void*" | "void *" => "*mut void",
        other => panic!(
            "ABI drift test: unrecognized type spelling {other:?}. \
             Add it to `canon()` on both language sides."
        ),
    };
    canonical.to_string()
}

/// Parse the Rust `extern "C"` exports: symbol -> (arg canon types, return canon).
fn parse_rust_exports(src: &str) -> HashMap<String, (Vec<String>, String)> {
    let mut out = HashMap::new();
    let bytes = src.as_bytes();
    let needle = "extern \"C\" fn ";
    let mut cursor = 0;
    while let Some(rel) = src[cursor..].find(needle) {
        let name_start = cursor + rel + needle.len();
        let paren = src[name_start..].find('(').expect("fn without '('");
        let name = src[name_start..name_start + paren].trim().to_string();

        let args_start = name_start + paren + 1;
        let mut depth = 1;
        let mut i = args_start;
        while depth > 0 {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {},
            }
            i += 1;
        }
        let args_str = &src[args_start..i - 1];

        let brace = src[i..].find('{').expect("fn without body");
        let ret_str = src[i..i + brace].trim();
        let ret = ret_str
            .strip_prefix("->")
            .map_or_else(|| "void".to_string(), |r| canon(r.trim()));

        out.insert(name, (parse_rust_args(args_str), ret));
        cursor = i;
    }
    out
}

fn parse_rust_args(s: &str) -> Vec<String> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let ty = p
                .split_once(':')
                .map(|(_, ty)| ty)
                .expect("arg must be `name: type`");
            canon(ty.trim())
        })
        .collect()
}

/// Whether `symbol`'s attributes include `#[unsafe(no_mangle)]`, without which `dlsym` misses it.
fn is_unmangled(src: &str, symbol: &str) -> bool {
    let definition = src
        .find(&format!("extern \"C\" fn {symbol}("))
        .unwrap_or_else(|| panic!("ffi.rs defines no `{symbol}`"));
    src[..definition]
        .lines()
        .rev()
        .skip(1)
        .map(str::trim)
        .take_while(|line| line.starts_with("#[") || line.starts_with("///"))
        .any(|line| line == "#[unsafe(no_mangle)]")
}

/// FNV-1a 64-bit, since `DefaultHasher` is not stable across Rust versions or platforms.
fn stable_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Order-independent fingerprint of the export set, as the `dlsym` contract is by name.
fn compute_abi_hash(exports: &HashMap<String, (Vec<String>, String)>) -> u64 {
    let mut entries: Vec<String> = exports
        .iter()
        .map(|(name, (args, ret))| format!("{name}({})->{ret}", args.join(",")))
        .collect();
    entries.sort();
    stable_hash(entries.join("\n").as_bytes())
}

struct Consumer {
    define_version: u32,
    typedefs: HashMap<String, (Vec<String>, String)>,
    bindings: Vec<Binding>,
}

/// One `t.<field> = load_symbol<TYPEDEF>("symbol")` statement of `rust_ffi()`.
struct Binding {
    field: String,
    typedef: String,
    symbol: String,
}

/// The `RustFfi` field assigned by the statement that `prefix` ends inside.
fn assigned_field(prefix: &str) -> String {
    let statement = prefix.rsplit([';', '{', '}']).next().unwrap_or(prefix);
    let target: String = statement
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("//"))
        .collect();
    target
        .strip_prefix("t.")
        .and_then(|rest| rest.strip_suffix('='))
        .map_or_else(
            || panic!("`load_symbol` must assign a `t.<field>`, not `{target}`"),
            |field| field.trim().to_string(),
        )
}

fn parse_consumer(src: &str) -> Consumer {
    let define_version = src
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("#define PYBAMM_RUST_ABI_VERSION")
                .map(|rest| {
                    rest.trim()
                        .parse::<u32>()
                        .expect("PYBAMM_RUST_ABI_VERSION must be a u32 literal")
                })
        })
        .expect("consumer header must `#define PYBAMM_RUST_ABI_VERSION N`");

    // `using NAME = RET (*)(ARGS);` — may span multiple physical lines.
    let mut typedefs = HashMap::new();
    let mut rest = src;
    while let Some(p) = rest.find("using ") {
        let after = &rest[p + "using ".len()..];
        let semi = after.find(';').expect("`using` without ';'");
        let stmt = &after[..semi];
        rest = &after[semi + 1..];
        if let Some((name, def)) = stmt.split_once('=') {
            let def = def.trim();
            if let Some(star) = def.find("(*)") {
                let ret = canon(def[..star].trim());
                let tail = &def[star + 3..];
                let open = tail.find('(').expect("typedef without '('");
                let close = tail.rfind(')').expect("typedef without ')'");
                let args = parse_c_args(&tail[open + 1..close]);
                typedefs.insert(name.trim().to_string(), (args, ret));
            }
        }
    }

    // `load_symbol<TYPEDEF>("symbol")` — the authoritative typedef->symbol map.
    let mut bindings = Vec::new();
    let mut rest = src;
    while let Some(p) = rest.find("load_symbol<") {
        let field = assigned_field(&rest[..p]);
        let after = &rest[p + "load_symbol<".len()..];
        let gt = after.find('>').expect("load_symbol<...> without '>'");
        let typedef = after[..gt].trim().to_string();
        let tail = &after[gt + 1..];
        let q1 = tail.find('"').expect("load_symbol without symbol string");
        let q2 = tail[q1 + 1..]
            .find('"')
            .expect("unterminated symbol string");
        let symbol = tail[q1 + 1..q1 + 1 + q2].to_string();
        bindings.push(Binding {
            field,
            typedef,
            symbol,
        });
        rest = &tail[q1 + 1 + q2..];
    }

    Consumer {
        define_version,
        typedefs,
        bindings,
    }
}

fn parse_c_args(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.is_empty() || s == "void" {
        return Vec::new();
    }
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| canon(p.trim()))
        .collect()
}

#[test]
fn c_consumer_matches_rust_exports() {
    let rust_src =
        fs::read_to_string(FFI_RS).unwrap_or_else(|e| panic!("cannot read {FFI_RS}: {e}"));
    let c_src = fs::read_to_string(CONSUMER_HEADER)
        .unwrap_or_else(|e| panic!("cannot read consumer header {CONSUMER_HEADER}: {e}"));

    let exports = parse_rust_exports(&rust_src);
    let consumer = parse_consumer(&c_src);

    assert_eq!(
        ffi::RUST_ABI_VERSION,
        consumer.define_version,
        "RUST_ABI_VERSION ({}) != PYBAMM_RUST_ABI_VERSION ({})",
        ffi::RUST_ABI_VERSION,
        consumer.define_version
    );

    for Binding {
        field,
        typedef,
        symbol,
    } in &consumer.bindings
    {
        // The count queries share one type, so only the name tells two of them apart.
        assert_eq!(
            symbol.strip_prefix("pybamm_rust_"),
            Some(field.as_str()),
            "`t.{field}` is loaded from `{symbol}`"
        );
        let (c_args, c_ret) = consumer
            .typedefs
            .get(typedef)
            .unwrap_or_else(|| panic!("no typedef `{typedef}` for symbol `{symbol}`"));
        let (r_args, r_ret) = exports.get(symbol).unwrap_or_else(|| {
            panic!("consumer binds `{symbol}` but ffi.rs exports no such function")
        });
        assert_eq!(c_ret, r_ret, "return type mismatch for `{symbol}`");
        assert_eq!(c_args, r_args, "argument type mismatch for `{symbol}`");
        assert!(
            is_unmangled(&rust_src, symbol),
            "`{symbol}` lacks `#[unsafe(no_mangle)]`, so the consumer cannot resolve it"
        );
    }

    for symbol in exports.keys() {
        let bound = consumer
            .bindings
            .iter()
            .any(|binding| binding.symbol == *symbol);
        assert!(
            bound,
            "ffi.rs exports `{symbol}`, which IDAKLU never resolves; delete it or bind it"
        );
    }
}

/// The consumer names statuses by its own `#define`s, so each must equal a Rust constant.
#[test]
fn status_codes_match_the_consumer_header() {
    let header = fs::read_to_string(CONSUMER_HEADER)
        .unwrap_or_else(|e| panic!("cannot read consumer header {CONSUMER_HEADER}: {e}"));
    let defines: HashMap<&str, i32> = header
        .lines()
        .filter_map(|line| {
            let (name, value) = line.trim().strip_prefix("#define ")?.split_once(' ')?;
            Some((name, value.trim().parse().ok()?))
        })
        .collect();

    let statuses = [
        ("PYBAMM_SUCCESS", ffi::SUCCESS),
        ("PYBAMM_ERROR_NULL", ffi::ERROR_NULL_POINTER),
        ("PYBAMM_ERROR_PANIC", ffi::ERROR_PANIC),
        ("PYBAMM_ERROR_INVALID_PARAM", ffi::ERROR_INVALID_PARAM),
        ("PYBAMM_ERROR_INVALID_OUTPUT", ffi::ERROR_INVALID_OUTPUT),
        ("PYBAMM_ERROR_NO_SENS", ffi::ERROR_NO_SENS),
        ("PYBAMM_ERROR_NO_OUTPUTS", ffi::ERROR_NO_OUTPUTS),
        ("PYBAMM_ERROR_NO_ALG", ffi::ERROR_NO_ALG),
        ("PYBAMM_ERROR_NO_EVENTS", ffi::ERROR_NO_EVENTS),
    ];
    for (define, status) in statuses {
        assert_eq!(
            defines.get(define),
            Some(&status),
            "`{define}` in the consumer header differs from ffi.rs"
        );
    }

    // A status Rust never returns would name a failure mode that cannot occur.
    for define in defines
        .keys()
        .filter(|name| **name == "PYBAMM_SUCCESS" || name.starts_with("PYBAMM_ERROR_"))
    {
        assert!(
            statuses.iter().any(|(name, _)| name == define),
            "the consumer header defines `{define}`, which ffi.rs never returns"
        );
    }
}

#[test]
fn abi_export_set_hash_is_pinned() {
    let rust_src =
        fs::read_to_string(FFI_RS).unwrap_or_else(|e| panic!("cannot read {FFI_RS}: {e}"));
    let exports = parse_rust_exports(&rust_src);
    let actual = compute_abi_hash(&exports);
    let expected = ffi::EXPECTED_ABI_HASH;
    assert_eq!(
        actual, expected,
        "ABI export set changed: computed hash {actual:#018x} != EXPECTED_ABI_HASH \
         {expected:#018x}. The Rust extern \"C\" surface changed. Update \
         EXPECTED_ABI_HASH in ffi.rs to {actual:#018x} AND bump both RUST_ABI_VERSION \
         and PYBAMM_RUST_ABI_VERSION in lockstep."
    );
}

/// A dropped status would pass a failed call's stale output to SUNDIALS, so every call must go
/// through `PYBAMM_RUST_CALL` or `PYBAMM_RUST_VALUE`, which raise on failure.
#[test]
fn consumer_checks_every_ffi_return_status() {
    let src = fs::read_to_string(CONSUMER_IMPL)
        .unwrap_or_else(|e| panic!("cannot read {CONSUMER_IMPL}: {e}"));

    let unchecked: Vec<&str> = src
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("//") && !line.starts_with('*'))
        .filter(|line| line.contains("rust_ffi()."))
        .collect();

    assert!(
        unchecked.is_empty(),
        "these calls drop the Rust FFI status code; wrap each in PYBAMM_RUST_CALL \
         or PYBAMM_RUST_VALUE:\n  {}",
        unchecked.join("\n  ")
    );
}

/// `rust_ffi()` resolves its whole table eagerly, so `RustFfi` must bind only what is called.
#[test]
fn every_bound_ffi_entry_point_is_called() {
    // Resolved by the version handshake, not the call macros.
    const RESOLVER_INTERNAL: [&str; 1] = ["abi_version"];

    let header = fs::read_to_string(CONSUMER_HEADER)
        .unwrap_or_else(|e| panic!("cannot read consumer header {CONSUMER_HEADER}: {e}"));
    let impl_src = fs::read_to_string(CONSUMER_IMPL)
        .unwrap_or_else(|e| panic!("cannot read {CONSUMER_IMPL}: {e}"));

    let open = header
        .find("struct RustFfi {")
        .expect("consumer header must declare `struct RustFfi {`");
    let body_start = open + "struct RustFfi {".len();
    let body_len = header[body_start..]
        .find('}')
        .expect("`struct RustFfi` without closing '}'");
    // Each field is `<function-pointer type> <name>;`.
    let fields: Vec<&str> = header[body_start..body_start + body_len]
        .lines()
        .map(str::trim)
        .filter(|l| l.ends_with(';'))
        .filter_map(|l| l.trim_end_matches(';').split_whitespace().nth(1))
        .collect();
    assert!(!fields.is_empty(), "parsed no fields from `struct RustFfi`");

    let uncalled: Vec<&str> = fields
        .into_iter()
        .filter(|f| !RESOLVER_INTERNAL.contains(f))
        .filter(|f| {
            !impl_src.contains(&format!("PYBAMM_RUST_CALL({f},"))
                && !impl_src.contains(&format!("PYBAMM_RUST_VALUE({f},"))
        })
        .collect();

    assert!(
        uncalled.is_empty(),
        "these `RustFfi` entry points are resolved but never called; drop them \
         from the table (and re-add when a call site lands):\n  {}",
        uncalled.join("\n  ")
    );
}
