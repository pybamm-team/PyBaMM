//! ABI contract drift test (T4.1).
//!
//! The C++ IDAKLU consumer resolves every Rust FFI entry point by name via
//! `dlsym`, which matches on name only — a signature change in `ffi.rs` would be
//! called through a wrong-typed function pointer (silent UB). This test parses
//! the Rust `extern "C"` exports and the C++ consumer's `RustFfi` typedefs and
//! asserts every consumed symbol matches (name + normalized arg/return types),
//! and that the two ABI version numbers are equal.

use std::collections::HashMap;
use std::fs;

const FFI_RS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi.rs");
const CONSUMER_HEADER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../pybammsolvers/src/pybammsolvers/idaklu_source/Expressions/Rust/pybamm_rust_ffi.h"
);
const CONSUMER_IMPL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../pybammsolvers/src/pybammsolvers/idaklu_source/Expressions/Rust/RustFunctions.hpp"
);

/// Map a Rust or C type spelling to a shared canonical token. Panics on an
/// unrecognized spelling so a new FFI type must be taught to both sides here.
fn canon(raw: &str) -> String {
    let t = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let canonical = match t.as_str() {
        "f64" | "double" => "f64",
        "u32" | "uint32_t" => "u32",
        "c_int" | "int" => "i32",
        "*const f64" | "const double*" | "const double *" => "*const f64",
        "*mut f64" | "double*" | "double *" => "*mut f64",
        "*mut c_int" | "int*" | "int *" => "*mut i32",
        "*mut i64" | "int64_t*" | "int64_t *" => "*mut i64",
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

/// Parse `pub const RUST_ABI_VERSION: u32 = N;` from source text.
fn parse_rust_abi_version(src: &str) -> u32 {
    src.lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("pub const RUST_ABI_VERSION: u32 =")
                .map(|rest| {
                    rest.trim()
                        .trim_end_matches(';')
                        .trim()
                        .parse::<u32>()
                        .expect("RUST_ABI_VERSION must be a u32 literal")
                })
        })
        .expect("ffi.rs must declare `pub const RUST_ABI_VERSION: u32 = N;`")
}

/// FNV-1a 64-bit — a tiny, portable, deterministic hash. Chosen over
/// `std::hash::DefaultHasher`, whose output is not guaranteed stable across
/// Rust versions or platforms; this value is committed as a golden constant.
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

/// Canonical, order-independent fingerprint of the parsed export set. Sorted by
/// symbol name because the dlsym contract is by-name — reordering exports is a
/// no-op for the ABI, so ordering must not affect the hash.
fn compute_abi_hash(exports: &HashMap<String, (Vec<String>, String)>) -> u64 {
    let mut entries: Vec<String> = exports
        .iter()
        .map(|(name, (args, ret))| format!("{name}({})->{ret}", args.join(",")))
        .collect();
    entries.sort();
    stable_hash(entries.join("\n").as_bytes())
}

/// Parse `pub const EXPECTED_ABI_HASH: u64 = 0x...;` from source text (hex,
/// `_` separators allowed), mirroring `parse_rust_abi_version`.
fn parse_expected_abi_hash(src: &str) -> u64 {
    src.lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("pub const EXPECTED_ABI_HASH: u64 =")
                .map(|rest| {
                    let raw = rest.trim().trim_end_matches(';').trim();
                    let raw = raw.strip_prefix("0x").unwrap_or(raw);
                    let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
                    u64::from_str_radix(&cleaned, 16)
                        .expect("EXPECTED_ABI_HASH must be a u64 hex literal")
                })
        })
        .expect("ffi.rs must declare `pub const EXPECTED_ABI_HASH: u64 = 0x...;`")
}

struct Consumer {
    define_version: u32,
    typedefs: HashMap<String, (Vec<String>, String)>,
    bindings: Vec<(String, String)>,
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
        let after = &rest[p + "load_symbol<".len()..];
        let gt = after.find('>').expect("load_symbol<...> without '>'");
        let typedef = after[..gt].trim().to_string();
        let tail = &after[gt + 1..];
        let q1 = tail.find('"').expect("load_symbol without symbol string");
        let q2 = tail[q1 + 1..]
            .find('"')
            .expect("unterminated symbol string");
        let symbol = tail[q1 + 1..q1 + 1 + q2].to_string();
        bindings.push((typedef, symbol));
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

    // Version numbers must be in lockstep.
    let rust_version = parse_rust_abi_version(&rust_src);
    assert_eq!(
        rust_version, consumer.define_version,
        "RUST_ABI_VERSION ({rust_version}) != PYBAMM_RUST_ABI_VERSION ({})",
        consumer.define_version
    );

    // Every consumed symbol must match its Rust export exactly.
    for (typedef, symbol) in &consumer.bindings {
        let (c_args, c_ret) = consumer
            .typedefs
            .get(typedef)
            .unwrap_or_else(|| panic!("no typedef `{typedef}` for symbol `{symbol}`"));
        let (r_args, r_ret) = exports.get(symbol).unwrap_or_else(|| {
            panic!("consumer binds `{symbol}` but ffi.rs exports no such function")
        });
        assert_eq!(c_ret, r_ret, "return type mismatch for `{symbol}`");
        assert_eq!(c_args, r_args, "argument type mismatch for `{symbol}`");
    }
}

#[test]
fn abi_export_set_hash_is_pinned() {
    let rust_src =
        fs::read_to_string(FFI_RS).unwrap_or_else(|e| panic!("cannot read {FFI_RS}: {e}"));
    let exports = parse_rust_exports(&rust_src);
    let actual = compute_abi_hash(&exports);
    let expected = parse_expected_abi_hash(&rust_src);
    assert_eq!(
        actual, expected,
        "ABI export set changed: computed hash {actual:#018x} != EXPECTED_ABI_HASH \
         {expected:#018x}. The Rust extern \"C\" surface changed. Update \
         EXPECTED_ABI_HASH in ffi.rs to {actual:#018x} AND bump both RUST_ABI_VERSION \
         and PYBAMM_RUST_ABI_VERSION in lockstep."
    );
}

/// Every Rust FFI entry point returns a status (`SUCCESS` or a negative
/// `ERROR_*`), and a panic caught at the boundary returns `ERROR_PANIC` without
/// having written the output buffer. A call that drops the status therefore lets
/// the previous step's stale data flow on into SUNDIALS as a valid evaluation.
///
/// Enforced structurally: the consumer must route every call through
/// `PYBAMM_RUST_CALL` (status-returning) or `PYBAMM_RUST_VALUE` (count-returning),
/// both of which raise on a failure code. No raw `rust_ffi().x(...)` may remain.
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

/// `rust_ffi()` resolves its whole table eagerly and throws on the first missing
/// symbol, so a bound-but-uncalled entry point couples IDAKLU to a Rust export it
/// does not need: renaming that export breaks the Rust path at first use for no
/// reason. `RustFfi` must therefore stay a subset of `ffi.rs` covering exactly
/// what the consumer calls, not a mirror of the full export surface.
#[test]
fn every_bound_ffi_entry_point_is_called() {
    // Resolved as part of the version handshake in `rust_ffi()` itself rather
    // than through the call macros.
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
    // Each field is `rust_<name>_t <name>;`.
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
