//! ABI drift test: the C++ IDAKLU consumer reads Rust's `FfiTable` and `EvaluatorHandle` through
//! its own C mirror, so a layout change would be silent UB unless the two are compared here.

use std::collections::BTreeMap;
use std::fs;

use pybamm_core::ffi;

const FFI_RS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi.rs");
const RUST_IDAKLU: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../pybammsolvers/src/pybammsolvers/idaklu_source/Expressions/Rust"
);

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

fn consumer_header() -> String {
    read(&format!("{RUST_IDAKLU}/pybamm_rust_ffi.h"))
}

/// Every C++ source of idaklu but the FFI header, which defines the call macros themselves.
fn consumer_sources() -> Vec<(String, String)> {
    fn walk(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
        for entry in
            fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()))
        {
            let path = entry.expect("readable directory entry").path();
            let is_source = path.extension().is_some_and(|ext| {
                ["cpp", "hpp", "inl", "h"].contains(&ext.to_str().unwrap_or(""))
            });
            if path.is_dir() {
                walk(&path, out);
            } else if is_source && !path.ends_with("pybamm_rust_ffi.h") {
                let path = path.to_string_lossy().into_owned();
                out.push((path.clone(), read(&path)));
            }
        }
    }
    let mut sources = Vec::new();
    walk(
        std::path::Path::new(&format!("{RUST_IDAKLU}/../../..")),
        &mut sources,
    );
    assert!(
        sources
            .iter()
            .any(|(path, _)| path.ends_with("RustFunctions.hpp")),
        "found no consumer sources under {RUST_IDAKLU}/../../.."
    );
    sources
}

/// Map a Rust or C type spelling to a canonical token, panicking on an unknown spelling.
fn canon(raw: &str) -> String {
    let t = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let canonical = match t.as_str() {
        "" | "void" => "void",
        "f64" | "double" => "f64",
        "u32" | "uint32_t" => "u32",
        "c_int" | "int" => "i32",
        "*const f64" | "const double*" | "const double *" => "*const f64",
        "*mut f64" | "double*" | "double *" => "*mut f64",
        "*const c_void" | "const void*" | "const void *" => "*const void",
        "*mut c_void" | "void*" | "void *" => "*mut void",
        "*mut EvaluatorHandle" | "PybammRustEvaluator*" => "*mut handle",
        "&'static FfiTable" | "const PybammRustFfi*" => "*const table",
        other => panic!(
            "ABI drift test: unrecognized type spelling {other:?}. \
             Add it to `canon()` on both language sides."
        ),
    };
    canonical.to_string()
}

/// A struct's fields in declaration order: `(name, canonical type)`.
type Fields = Vec<(String, String)>;

/// The text between `open`, which must end in `{`, and the next `close`.
fn body<'a>(src: &'a str, open: &str, close: &str) -> &'a str {
    let start = src
        .find(open)
        .unwrap_or_else(|| panic!("no `{open}` found"))
        + open.len();
    let len = src[start..]
        .find(close)
        .unwrap_or_else(|| panic!("`{open}` without `{close}`"));
    &src[start..start + len]
}

/// Split `s` on `sep` outside parentheses.
fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let (mut parts, mut depth, mut start) = (Vec::new(), 0_i32, 0);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            c if c == sep && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            },
            _ => {},
        }
    }
    parts.push(&s[start..]);
    parts.into_iter().filter(|p| !p.trim().is_empty()).collect()
}

/// Canonical `fn(args)->ret` for a function-pointer type, given its args and return spelling.
fn fn_type(args: &str, ret: &str) -> String {
    let args: Vec<String> = split_top_level(args, ',').into_iter().map(canon).collect();
    format!("fn({})->{}", args.join(","), canon(ret))
}

/// Fields of a Rust `#[repr(C)]` struct, skipping doc comments.
fn rust_fields(src: &str, name: &str) -> Fields {
    let body = body(src, &format!("pub struct {name} {{"), "\n}");
    let code: String = body
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    split_top_level(&code, ',')
        .into_iter()
        .map(|field| {
            let (field_name, ty) = field
                .trim()
                .strip_prefix("pub ")
                .and_then(|f| f.split_once(':'))
                .unwrap_or_else(|| panic!("`{name}` field must be `pub name: type`: {field}"));
            let ty = ty.trim();
            let ty = ty.strip_prefix("unsafe extern \"C\" fn").map_or_else(
                || canon(ty),
                |signature| {
                    let close = signature.rfind(')').expect("fn type without ')'");
                    let ret = signature[close + 1..].trim().trim_start_matches("->");
                    fn_type(&signature.trim()[1..close], ret)
                },
            );
            (field_name.trim().to_string(), ty)
        })
        .collect()
}

/// Fields of a C struct, skipping comments; function pointers are `ret (*name)(args)`.
fn c_fields(body: &str) -> Fields {
    let mut code = String::new();
    let mut rest = body;
    while let Some(open) = rest.find("/*") {
        code.push_str(&rest[..open]);
        let close = rest[open..].find("*/").expect("unterminated comment") + open + 2;
        rest = &rest[close..];
    }
    code.push_str(rest);
    code.split(';')
        .map(str::trim)
        .filter(|decl| !decl.is_empty())
        .map(|decl| {
            decl.find("(*").map_or_else(
                || {
                    let (ty, field_name) = decl.rsplit_once(' ').expect("field without a type");
                    (field_name.to_string(), canon(ty))
                },
                |star| {
                    let name_end = decl[star..].find(')').expect("`(*name` without ')'") + star;
                    let args = &decl[name_end + 1..];
                    let args = &args[args.find('(').expect("fn pointer without args") + 1
                        ..args.rfind(')').expect("fn pointer without ')'")];
                    (
                        decl[star + 2..name_end].trim().to_string(),
                        fn_type(args, &decl[..star]),
                    )
                },
            )
        })
        .collect()
}

fn rust_table() -> Fields {
    rust_fields(&read(FFI_RS), "FfiTable")
}

fn c_table() -> Fields {
    c_fields(body(
        &consumer_header(),
        "typedef struct PybammRustFfi {",
        "}",
    ))
}

/// The string or integer a `#define` in the consumer header expands to.
fn define(header: &str, name: &str) -> String {
    header
        .lines()
        .find_map(|line| line.trim().strip_prefix(&format!("#define {name} ")))
        .unwrap_or_else(|| panic!("the consumer header must `#define {name}`"))
        .trim()
        .trim_matches('"')
        .to_string()
}

#[test]
fn c_table_mirrors_the_rust_table() {
    let (rust, c) = (rust_table(), c_table());
    assert!(rust.len() > 2, "parsed no entries from `FfiTable`");
    for (i, (rust_field, c_field)) in rust.iter().zip(&c).enumerate() {
        assert_eq!(rust_field, c_field, "table field {i} differs");
    }
    assert_eq!(
        rust.len(),
        c.len(),
        "the tables have different field counts"
    );
    assert_eq!(
        rust[..2],
        [
            ("abi_version".to_string(), "u32".to_string()),
            ("size".to_string(), "u32".to_string()),
        ],
        "every version's table must lead with `abi_version` and `size`"
    );
}

#[test]
fn c_handle_mirrors_the_rust_handle() {
    let rust = rust_fields(&read(FFI_RS), "EvaluatorHandle");
    let c = c_fields(body(
        &consumer_header(),
        "struct PybammRustEvaluator {",
        "}",
    ));
    assert_eq!(rust, c);
    assert_eq!(
        rust[0].0, "ffi",
        "every version's handle must lead with `ffi`"
    );
}

#[test]
fn version_and_capsule_names_match_the_consumer_header() {
    let header = consumer_header();
    assert_eq!(
        define(&header, "PYBAMM_RUST_ABI_VERSION"),
        ffi::RUST_ABI_VERSION.to_string()
    );
    assert_eq!(
        define(&header, "PYBAMM_RUST_EVALUATOR_CAPSULE"),
        ffi::EVALUATOR_CAPSULE.to_str().unwrap()
    );
    assert_eq!(
        define(&header, "PYBAMM_RUST_CLAIMED_EVALUATOR_CAPSULE"),
        ffi::CLAIMED_EVALUATOR_CAPSULE.to_str().unwrap()
    );
}

/// The consumer names statuses by its own `#define`s, so each must equal a Rust constant.
#[test]
fn status_codes_match_the_consumer_header() {
    let header = consumer_header();
    let defines: BTreeMap<&str, i32> = header
        .lines()
        .filter_map(|line| {
            let (name, value) = line.trim().strip_prefix("#define ")?.split_once(' ')?;
            // Negative codes are parenthesised so they expand safely inside expressions
            let value = value.trim();
            let value = value
                .strip_prefix('(')
                .and_then(|inner| inner.strip_suffix(')'))
                .unwrap_or(value);
            Some((name, value.parse().ok()?))
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
        ("PYBAMM_ERROR_NO_UPSAMPLING", ffi::ERROR_NO_UPSAMPLING),
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

/// Render `fields` as the `(name, type)` literals of a frozen layout below.
fn render(fields: &Fields) -> String {
    fields
        .iter()
        .map(|(name, ty)| format!("    ({name:?}, {ty:?}),"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The version whose layout is frozen below, and that layout. A consumer built for version 2
/// reads these fields at these offsets, so within the version the handle stays as it is and
/// the table only grows at its end. Bumping `RUST_ABI_VERSION` replaces all three.
const FROZEN_VERSION: u32 = 2;
const FROZEN_HANDLE: &[(&str, &str)] = &[("ffi", "*const table"), ("model", "*mut void")];
const FROZEN_TABLE: &[(&str, &str)] = &[
    ("abi_version", "u32"),
    ("size", "u32"),
    ("release", "fn(*mut handle)->void"),
    ("n_states", "fn(*const void)->i32"),
    ("n_inputs", "fn(*const void)->i32"),
    ("jac_nnz", "fn(*const void)->i32"),
    ("n_sens_params", "fn(*const void)->i32"),
    (
        "eval_rhs",
        "fn(f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
    (
        "jac_assemble",
        "fn(f64,*const f64,*const f64,f64,*mut f64,*mut void)->i32",
    ),
    (
        "jac_action",
        "fn(f64,*const f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
    (
        "jac_action_batch",
        "fn(f64,*const f64,*const f64,i32,*const f64,*mut f64,*mut void)->i32",
    ),
    ("mass_action", "fn(*const f64,*mut f64,*mut void)->i32"),
    (
        "sens_eval_all",
        "fn(f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
    (
        "output_sens_project",
        "fn(f64,*const f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
    ("n_outputs", "fn(*const void)->i32"),
    ("output_len", "fn(*const void,i32)->i32"),
    (
        "output_eval_batch",
        "fn(*const f64,*const f64,i32,*const f64,*mut f64,*mut void)->i32",
    ),
    ("n_algebraic", "fn(*const void)->i32"),
    ("alg_jac_nnz", "fn(*const void)->i32"),
    (
        "alg_res",
        "fn(f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
    (
        "alg_jac_assemble",
        "fn(f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
    ("total_event_len", "fn(*const void)->i32"),
    (
        "events_eval",
        "fn(f64,*const f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
    ("upsampling_len", "fn(*const void)->i32"),
    (
        "upsampling_eval",
        "fn(f64,*const f64,*const f64,*mut f64,*mut void)->i32",
    ),
];

#[test]
fn the_versions_layout_is_only_appended_to() {
    let src = read(FFI_RS);
    let (handle, table) = (
        rust_fields(&src, "EvaluatorHandle"),
        rust_fields(&src, "FfiTable"),
    );
    let frozen = |layout: &[(&str, &str)]| -> Fields {
        layout
            .iter()
            .map(|(name, ty)| ((*name).to_string(), (*ty).to_string()))
            .collect()
    };
    assert_eq!(
        ffi::RUST_ABI_VERSION,
        FROZEN_VERSION,
        "RUST_ABI_VERSION changed: set FROZEN_VERSION to it, FROZEN_HANDLE to\n{}\nand \
         FROZEN_TABLE to\n{}",
        render(&handle),
        render(&table)
    );
    assert_eq!(
        handle,
        frozen(FROZEN_HANDLE),
        "`EvaluatorHandle` changed within ABI version {FROZEN_VERSION}, which a released \
         consumer would misread; bump RUST_ABI_VERSION and PYBAMM_RUST_ABI_VERSION"
    );
    assert!(
        table.starts_with(&frozen(FROZEN_TABLE)),
        "`FfiTable` changed other than by appending entries within ABI version \
         {FROZEN_VERSION}, which a released consumer would misread; bump RUST_ABI_VERSION and \
         PYBAMM_RUST_ABI_VERSION. It is now:\n{}",
        render(&table)
    );
}

/// rustc checks only each entry's type, so same-typed entries (the counts, say) could be crossed.
#[test]
fn every_table_entry_points_at_its_own_function() {
    let src = read(FFI_RS);
    let initializer = body(&src, "pub static FFI_TABLE: FfiTable = FfiTable {", "};");
    let entries: Vec<(&str, &str)> = split_top_level(initializer, ',')
        .into_iter()
        .filter_map(|entry| entry.split_once(':'))
        .map(|(field, value)| (field.trim(), value.trim()))
        .collect();
    assert_eq!(
        entries.len(),
        rust_table().len(),
        "FFI_TABLE must set every field"
    );
    let crossed: Vec<String> = entries
        .into_iter()
        .filter(|&(field, value)| match field {
            "abi_version" => value != "RUST_ABI_VERSION",
            "size" => value != "FFI_TABLE_SIZE",
            _ => value != format!("pybamm_rust_{field}"),
        })
        .map(|(field, value)| format!("{field}: {value}"))
        .collect();
    assert!(
        crossed.is_empty(),
        "these FFI_TABLE entries point at another entry's function:\n  {}",
        crossed.join("\n  ")
    );
}

/// The entries the consumer calls through a table directly, as `ffi->entry(` with `(file, line)`.
fn direct_calls() -> Vec<String> {
    let mut found = Vec::new();
    for (path, src) in consumer_sources() {
        for (number, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for (at, _) in code.match_indices("ffi->") {
                let rest = &code[at + "ffi->".len()..];
                let name_len = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                if rest[name_len..].trim_start().starts_with('(') {
                    found.push(format!("{path}:{}: {}", number + 1, line.trim()));
                }
            }
        }
    }
    found
}

/// A dropped status would pass a failed call's stale output to SUNDIALS, so every call must go
/// through `PYBAMM_RUST_CALL` or `PYBAMM_RUST_VALUE`, which raise on failure.
#[test]
fn consumer_checks_every_ffi_return_status() {
    let unchecked = direct_calls();
    assert!(
        unchecked.is_empty(),
        "these calls drop the Rust FFI status code; wrap each in PYBAMM_RUST_CALL \
         or PYBAMM_RUST_VALUE:\n  {}",
        unchecked.join("\n  ")
    );
}

/// An entry nothing calls would stay in the table for the rest of its major version.
#[test]
fn every_table_entry_is_called() {
    assert!(
        consumer_header().contains("handle->ffi->release(handle)"),
        "the consumer header must release each evaluator it owns"
    );
    let sources: String = consumer_sources().into_iter().map(|(_, src)| src).collect();
    // The entry each macro call names, which follows the evaluator it is called on.
    let called: Vec<&str> = ["PYBAMM_RUST_CALL(", "PYBAMM_RUST_VALUE("]
        .into_iter()
        .flat_map(|macro_call| sources.split(macro_call).skip(1))
        .filter_map(|args| args.split(',').nth(1).map(str::trim))
        .collect();
    let uncalled: Vec<String> = c_table()
        .into_iter()
        .filter(|(_, ty)| ty.ends_with("->i32"))
        .map(|(name, _)| name)
        .filter(|name| !called.contains(&name.as_str()))
        .collect();
    assert!(
        uncalled.is_empty(),
        "these `PybammRustFfi` entries are never called; drop them from the table:\n  {}",
        uncalled.join("\n  ")
    );
}
