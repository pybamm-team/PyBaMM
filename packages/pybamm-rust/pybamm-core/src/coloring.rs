//! Column coloring that compresses a sparse Jacobian into few JVP passes.
//!
//! Only the pattern is consulted, so a coloring stays valid for every evaluation.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::sparsity::{SparsityPattern, transpose};

/// A column no sweep produces, so it never receives a color.
pub const UNSEEDED: usize = usize::MAX;

/// One flag per CSR entry of a [`SparsityPattern`].
///
/// Typed apart from `ColumnMask` because the two can have equal lengths and swap silently.
#[derive(Clone, Copy, Debug)]
pub struct EntryMask<'a>(&'a [bool]);

/// One flag per column of a [`SparsityPattern`].
#[derive(Clone, Copy, Debug)]
struct ColumnMask<'a>(&'a [bool]);

impl<'a> EntryMask<'a> {
    /// Wrap one flag per stored entry of `pattern`.
    pub fn new(pattern: &SparsityPattern, flags: &'a [bool]) -> Self {
        assert_eq!(
            flags.len(),
            pattern.nnz(),
            "an entry mask needs one flag per stored entry",
        );
        Self(flags)
    }

    #[inline]
    pub fn get(self, entry: usize) -> bool {
        self.0[entry]
    }

    /// The flags of one row, given its `indptr` span.
    #[inline]
    pub fn span(self, span: std::ops::Range<usize>) -> &'a [bool] {
        &self.0[span]
    }

    /// The flags in entry order, aligned with `pattern.indices`.
    #[inline]
    pub const fn as_slice(self) -> &'a [bool] {
        self.0
    }
}

impl<'a> ColumnMask<'a> {
    fn new(pattern: &SparsityPattern, flags: &'a [bool]) -> Self {
        assert_eq!(
            flags.len(),
            pattern.ncols(),
            "a column mask needs one flag per column",
        );
        Self(flags)
    }

    #[inline]
    fn get(self, col: usize) -> bool {
        self.0[col]
    }
}

/// Result of graph coloring, groups columns that can be computed together.
#[derive(Debug, Clone)]
pub struct ColumnColoring {
    colors: Vec<usize>,
    /// For each color, the columns assigned that color, in ascending order.
    color_to_columns: Vec<Vec<usize>>,
}

impl ColumnColoring {
    /// Color assigned to each column (0-indexed), or [`UNSEEDED`] for a column with no swept
    /// entry.
    #[inline]
    pub(crate) const fn colors(&self) -> &[usize] {
        self.colors.as_slice()
    }

    /// Total number of colors (= number of JVP calls needed).
    #[inline]
    pub const fn n_colors(&self) -> usize {
        self.color_to_columns.len()
    }

    /// Columns assigned a given color.
    ///
    /// Structurally-zero columns are colored too, so scatter results through a
    /// pattern-restricted entry table, never by writing all same-color columns.
    #[inline]
    pub(crate) fn columns_with_color(&self, color: usize) -> &[usize] {
        &self.color_to_columns[color]
    }

    /// Columns a sweep produces, i.e. those that received a color.
    #[inline]
    pub(crate) fn n_seeded_columns(&self) -> usize {
        self.color_to_columns.iter().map(Vec::len).sum()
    }
}

/// Column-adjacency graph in CSR form.
struct ColumnAdjacency {
    indptr: Vec<usize>,
    indices: Vec<usize>,
}

/// Column adjacency restricted to the entries a sweep must recover.
///
/// In a row, a swept column conflicts with every other seeded column; two unswept ones never do.
fn build_column_adjacency_masked(
    pattern: &SparsityPattern,
    swept: EntryMask<'_>,
    seeded: ColumnMask<'_>,
) -> ColumnAdjacency {
    let ncols = pattern.ncols();
    // Column-major view carrying each CSR entry index, so swept-ness is a lookup.
    let (col_start, col_entries) = transpose(ncols, pattern.indptr(), pattern.indices());
    let row_has_swept: Vec<bool> = (0..pattern.nrows())
        .map(|row| swept.span(pattern.row_span(row)).iter().any(|&s| s))
        .collect();

    // Dedup by per-source stamp: a sorted multiset would hold d * ncols^2 entries on d dense rows.
    let mut indptr = Vec::with_capacity(ncols + 1);
    let mut indices = Vec::new();
    indptr.push(0);
    let mut stamp = vec![usize::MAX; ncols];
    for col in 0..ncols {
        stamp[col] = col;
        let start = indices.len();
        for &(row, entry) in &col_entries[col_start[col]..col_start[col + 1]] {
            let col_swept = swept.get(entry);
            if !row_has_swept[row] || (!col_swept && !seeded.get(col)) {
                continue;
            }
            let span = pattern.row_span(row);
            for (&other, &other_swept) in
                pattern.indices()[span.clone()].iter().zip(swept.span(span))
            {
                let conflicts = if col_swept {
                    other_swept || seeded.get(other)
                } else {
                    other_swept
                };
                if conflicts && stamp[other] != col {
                    stamp[other] = col;
                    indices.push(other);
                }
            }
        }
        indices[start..].sort_unstable();
        indptr.push(indices.len());
    }
    ColumnAdjacency { indptr, indices }
}

/// Every entry swept and every column seeded, empty ones too: the unmasked colouring.
#[cfg(test)]
pub fn color_columns(pattern: &SparsityPattern) -> ColumnColoring {
    let all_entries = vec![true; pattern.nnz()];
    let all_columns = vec![true; pattern.ncols()];
    color_with_seeds(
        pattern,
        EntryMask::new(pattern, &all_entries),
        ColumnMask::new(pattern, &all_columns),
    )
}

/// DSATUR graph coloring of the Jacobian columns, told which entries a sweep must recover.
///
/// Ties break by higher degree, then lower index, so the result is deterministic. A column
/// with no swept entry gets [`UNSEEDED`] in `colors`.
pub fn color_columns_masked(pattern: &SparsityPattern, swept: EntryMask<'_>) -> ColumnColoring {
    let mut seeded = vec![false; pattern.ncols()];
    for (&col, &is_swept) in pattern.indices().iter().zip(swept.as_slice()) {
        if is_swept {
            seeded[col] = true;
        }
    }
    let seeded = ColumnMask::new(pattern, &seeded);
    color_with_seeds(pattern, swept, seeded)
}

/// Insert `value` into the ascending `set[..*len]`, growing `len`; false if already present.
fn insert_sorted(set: &mut [usize], len: &mut usize, value: usize) -> bool {
    let Err(pos) = set[..*len].binary_search(&value) else {
        return false;
    };
    set.copy_within(pos..*len, pos + 1);
    set[pos] = value;
    *len += 1;
    true
}

/// DSATUR over the seeded columns, with `swept` deciding which entries conflict.
fn color_with_seeds(
    pattern: &SparsityPattern,
    swept: EntryMask<'_>,
    seeded: ColumnMask<'_>,
) -> ColumnColoring {
    let ncols = pattern.ncols();
    if ncols == 0 {
        return ColumnColoring {
            colors: vec![],
            color_to_columns: vec![],
        };
    }

    let adj = build_column_adjacency_masked(pattern, swept, seeded);
    let degree = |v: usize| adj.indptr[v + 1] - adj.indptr[v];

    let mut colors = vec![UNSEEDED; ncols];
    // Distinct neighbour colours per uncolored column, ascending, in the column's own
    // adjacency span: the graph is symmetric, so a column has at most `degree` of them.
    let mut neighbour_colors = vec![0usize; adj.indices.len()];
    let mut saturation = vec![0usize; ncols];
    let mut forbidden = vec![false; ncols];
    let mut n_colors: usize = 0;

    // Max-heap on (saturation, degree, lowest index); a column's stale entries are skipped.
    let mut queue: BinaryHeap<(usize, usize, Reverse<usize>)> = (0..ncols)
        .filter(|&col| seeded.get(col))
        .map(|col| (0, degree(col), Reverse(col)))
        .collect();

    while let Some((queued_saturation, _, Reverse(col))) = queue.pop() {
        if colors[col] != UNSEEDED || queued_saturation != saturation[col] {
            continue;
        }

        let neighbours = &adj.indices[adj.indptr[col]..adj.indptr[col + 1]];

        for &nbr in neighbours {
            if colors[nbr] != UNSEEDED {
                forbidden[colors[nbr]] = true;
            }
        }
        let mut c = 0;
        while c < forbidden.len() && forbidden[c] {
            c += 1;
        }
        for &nbr in neighbours {
            if colors[nbr] != UNSEEDED {
                forbidden[colors[nbr]] = false;
            }
        }

        colors[col] = c;
        n_colors = n_colors.max(c + 1);

        for &nbr in neighbours {
            let seen = &mut neighbour_colors[adj.indptr[nbr]..adj.indptr[nbr + 1]];
            if colors[nbr] == UNSEEDED && insert_sorted(seen, &mut saturation[nbr], c) {
                queue.push((saturation[nbr], degree(nbr), Reverse(nbr)));
            }
        }
    }

    let mut color_to_columns: Vec<Vec<usize>> = vec![Vec::new(); n_colors];
    for (col, &c) in colors.iter().enumerate() {
        if c != UNSEEDED {
            color_to_columns[c].push(col);
        }
    }

    ColumnColoring {
        colors,
        color_to_columns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparsity::tests::pattern_from_rows;
    use std::collections::HashSet;

    fn make_dense_pattern(nrows: usize, ncols: usize) -> SparsityPattern {
        SparsityPattern::dense(nrows, ncols)
    }

    fn make_diagonal_pattern(n: usize) -> SparsityPattern {
        pattern_from_rows(n, &(0..n).map(|i| vec![i]).collect::<Vec<_>>())
    }

    fn make_tridiagonal_pattern(n: usize) -> SparsityPattern {
        let rows: Vec<Vec<usize>> = (0..n)
            .map(|row| (row.saturating_sub(1)..(row + 2).min(n)).collect())
            .collect();
        pattern_from_rows(n, &rows)
    }

    #[test]
    fn test_color_dense_matrix() {
        let pattern = make_dense_pattern(3, 4);
        let coloring = color_columns(&pattern);
        assert_eq!(coloring.n_colors(), 4);
        let unique_colors: HashSet<_> = coloring.colors.iter().copied().collect();
        assert_eq!(unique_colors.len(), 4);
    }

    #[test]
    fn test_color_diagonal_matrix() {
        let pattern = make_diagonal_pattern(5);
        let coloring = color_columns(&pattern);
        assert_eq!(coloring.n_colors(), 1);
        assert!(coloring.colors.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_color_tridiagonal_matrix() {
        let pattern = make_tridiagonal_pattern(6);
        let coloring = color_columns(&pattern);
        assert!(coloring.n_colors() <= 3);
        for col in 0..5 {
            assert_ne!(coloring.colors[col], coloring.colors[col + 1]);
        }
    }

    #[test]
    fn test_columns_with_color() {
        let pattern = make_diagonal_pattern(4);
        let coloring = color_columns(&pattern);
        let cols = coloring.columns_with_color(0);
        assert_eq!(cols, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_color_to_columns_matches_columns_with_color() {
        let pattern = make_tridiagonal_pattern(8);
        let coloring = color_columns(&pattern);
        for color in 0..coloring.n_colors() {
            let direct: Vec<usize> = coloring
                .colors
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| (c == color).then_some(i))
                .collect();
            assert_eq!(coloring.color_to_columns[color], direct);
        }
    }

    fn assert_valid_coloring(pattern: &SparsityPattern, coloring: &ColumnColoring) {
        assert_valid_masked_coloring(pattern, &vec![true; pattern.nnz()], coloring);
    }

    fn assert_valid_masked_coloring(
        pattern: &SparsityPattern,
        swept: &[bool],
        coloring: &ColumnColoring,
    ) {
        let seeded: Vec<bool> = (0..pattern.ncols())
            .map(|col| coloring.colors[col] != UNSEEDED)
            .collect();
        let adj = build_column_adjacency_masked(
            pattern,
            EntryMask::new(pattern, swept),
            ColumnMask::new(pattern, &seeded),
        );
        for (col, _) in seeded.iter().enumerate().filter(|&(_, &s)| s) {
            for &nbr in &adj.indices[adj.indptr[col]..adj.indptr[col + 1]] {
                assert_ne!(
                    coloring.colors[col], coloring.colors[nbr],
                    "adjacent columns {col} and {nbr} share a colour"
                );
            }
        }
    }

    /// One matrix row per edge, so the column graph is exactly `edges`.
    fn make_graph_from_edges(ncols: usize, edges: &[(usize, usize)]) -> SparsityPattern {
        let rows: Vec<Vec<usize>> = edges
            .iter()
            .map(|&(i, j)| {
                assert_ne!(i, j, "self-loops are not valid graph edges");
                vec![i.min(j), i.max(j)]
            })
            .collect();
        pattern_from_rows(ncols, &rows)
    }

    fn make_bidiagonal_pattern(n: usize) -> SparsityPattern {
        pattern_from_rows(
            n,
            &(0..n - 1).map(|row| vec![row, row + 1]).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn test_dsatur_reaches_optimum_on_path() {
        let pattern = make_bidiagonal_pattern(12);
        let coloring = color_columns(&pattern);
        assert_valid_coloring(&pattern, &coloring);
        assert_eq!(
            coloring.n_colors(),
            2,
            "DSATUR should 2-colour a path graph"
        );
    }

    #[test]
    fn test_dsatur_valid_on_dense_and_tridiagonal() {
        for pattern in [make_dense_pattern(5, 5), make_tridiagonal_pattern(20)] {
            let coloring = color_columns(&pattern);
            assert_valid_coloring(&pattern, &coloring);
        }
    }

    #[test]
    fn test_dsatur_matches_known_chromatic_number() {
        // (name, n_columns, edges, chromatic number)
        type ChromaticCase = (&'static str, usize, Vec<(usize, usize)>, usize);
        let cases: [ChromaticCase; 4] = [
            ("K3", 3, vec![(0, 1), (1, 2), (2, 0)], 3),
            // Odd cycle: chromatic number 3 though its largest clique is 2.
            ("C5", 5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 3),
            (
                "C6",
                6,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
                2,
            ),
            (
                "K4",
                4,
                vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                4,
            ),
        ];
        for (name, ncols, edges, chromatic) in &cases {
            let pattern = make_graph_from_edges(*ncols, edges);
            let coloring = color_columns(&pattern);
            assert_valid_coloring(&pattern, &coloring);
            assert_eq!(
                coloring.n_colors(),
                *chromatic,
                "{name}: DSATUR should use exactly {chromatic} colours, got {}",
                coloring.n_colors()
            );
        }
    }

    /// Brute-force restatement of the adjacency edge rule.
    fn reference_adjacency(
        pattern: &SparsityPattern,
        swept: &[bool],
        seeded: &[bool],
    ) -> Vec<Vec<usize>> {
        let mut sets = vec![std::collections::BTreeSet::new(); pattern.ncols()];
        for row in 0..pattern.nrows() {
            let (start, end) = (pattern.indptr()[row], pattern.indptr()[row + 1]);
            let cols = &pattern.indices()[start..end];
            let (swept_cols, other_cols): (Vec<usize>, Vec<usize>) = (
                cols.iter()
                    .zip(&swept[start..end])
                    .filter(|&(_, &s)| s)
                    .map(|(&c, _)| c)
                    .collect(),
                cols.iter()
                    .zip(&swept[start..end])
                    .filter(|&(&c, &s)| !s && seeded[c])
                    .map(|(&c, _)| c)
                    .collect(),
            );
            if swept_cols.is_empty() {
                continue;
            }
            for &col in &swept_cols {
                sets[col].extend(swept_cols.iter().filter(|&&o| o != col));
                sets[col].extend(other_cols.iter().filter(|&&o| o != col));
            }
            for &col in &other_cols {
                sets[col].extend(swept_cols.iter().filter(|&&o| o != col));
            }
        }
        sets.into_iter()
            .map(|set| set.into_iter().collect())
            .collect()
    }

    fn assert_matches_reference(pattern: &SparsityPattern, swept: &[bool], seeded: &[bool]) {
        let adj = build_column_adjacency_masked(
            pattern,
            EntryMask::new(pattern, swept),
            ColumnMask::new(pattern, seeded),
        );
        for (col, expected) in reference_adjacency(pattern, swept, seeded)
            .iter()
            .enumerate()
        {
            assert_eq!(
                &adj.indices[adj.indptr[col]..adj.indptr[col + 1]],
                expected.as_slice(),
                "column {col} neighbours differ from the reference rule"
            );
        }
    }

    #[test]
    fn column_adjacency_matches_the_reference_rule() {
        let dense = make_dense_pattern(6, 5);
        assert_matches_reference(&dense, &vec![true; dense.nnz()], &[true; 5]);

        let banded = make_tridiagonal_pattern(9);
        assert_matches_reference(&banded, &vec![true; banded.nnz()], &[true; 9]);

        let mut mask = vec![true; dense.nnz()];
        for (entry, flag) in mask.iter_mut().enumerate() {
            *flag = entry % 3 != 0;
        }
        let mut seeded = vec![false; 5];
        for (&col, &is_swept) in dense.indices().iter().zip(&mask) {
            seeded[col] |= is_swept;
        }
        assert_matches_reference(&dense, &mask, &seeded);
    }

    #[test]
    fn insert_sorted_keeps_a_set_ascending_and_counts_new_members() {
        let mut set = [0; 4];
        let mut len = 0;
        for (value, inserted) in [(5, true), (2, true), (5, false), (9, true), (3, true)] {
            assert_eq!(insert_sorted(&mut set, &mut len, value), inserted);
        }
        assert_eq!((len, set), (4, [2, 3, 5, 9]));
    }

    #[test]
    fn masked_coloring_ignores_wholly_constant_rows() {
        let pattern = make_dense_pattern(4, 4);
        let coloring = color_columns_masked(
            &pattern,
            EntryMask::new(&pattern, &vec![false; pattern.nnz()]),
        );
        assert_eq!(coloring.n_colors(), 0);
        assert!(coloring.colors.iter().all(|&c| c == UNSEEDED));
        assert_eq!(coloring.n_seeded_columns(), 0);
    }

    #[test]
    fn masked_coloring_keeps_swept_entries_unpolluted() {
        let pattern = make_dense_pattern(1, 5);
        let mut swept = vec![false; pattern.nnz()];
        swept[0] = true;
        let coloring = color_columns_masked(&pattern, EntryMask::new(&pattern, &swept));
        assert_valid_masked_coloring(&pattern, &swept, &coloring);
        assert_eq!(coloring.n_colors(), 1, "only one column is ever seeded");
        assert_eq!(coloring.colors[0], 0);
        assert!(coloring.colors[1..].iter().all(|&c| c == UNSEEDED));
    }

    #[test]
    fn masked_coloring_beats_the_full_rule_on_a_mixed_row() {
        // Column 1 is constant wherever it appears, so the masked rule leaves it unseeded.
        let pattern = SparsityPattern::try_new(2, 3, vec![0, 3, 4], vec![0, 1, 2, 0])
            .expect("a valid pattern");
        let swept = vec![true, false, true, true];
        assert_eq!(color_columns(&pattern).n_colors(), 3);
        let coloring = color_columns_masked(&pattern, EntryMask::new(&pattern, &swept));
        assert_valid_masked_coloring(&pattern, &swept, &coloring);
        assert_eq!(coloring.n_colors(), 2);
        assert_ne!(coloring.colors[0], coloring.colors[2]);
        assert_eq!(coloring.colors[1], UNSEEDED);
    }

    #[test]
    fn an_all_swept_mask_matches_the_unmasked_coloring() {
        for pattern in [make_dense_pattern(5, 5), make_tridiagonal_pattern(20)] {
            let full = color_columns(&pattern);
            let masked = color_columns_masked(
                &pattern,
                EntryMask::new(&pattern, &vec![true; pattern.nnz()]),
            );
            assert_eq!(full.colors, masked.colors);
        }
    }
}
