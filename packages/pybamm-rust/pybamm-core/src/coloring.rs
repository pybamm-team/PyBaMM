//! Column grouping that compresses a sparse Jacobian into few JVP passes.
//!
//! Two columns that never share a row can be evaluated together: seed the
//! tangent vector with 1 in both and one forward pass returns both columns'
//! entries, unmixed. Grouping the columns of a [`SparsityPattern`] that way
//! turns `ncols` JVP passes into `n_colors`, which for a banded `PyBaMM` Jacobian
//! is a handful regardless of mesh size.
//!
//! Only the pattern is consulted, never the values, so a coloring computed once
//! at compile time stays valid for every later evaluation.
//!
//! Entries a compile pass already knows are exempt: [`color_columns_masked`]
//! seeds only the columns a sweep must still recover, and lets two of them
//! share a color unless a row makes one of them read the other.

use crate::sparsity::SparsityPattern;

/// A column no sweep produces, so it never receives a color.
pub const UNSEEDED: usize = usize::MAX;

/// Result of graph coloring, groups columns that can be computed together.
#[derive(Debug, Clone)]
pub struct ColumnColoring {
    /// Color assigned to each column (0-indexed), or [`UNSEEDED`].
    pub colors: Vec<usize>,
    /// Total number of colors (= number of JVP calls needed).
    pub n_colors: usize,
    /// For each color, the columns assigned that color, in ascending order.
    pub color_to_columns: Vec<Vec<usize>>,
}

impl ColumnColoring {
    /// Columns assigned a given color (slice into the precomputed table).
    ///
    /// Every seeded column is colored, including structurally-zero ones, so
    /// consumers must scatter results via a pattern-restricted entry table
    /// (e.g. `color_to_csc_entries`), never by writing all same-color columns.
    #[inline]
    pub fn columns_with_color(&self, color: usize) -> &[usize] {
        &self.color_to_columns[color]
    }

    /// Columns a sweep produces, i.e. those that received a color.
    #[inline]
    pub fn n_seeded_columns(&self) -> usize {
        self.color_to_columns.iter().map(Vec::len).sum()
    }
}

/// Column-adjacency graph in CSR form.
pub(crate) struct ColumnAdjacency {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
}

/// Column adjacency restricted to the entries a sweep must recover.
///
/// `swept` is indexed by CSR entry (one flag per entry, always) and `seeded` by
/// column. An edge is emitted only where a swept entry would pick a seeded
/// column up: within a row, every swept column conflicts with every other
/// seeded column present, while two columns that are both merely constant
/// there never do.
fn build_column_adjacency_masked(
    pattern: &SparsityPattern,
    swept: &[bool],
    seeded: &[bool],
) -> ColumnAdjacency {
    build_column_adjacency_masked_counted(pattern, swept, seeded).0
}

/// [`build_column_adjacency_masked`], also returning how many neighbours were
/// appended.
///
/// A test can assert that equals the graph's edge count, which is what pins the
/// dedup to insertion time: buffering duplicates instead costs one append per
/// (row, column) pair, so a structurally dense row would push `ncols` per column.
fn build_column_adjacency_masked_counted(
    pattern: &SparsityPattern,
    swept: &[bool],
    seeded: &[bool],
) -> (ColumnAdjacency, usize) {
    let ncols = pattern.ncols;
    // Column-major view of the pattern, carrying the CSR entry index so the walk
    // below can ask whether this column is swept in that row without rescanning.
    let mut col_start = vec![0usize; ncols + 1];
    for &col in &pattern.indices {
        col_start[col + 1] += 1;
    }
    for col in 0..ncols {
        col_start[col + 1] += col_start[col];
    }
    let mut col_entries = vec![(0usize, 0usize); pattern.indices.len()];
    let mut fill = col_start.clone();
    for row in 0..pattern.nrows {
        for entry in pattern.indptr[row]..pattern.indptr[row + 1] {
            let col = pattern.indices[entry];
            col_entries[fill[col]] = (row, entry);
            fill[col] += 1;
        }
    }
    // A row with no swept entry conflicts nothing, so hoist the test out of the
    // per-column walk that would otherwise repeat it once per column in the row.
    let row_has_swept: Vec<bool> = (0..pattern.nrows)
        .map(|row| {
            swept[pattern.indptr[row]..pattern.indptr[row + 1]]
                .iter()
                .any(|&s| s)
        })
        .collect();

    // Dedup against a per-source stamp rather than by sorting a multiset: each
    // structurally dense row otherwise pushes `ncols` entries into every one of
    // its columns, so d dense rows cost d * ncols^2 live before the dedup.
    let mut sets: Vec<Vec<usize>> = vec![Vec::new(); ncols];
    let mut stamp = vec![usize::MAX; ncols];
    let mut appended = 0;
    for col in 0..ncols {
        stamp[col] = col;
        for &(row, entry) in &col_entries[col_start[col]..col_start[col + 1]] {
            let col_swept = swept[entry];
            if !row_has_swept[row] || (!col_swept && !seeded[col]) {
                continue;
            }
            let span = pattern.indptr[row]..pattern.indptr[row + 1];
            for (&other, &other_swept) in pattern.indices[span.clone()].iter().zip(&swept[span]) {
                let conflicts = if col_swept {
                    other_swept || seeded[other]
                } else {
                    other_swept
                };
                if conflicts && stamp[other] != col {
                    stamp[other] = col;
                    sets[col].push(other);
                    appended += 1;
                }
            }
        }
        sets[col].sort_unstable();
    }

    let mut indptr = Vec::with_capacity(ncols + 1);
    let mut indices = Vec::new();
    indptr.push(0);
    for set in &sets {
        indices.extend_from_slice(set);
        indptr.push(indices.len());
    }
    (ColumnAdjacency { indptr, indices }, appended)
}

/// DSATUR (maximum saturation degree) graph coloring for Jacobian column
/// grouping.
///
/// At each step the uncolored column with the most distinctly-colored
/// neighbours is coloured next (ties broken by higher graph degree, then lower
/// index for determinism) and given the smallest colour no neighbour uses.
pub fn color_columns(pattern: &SparsityPattern) -> ColumnColoring {
    // Every column is seeded here, structurally-empty ones included, which is
    // what consumers that index `colors` unconditionally rely on.
    color_with_seeds(
        pattern,
        &vec![true; pattern.nnz()],
        &vec![true; pattern.ncols],
    )
}

/// As [`color_columns`], but told which entries a sweep must recover.
///
/// `swept` carries one flag per CSR entry. Two columns conflict on a shared row
/// when at least one is swept there; a column with no swept entry produces
/// nothing and gets [`UNSEEDED`] in `colors`.
pub fn color_columns_masked(pattern: &SparsityPattern, swept: &[bool]) -> ColumnColoring {
    let mut seeded = vec![false; pattern.ncols];
    for (&col, &is_swept) in pattern.indices.iter().zip(swept) {
        if is_swept {
            seeded[col] = true;
        }
    }
    color_with_seeds(pattern, swept, &seeded)
}

/// DSATUR over the seeded columns, with `swept` deciding which entries create a
/// conflict. Shared by the masked and unmasked entry points.
fn color_with_seeds(pattern: &SparsityPattern, swept: &[bool], seeded: &[bool]) -> ColumnColoring {
    let ncols = pattern.ncols;
    if ncols == 0 {
        return ColumnColoring {
            colors: vec![],
            n_colors: 0,
            color_to_columns: vec![],
        };
    }

    // Selection scans this list, not every column, which is what keeps the
    // O(n^2) loop proportional to the columns a sweep actually produces.
    let seeded_cols: Vec<usize> = (0..ncols).filter(|&col| seeded[col]).collect();

    let adj = build_column_adjacency_masked(pattern, swept, seeded);
    let degree = |v: usize| adj.indptr[v + 1] - adj.indptr[v];

    let mut colors = vec![UNSEEDED; ncols];
    // Distinct neighbour colours seen by each still-uncolored column; its
    // length is the column's saturation degree.
    let mut neighbour_colors: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); ncols];
    let mut forbidden = vec![false; ncols];
    let mut n_colors: usize = 0;

    for _ in 0..seeded_cols.len() {
        // Select the uncolored column with maximum saturation degree.
        let mut best: Option<usize> = None;
        for &v in &seeded_cols {
            if colors[v] != UNSEEDED {
                continue;
            }
            best = Some(best.map_or(v, |b| {
                let (sv, sb) = (neighbour_colors[v].len(), neighbour_colors[b].len());
                if sv > sb || (sv == sb && degree(v) > degree(b)) {
                    v
                } else {
                    b
                }
            }));
        }
        let col = best.expect("an uncolored seeded column remains while iterating");

        let neighbours = &adj.indices[adj.indptr[col]..adj.indptr[col + 1]];

        // Smallest colour not used by any neighbour.
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

        // Propagate the new colour into uncolored neighbours' saturation sets.
        for &nbr in neighbours {
            if colors[nbr] == UNSEEDED {
                neighbour_colors[nbr].insert(c);
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
        n_colors,
        color_to_columns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_dense_pattern(nrows: usize, ncols: usize) -> SparsityPattern {
        let mut pattern = SparsityPattern::new(nrows, ncols);
        for row in 0..nrows {
            pattern.indptr[row] = row * ncols;
            for col in 0..ncols {
                pattern.indices.push(col);
            }
        }
        pattern.indptr[nrows] = nrows * ncols;
        pattern
    }

    fn make_diagonal_pattern(n: usize) -> SparsityPattern {
        let mut pattern = SparsityPattern::new(n, n);
        for i in 0..n {
            pattern.indptr[i] = i;
            pattern.indices.push(i);
        }
        pattern.indptr[n] = n;
        pattern
    }

    fn make_tridiagonal_pattern(n: usize) -> SparsityPattern {
        let mut pattern = SparsityPattern::new(n, n);
        let mut idx = 0;
        for row in 0..n {
            pattern.indptr[row] = idx;
            if row > 0 {
                pattern.indices.push(row - 1);
                idx += 1;
            }
            pattern.indices.push(row);
            idx += 1;
            if row < n - 1 {
                pattern.indices.push(row + 1);
                idx += 1;
            }
        }
        pattern.indptr[n] = idx;
        pattern
    }

    #[test]
    fn test_color_dense_matrix() {
        let pattern = make_dense_pattern(3, 4);
        let coloring = color_columns(&pattern);
        assert_eq!(coloring.n_colors, 4);
        let unique_colors: HashSet<_> = coloring.colors.iter().copied().collect();
        assert_eq!(unique_colors.len(), 4);
    }

    #[test]
    fn test_color_diagonal_matrix() {
        let pattern = make_diagonal_pattern(5);
        let coloring = color_columns(&pattern);
        assert_eq!(coloring.n_colors, 1);
        assert!(coloring.colors.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_color_tridiagonal_matrix() {
        let pattern = make_tridiagonal_pattern(6);
        let coloring = color_columns(&pattern);
        assert!(coloring.n_colors <= 3);
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
        for color in 0..coloring.n_colors {
            let direct: Vec<usize> = coloring
                .colors
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| (c == color).then_some(i))
                .collect();
            assert_eq!(coloring.color_to_columns[color], direct);
        }
    }

    /// Validity check that a colouring respects column-adjacency (distance-1).
    fn assert_valid_coloring(pattern: &SparsityPattern, coloring: &ColumnColoring) {
        assert_valid_masked_coloring(pattern, &vec![true; pattern.nnz()], coloring);
    }

    /// As [`assert_valid_coloring`], under the mask the colouring was given.
    fn assert_valid_masked_coloring(
        pattern: &SparsityPattern,
        swept: &[bool],
        coloring: &ColumnColoring,
    ) {
        let seeded: Vec<bool> = (0..pattern.ncols)
            .map(|col| coloring.colors[col] != UNSEEDED)
            .collect();
        let adj = build_column_adjacency_masked(pattern, swept, &seeded);
        for (col, _) in seeded.iter().enumerate().filter(|&(_, &s)| s) {
            for &nbr in &adj.indices[adj.indptr[col]..adj.indptr[col + 1]] {
                assert_ne!(
                    coloring.colors[col], coloring.colors[nbr],
                    "adjacent columns {col} and {nbr} share a colour"
                );
            }
        }
    }

    /// Build a sparsity pattern whose column-adjacency graph is exactly `edges`
    /// (one matrix row per edge).
    fn make_graph_from_edges(ncols: usize, edges: &[(usize, usize)]) -> SparsityPattern {
        let mut pattern = SparsityPattern::new(edges.len(), ncols);
        let mut idx = 0;
        for (row, &(i, j)) in edges.iter().enumerate() {
            assert_ne!(i, j, "self-loops are not valid graph edges");
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            pattern.indptr[row] = idx;
            pattern.indices.push(lo);
            pattern.indices.push(hi);
            idx += 2;
        }
        pattern.indptr[edges.len()] = idx;
        pattern
    }

    /// Bidiagonal pattern: row k holds columns {k, k+1}. The resulting column
    /// graph is a simple path 0-1-...-(n-1), which is bipartite.
    fn make_bidiagonal_pattern(n: usize) -> SparsityPattern {
        let mut pattern = SparsityPattern::new(n - 1, n);
        let mut idx = 0;
        for row in 0..n - 1 {
            pattern.indptr[row] = idx;
            pattern.indices.push(row);
            pattern.indices.push(row + 1);
            idx += 2;
        }
        pattern.indptr[n - 1] = idx;
        pattern
    }

    #[test]
    fn test_dsatur_reaches_optimum_on_path() {
        // A path column-graph is bipartite, so 2 colours is optimal; DSATUR also
        // hits the structural lower bound on the battery Jacobians.
        let pattern = make_bidiagonal_pattern(12);
        let coloring = color_columns(&pattern);
        assert_valid_coloring(&pattern, &coloring);
        assert_eq!(coloring.n_colors, 2, "DSATUR should 2-colour a path graph");
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
        // DSATUR must use exactly the proven chromatic number on each graph, built
        // as a column-adjacency graph with one matrix row per edge.

        // (name, n_columns, edges, expected chromatic number)
        type ChromaticCase = (&'static str, usize, Vec<(usize, usize)>, usize);
        let cases: [ChromaticCase; 4] = [
            // K3 triangle: a 3-clique forces χ ≥ 3, and 3 suffices.
            ("K3", 3, vec![(0, 1), (1, 2), (2, 0)], 3),
            // C5 odd cycle: not bipartite, so χ = 3 though its largest clique is 2,
            // a stronger bound than clique size, and the likeliest to mis-colour.
            ("C5", 5, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 3),
            // C6 even cycle: bipartite, so χ = 2.
            (
                "C6",
                6,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
                2,
            ),
            // K4 complete graph: every pair adjacent, so χ = 4.
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
                coloring.n_colors, *chromatic,
                "{name}: DSATUR should use exactly {chromatic} colours, got {}",
                coloring.n_colors
            );
        }
    }

    #[test]
    fn test_column_adjacency_for_dense_matrix() {
        let pattern = make_dense_pattern(4, 4);
        let adj = build_column_adjacency_masked(&pattern, &[true; 16], &[true; 4]);
        for col in 0..4 {
            let neighbours = &adj.indices[adj.indptr[col]..adj.indptr[col + 1]];
            let expected: Vec<usize> = (0..4).filter(|&c| c != col).collect();
            assert_eq!(neighbours, expected.as_slice());
        }
    }

    /// Brute-force restatement of the edge rule this module documents, used to
    /// pin the incremental build against a version with no dedup subtlety.
    fn reference_adjacency(
        pattern: &SparsityPattern,
        swept: &[bool],
        seeded: &[bool],
    ) -> Vec<Vec<usize>> {
        let mut sets = vec![std::collections::BTreeSet::new(); pattern.ncols];
        for row in 0..pattern.nrows {
            let (start, end) = (pattern.indptr[row], pattern.indptr[row + 1]);
            let cols = &pattern.indices[start..end];
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
        let adj = build_column_adjacency_masked(pattern, swept, seeded);
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
        // Dense, banded and mixed-mask shapes: the last is the one where a
        // column is present in a row without conflicting there.
        let dense = make_dense_pattern(6, 5);
        assert_matches_reference(&dense, &vec![true; dense.nnz()], &[true; 5]);

        let banded = make_tridiagonal_pattern(9);
        assert_matches_reference(&banded, &vec![true; banded.nnz()], &[true; 9]);

        let mut mask = vec![true; dense.nnz()];
        for (entry, flag) in mask.iter_mut().enumerate() {
            *flag = entry % 3 != 0;
        }
        let mut seeded = vec![false; 5];
        for (&col, &is_swept) in dense.indices.iter().zip(&mask) {
            seeded[col] |= is_swept;
        }
        assert_matches_reference(&dense, &mask, &seeded);
    }

    #[test]
    fn dense_rows_never_buffer_duplicate_neighbours() {
        // Each of these rows alone makes every column adjacent to every other,
        // so appending before deduping would cost `n_dense * ncols` per column
        // instead of the ncols - 1 edges that survive.
        let (ncols, n_dense) = (24, 8);
        let mut pattern = SparsityPattern::new(n_dense, ncols);
        for row in 0..n_dense {
            pattern.indices.extend(0..ncols);
            pattern.indptr[row + 1] = pattern.indices.len();
        }
        let (adj, appended) = build_column_adjacency_masked_counted(
            &pattern,
            &vec![true; pattern.nnz()],
            &vec![true; ncols],
        );
        assert_eq!(adj.indices.len(), ncols * (ncols - 1));
        assert_eq!(
            appended,
            adj.indices.len(),
            "every append must survive the dedup, not be sorted out afterwards"
        );
    }

    #[test]
    fn masked_coloring_ignores_wholly_constant_rows() {
        // A dense pattern whose every entry is known needs no sweep at all.
        let pattern = make_dense_pattern(4, 4);
        let coloring = color_columns_masked(&pattern, &vec![false; pattern.nnz()]);
        assert_eq!(coloring.n_colors, 0);
        assert!(coloring.colors.iter().all(|&c| c == UNSEEDED));
        assert_eq!(coloring.n_seeded_columns(), 0);
    }

    #[test]
    fn masked_coloring_keeps_swept_entries_unpolluted() {
        // Row 0 is dense but only column 0 varies there, so column 0 must not
        // share a colour with any of the constants it would otherwise read.
        let pattern = make_dense_pattern(1, 5);
        let mut swept = vec![false; pattern.nnz()];
        swept[0] = true;
        let coloring = color_columns_masked(&pattern, &swept);
        assert_valid_masked_coloring(&pattern, &swept, &coloring);
        assert_eq!(coloring.n_colors, 1, "only one column is ever seeded");
        assert_eq!(coloring.colors[0], 0);
        assert!(coloring.colors[1..].iter().all(|&c| c == UNSEEDED));
    }

    #[test]
    fn masked_coloring_beats_the_full_rule_on_a_mixed_row() {
        // Column 1 is constant wherever it appears, so it costs a colour under
        // the full rule and none here.
        let mut pattern = SparsityPattern::new(2, 3);
        pattern.indptr = vec![0, 3, 4];
        pattern.indices = vec![0, 1, 2, 0];
        let swept = vec![true, false, true, true];
        assert_eq!(color_columns(&pattern).n_colors, 3);
        let coloring = color_columns_masked(&pattern, &swept);
        assert_valid_masked_coloring(&pattern, &swept, &coloring);
        assert_eq!(coloring.n_colors, 2);
        assert_ne!(coloring.colors[0], coloring.colors[2]);
        assert_eq!(coloring.colors[1], UNSEEDED);
    }

    #[test]
    fn an_all_swept_mask_matches_the_unmasked_coloring() {
        for pattern in [make_dense_pattern(5, 5), make_tridiagonal_pattern(20)] {
            let full = color_columns(&pattern);
            let masked = color_columns_masked(&pattern, &vec![true; pattern.nnz()]);
            assert_eq!(full.colors, masked.colors);
        }
    }
}
