const std = @import("std");
const rl = @import("raylib");
const build_options = @import("build_options");

const assert = debug.assert;
const debug = std.debug;
const fmt = std.fmt;
const fs = std.fs;
const heap = std.heap;
const io = std.io;
const log = std.log;
const math = std.math;
const mem = std.mem;
const rand = std.rand;
const testing = std.testing;
const time = std.time;

const Allocator = mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const StaticBitSet = std.bit_set.StaticBitSet;
const DynamicBitSet = std.bit_set.DynamicBitSet;
const Random = std.rand.Random;

const Color = rl.Color;


// ================================ SOLVER STUFF ================================
// ==============================================================================

/// Terminology:
///
/// Cell: a square on the board where you place numbers.
/// Candidates: the numbers which are valid to place in a cell.
/// Line: a row or column on the board.
/// Neighbour: all the cells that share a row or column with a cell.
///
/// This solver basically consists of "depth first search + constraint
/// propagation". The "depth first search" part is implemented in `search` and
/// the "constraint propagation" part is `Line.remove_candidate_and_propagate`,
/// so those are both good places to start to see how the solver works.
pub const std_options: std.Options = .{
    .fmt_max_depth = 10,
    .log_level = .info,
};
const has_gui = build_options.mode == .gui;

const WALL: u8 = 10;
const max_candidates: u8 = 9;

/// A Kakuro game, partitioned into into a static (AuxData) and a dynamic (State) part.
const Kakuro = struct {
    state: State,
    aux_data: AuxData,

    fn deinit(self: *@This(), allocator: Allocator) void {
        self.state.deinit(allocator);
        self.aux_data.deinit(allocator);
    }
};

/// The _dynamic_ data for Kakuro board, the state of the board at a point in
/// time. This is the main data structure used in solving so it is essential
/// that the size is kept small.
///
/// See `AuxData` for static data.
const State = struct {
    /// The candidates for each cell on the board, indexed from top left to
    /// bottom right.
    candidates: []Candidates,

    /// Where the last move was made on the board, used during search as the
    /// starting point of propagation.
    move_index: ?u16 = null,

    /// Number of cells filled, used to quickly test for terminal states (i.e
    /// all cells are filled).
    n_filled: u16,

    nmoves: std.bit_set.DynamicBitSetUnmanaged,

    const Self = @This();

    fn place(self: *Self, index: usize, val: u8) void {
        assert(self.candidates[index].is_candidate(val));
        var cnds = &self.candidates[index];
        cnds.set_unique(val);
        cnds.set_filled();
        self.n_filled += 1;
    }

    fn get(self: Self, index: usize) u8 {
        return self.candidates[index].get_unique();
    }

    fn is_filled(self: Self, index: usize) bool {
        return self.candidates[index].is_filled();
    }

    fn is_terminal(self: Self, n_cells: usize) bool {
        return self.n_filled == n_cells;
    }

    fn is_solved(self: Self, precomputed_lines: PrecomputedLines) bool {
        for (precomputed_lines.lines) |line| {
            if (!line.is_solved(self)) return false;
        }
        return true;
    }

    fn init(allocator: Allocator, aux_data: AuxData) !Self {
        const candidates = blk: {
            var n_cells: usize = 0;
            for (aux_data.board) |c| {
                if (c != WALL) n_cells += 1;
            }
            const all_candidates = try allocator.alloc(Candidates, n_cells);
            @memset(all_candidates, .{ .cs = 0b111_111_111 });
            break :blk all_candidates;
        };

        var state = Self{
            .candidates = candidates,
            .n_filled = 0,
            .move_index = null,
            .nmoves = try std.bit_set.DynamicBitSetUnmanaged.initEmpty(allocator, aux_data.precomputed_lines.lines.len * 4),
        };

        // Apply upper and lower bounds on all cells as a pre-pass.
        for (aux_data.precomputed_lines.lines) |line| {
            const mask = line_constraints_mask(line.len, line.constraint);
            var i: u8 = 0;
            while (i < line.len) : (i += 1) {
                const index = line.indices[i];
                var cnds = &state.candidates[index];
                cnds.mask(mask);
            }
            var n_empty: usize = 0;
            for (line.constSlice()) |j| {
                if (!state.candidates[j].is_filled())
                    n_empty += 1;
            }
            switch (n_empty) {
                2 => state.nmoves.set(line.id * 4),
                3 => state.nmoves.set(line.id * 4 + 1),
                4 => state.nmoves.set(line.id * 4 + 2),
                5 => state.nmoves.set(line.id * 4 + 3),
                else => {},
            }
        }

        return state;
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.candidates);
        self.nmoves.deinit(allocator);
    }

    fn clone(self: Self, allocator: Allocator, move_index: ?u16) !State {
        return Self{
            .candidates = try allocator.dupe(Candidates, self.candidates),
            .n_filled = self.n_filled,
            .move_index = move_index,
            .nmoves = try self.nmoves.clone(allocator),
        };
    }
};

/// The _static_ data for Kakuro board, data that is the same at all points in
/// time for a particular Kakuro board. Kept separate from `State` to minimize
/// the data copied around during a search.
///
/// See `State` for dynamic data.
const AuxData = struct {
    board: []const u8,
    n_rows: usize,
    n_cols: usize,
    n_cells: usize,
    precomputed_lines: PrecomputedLines,

    fn deinit(self: *@This(), allocator: Allocator) void {
        self.precomputed_lines.deinit(allocator);
        allocator.free(self.board);
    }
};

const sum_1_to_9: u8 = 45;

inline fn lower_bound(n: u8, constraint: u8) u8 {
    const bound = sum_1_to_9 - ((11 - n) * (10 - n)) / 2;
    return if (bound >= constraint) 0 else constraint - bound;
}

inline fn upper_bound(val: u8, constraint: u8) u8 {
    const bound = (val * (val - 1)) / 2;
    return if (bound > constraint) 0 else constraint - bound;
}

/// The state of a cell on the board. The state is represented as a bitmap of
/// candidates:
///
/// Bits 1-9: set to 1 if the number at the position is a valid placement, 0 otherwise.
/// Bit 10: set to 1 if cell is filled, 0 otherwise.
///
/// A cell filled with the number `x` will have bit `x` and bit 9 set to 1, and
/// all other bits set to 0.
///
/// Even though this struct is quite small the solver spends roughly 30% of
/// its time copying arrays of candidates around, so anything to make this
/// struct smaller is worth considering.
const Candidates = struct {
    const Self = @This();
    const Int = u10;

    cs: Int,

    fn count(self: Self) u8 {
        return @popCount(self.cs & ~(@as(u10, 1) << 9));
    }

    fn is_candidate(self: Self, val: u8) bool {
        assert(val > 0 and val <= max_candidates);
        return ((self.cs >> @as(u4, @intCast(val - 1))) & 1) == 1;
    }

    fn get_unique(self: Self) u8 {
        assert(self.count() == 1);
        return @ctz(self.cs) + 1;
    }

    fn set_unique(self: *Self, val: u8) void {
        assert(self.is_candidate(val));
        self.cs = (@as(Int, 1) << @as(u4, @intCast(val - 1)));
    }

    fn set_filled(self: *Self) void {
        self.cs |= @as(Int, 1) << 9;
    }

    fn is_filled(self: Self) bool {
        return self.cs >> 9 == 1;
    }

    fn mask(self: *Self, m: Int) void {
        self.cs &= m;
    }

    fn print(self: Self) void {
        var i: u8 = 1;
        debug.print("[", .{});
        while (i <= max_candidates) : (i += 1) {
            const v: u8 = if (self.is_candidate(i)) 1 else 0;
            debug.print("{d} ", .{v});
        }
        debug.print("]\n", .{});
    }

    fn iterator(self: Self) Iterator {
        return Iterator{ .cs = self.cs, .total = 0 };
    }

    const Iterator = struct {
        cs: Candidates.Int,
        total: u8,

        const Iter = @This();

        fn next(self: *Iter) ?u8 {
            if (self.total >= max_candidates) return null;

            const first: u4 = @ctz(self.cs);
            self.total += first + 1;
            if (self.total > max_candidates) return null;
            self.cs >>= @as(u4, @intCast(first + 1));
            return self.total;
        }
    };
};

const PrecomputedLines = struct {
    lines: []Line,
    index_to_lines: []LinePair,

    const Self = @This();

    fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.lines);
        allocator.free(self.index_to_lines);
    }
};

/// Sets all bits above `n` to 0.
fn zero_above_inclusive(n: Candidates.Int) Candidates.Int {
    // TODO figure out how to remove this branch
    if (n >= max_candidates) return 0b111_111_111;
    const one = @as(Candidates.Int, 1);
    return 0b111_111_111 & ((one << @as(u4, @intCast(n))) -| 1);
}

/// Sets all bits below `n` to 0.
fn zero_below_inclusive(n: Candidates.Int) Candidates.Int {
    const all_ones = ~@as(Candidates.Int, 0);
    return 0b111_111_111 & math.shl(Candidates.Int, all_ones, n -| 1);
}

/// Computes a bitmask where the 0's are the candidates that are NOT valid
/// for the given line length and constraint.
fn line_constraints_mask(line_len: u8, line_constraint: u8) Candidates.Int {
    const ub = upper_bound(line_len, line_constraint);
    const lb = lower_bound(line_len, line_constraint);
    const ma = zero_above_inclusive(ub);
    const mb = zero_below_inclusive(lb);
    return 0b111_111_111 & ma & mb;
}

/// Same as `line_constraints_mask` but also removes a specific candidate.
fn candidates_mask(line_len: u8, line_constraint: u8, candidate_to_remove: u8) Candidates.Int {
    const start_mask = line_constraints_mask(line_len, line_constraint);
    const candidate_mask = ~@as(Candidates.Int, 0) ^ (@as(Candidates.Int, 1) << @as(u4, @intCast(candidate_to_remove - 1)));
    const out = start_mask & candidate_mask;
    // log.debug("candidates_mask >> len {d} constraint {d} val {d} ub {d} lb {d} ma {b:9} mb {b:9} val_mask {b:9} out {b:9}\n", .{line_len, line_constraint, val_to_remove, ub, lb, ma, mb, val_mask, out});
    return out;
}

// The current values of a line computed from `State`.
const LineState = struct {
    /// The cells that comprise this line -> indices into `State.candidates`.
    indices: [5]u16,
    line_id: u16,
    n_empty: u8,
    current_constraint: u8,
};

/// Extracts the current state of a line on the board.
fn computeLineState(state: *const State, line: Line) LineState {
    var sum: u8 = 0;
    var n_filled: u8 = 0;
    var empty_indices = std.BoundedArray(u16, 9).init(0) catch unreachable;
    for (line.constSlice()) |index| {
        const cnds = state.candidates[index];
        if (cnds.is_filled()) {
            const val = state.get(index);
            sum += @as(u8, @intCast(val));
            n_filled += 1;
        } else {
            empty_indices.append(index) catch unreachable;
        }
    }

    const n_empty = line.len - n_filled;
    var line_state = LineState{
        .indices = undefined,
        .line_id = line.id,
        .n_empty = n_empty,
        .current_constraint = line.constraint - sum,
    };
    // Store indices for exact line solving.
    if (n_empty >= 2 and n_empty <= 5) {
        std.mem.copyForwards(u16, &line_state.indices, empty_indices.constSlice());
    }
    return line_state;
}

/// Represents a row or column on the board. Only contains static data,
/// uses `indices` to fetch data from the current `State` during solving.
const Line = struct {
    id: u16,
    /// The cells that comprise this line -> indices into `State.candidates`.
    indices: [9]u16,
    len: u8,
    constraint: u8,

    const Self = @This();

    fn constSlice(self: *const Self) []const u16 {
        return self.indices[0..self.len];
    }

    fn is_solved(self: Self, state: State) bool {
        var i: u8 = 0;
        var sum: u8 = 0;
        var used = StaticBitSet(9).initEmpty();

        while (i < self.len) : (i += 1) {
            const index = self.indices[i];
            if (!state.is_filled(index)) return false;

            const val = state.get(index);
            if (used.isSet(val - 1)) return false;

            used.set(val - 1);
            sum += val;
        }
        return sum == self.constraint;
    }

    ///  After a number has been placed in a cell, this function figures out
    ///  which candidates are no longer valid in that line and applies it to
    ///  all neighbours. If any of those neighbours only have one possible
    ///  candidate after this step, the procedure repeats for that cell.
    fn remove_candidate_and_propagate(self: Self, state: *State, stack: *PropagateStack, candidate: u8, run_context: *RunContext, should_draw_propagation: bool) !bool {
        const line_state = computeLineState(state, self);
        const n_empty = line_state.n_empty;
        if (n_empty == 0) {
            return true;
        }

        // Note down which lines have 2-5 empty cells so we can solve them
        // exactly later.
        state.nmoves.setValue(self.id * 4, n_empty == 2);
        state.nmoves.setValue(self.id * 4 + 1, n_empty == 3);
        state.nmoves.setValue(self.id * 4 + 2, n_empty == 4);
        state.nmoves.setValue(self.id * 4 + 3, n_empty == 5);

        const current_constraint = line_state.current_constraint;
        const candidates_to_remove = candidates_mask(n_empty, current_constraint, candidate);
        var i: u8 = 0;
        while (i < self.len) : (i += 1) {
            const index = self.indices[i];
            // debug.print("i {d} index {d} cnds {b:10} filled {s}\n", .{i, index, state.candidates[index].cs, state.is_filled(index)});
            if (state.is_filled(index)) continue;

            var cnds = &state.candidates[index];
            if (has_gui and should_draw_propagation) {
                const old = cnds.*;
                cnds.mask(candidates_to_remove);
                if (old.cs != cnds.cs) {
                    try run_context.markPropagation(index, old, cnds.*);
                }
                std.time.sleep(run_context.sleep_time_ns / 3);
            } else {
                cnds.mask(candidates_to_remove);
            }
            switch (cnds.count()) {
                0 => return false,
                1 => try stack.append(index),
                else => {},
            }
        }

        return true;
    }
};

const LinePair = struct {
    row: Line,
    col: Line,
};

inline fn cell_index(n_cols: usize, row: usize, col: usize) usize {
    return (row * n_cols) + col;
}

fn compute_lines(
    allocator: Allocator,
    desc: Description,
    n_cells: usize,
) !PrecomputedLines {
    const n_rows = desc.n_rows;
    const n_cols = desc.n_cols;
    const board = desc.board;
    const row_constraints = desc.row_constraints;
    const col_constraints = desc.col_constraints;

    var index_to_lines_rows = try allocator.alloc(Line, n_cells);
    defer allocator.free(index_to_lines_rows);

    var index_to_lines_cols = try allocator.alloc(Line, n_cells);
    defer allocator.free(index_to_lines_cols);

    const board_size = board.len;

    const index_to_cell_index = try allocator.alloc(u16, board_size);
    {
        var cindex: u16 = 0;
        for (board, 0..) |c, i| {
            if (c != WALL) {
                index_to_cell_index[i] = cindex;
                cindex += 1;
            }
        }
    }

    const IndicesType = [9]u16;

    var i: usize = 0;
    var indices = mem.zeroes(IndicesType);
    var len: u8 = 0;

    // Compute all rows
    var lines = try ArrayList(Line).initCapacity(allocator, n_cells);
    while (i < n_rows) : (i += 1) {
        var j: usize = 0;
        while (j < n_cols) : (j += 1) {
            const index = cell_index(n_cols, i, j);

            // debug.print("ROW >>> c {d} i {d} j {d} index {d} len {d}\n", .{board[index], i, j, index, len});

            if (board[index] == WALL and len == 0)
                continue;

            if (board[index] != WALL) {
                indices[len] = index_to_cell_index[index];
                len += 1;
            }

            if ((board[index] == WALL or j == n_cols - 1) and len > 0) {
                const prev_index = cell_index(n_cols, i, j - 1);

                // Edge case: a length 1 row on the edge of the board.
                // In this case, we grab the row constraint from the current
                // cell rather than the previous cell.
                const constraint_index = if (board[prev_index] == WALL)
                    index
                else
                    prev_index;

                const row = Line{
                    .id = @as(u16, @intCast(lines.items.len)),
                    .indices = indices,
                    .len = len,
                    .constraint = row_constraints[constraint_index],
                };
                lines.appendAssumeCapacity(row);
                var k: u8 = 0;
                while (k < len) : (k += 1) {
                    index_to_lines_rows[indices[k]] = row;
                }
                indices = mem.zeroes(IndicesType);
                len = 0;
            }
        }
    }

    // debug.print("curr cell {d}\n", .{curr_cell});

    // Compute all cols
    i = 0;
    indices = mem.zeroes(IndicesType);
    len = 0;
    while (i < n_cols) : (i += 1) {
        var j: usize = 0;
        while (j < n_rows) : (j += 1) {
            const index = cell_index(n_cols, j, i);

            // debug.print("COL >>> c {d} i {d} j {d} index {d} len {d}\n", .{board[index], i, j, index, len});

            if (board[index] == WALL and len == 0)
                continue;

            if (board[index] != WALL) {
                indices[len] = index_to_cell_index[index];
                len += 1;
            }

            if ((board[index] == WALL or j == n_rows - 1) and len > 0) {
                const prev_index = cell_index(n_cols, j - 1, i);

                // Edge case: a length 1 col on the edge of the board.
                // In this case, we grab the col constraint from the current
                // cell rather than the previous cell.
                const constraint_index = if (board[prev_index] == WALL)
                    index
                else
                    prev_index;

                const col = Line{
                    .id = @as(u16, @intCast(lines.items.len)),
                    .indices = indices,
                    .len = len,
                    .constraint = col_constraints[constraint_index],
                };
                lines.appendAssumeCapacity(col);
                var k: u8 = 0;
                while (k < len) : (k += 1) {
                    index_to_lines_cols[indices[k]] = col;
                }
                indices = mem.zeroes(IndicesType);
                len = 0;
            }
        }
    }

    var index_to_lines = try allocator.alloc(LinePair, n_cells);
    {
        var j: usize = 0;
        while (j < n_cells) : (j += 1) {
            const row = index_to_lines_rows[j];
            const col = index_to_lines_cols[j];
            index_to_lines[j] = LinePair{
                .row = row,
                .col = col,
            };
        }
    }

    return PrecomputedLines{
        .lines = try lines.toOwnedSlice(),
        .index_to_lines = index_to_lines,
    };
}

/// A stack that only allows unique items. Attempting to add a number
/// already in the stack will silently do nothing.
const PropagateStack = struct {
    stack: ArrayList(usize),
    in_stack: DynamicBitSet,

    const Self = @This();

    fn append(self: *Self, val: usize) !void {
        if (self.in_stack.isSet(val)) return;

        try self.stack.append(val);
        self.in_stack.set(val);
    }

    fn pop(self: *Self) usize {
        const val = self.stack.pop();
        self.in_stack.unset(val);
        return val;
    }

    fn init(allocator: Allocator, size: usize) !Self {
        return Self{
            .stack = ArrayList(usize).init(allocator),
            .in_stack = try DynamicBitSet.initEmpty(allocator, size),
        };
    }

    fn deinit(self: *Self) void {
        self.stack.deinit();
        self.in_stack.deinit();
    }

    fn clear(self: *Self) void {
        self.stack.clearRetainingCapacity();

        // Copied from the source of DynamicBitSet since it lacks a
        // `clear` method.
        const bit_length = self.in_stack.unmanaged.bit_length;
        const num_masks = (bit_length + (@bitSizeOf(usize) - 1)) / @bitSizeOf(usize);
        for (self.in_stack.unmanaged.masks[0..num_masks]) |*mask| {
            mask.* = 0;
        }
    }
};

fn propagate(stack: *PropagateStack, state: *State, precomputed_lines: PrecomputedLines, run_context: *RunContext) !bool {
    defer stack.clear();

    if (state.move_index) |move_index| try stack.append(move_index);

    const should_draw_propagation = if (has_gui)
        @atomicLoad(u8, &run_context.should_draw_propagation, .seq_cst) == 1
    else
        false;

    while (stack.stack.items.len > 0) {
        const index = stack.pop();
        assert(!state.is_filled(index));

        const val = state.candidates[index].get_unique();
        state.place(index, val);

        const lines = precomputed_lines.index_to_lines[index];

        const row_consistent = try lines.row.remove_candidate_and_propagate(state, stack, val, run_context, should_draw_propagation);
        if (!row_consistent) return false;

        const col_consistent = try lines.col.remove_candidate_and_propagate(state, stack, val, run_context, should_draw_propagation);
        if (!col_consistent) return false;
    }

    return true;
}

const SearchOpts = struct {
    max_iters: usize,
};

const SearchResult = struct {
    allocator: Allocator,
    solution: ?*State,
    iters: usize,
    start_time: i64,
    end_time: i64,

    const Self = @This();

    fn deinit(self: *Self) void {
        if (self.solution) |s| {
            self.allocator.destroy(s);
        }
    }

    fn elapsed(self: Self) i64 {
        return self.end_time - self.start_time;
    }

    fn iters_per_second(self: Self) f64 {
        return @as(f64, @floatFromInt(self.iters)) / (@as(f64, @floatFromInt(self.elapsed())) / time.ns_per_s);
    }
};

const max_line_solutions: usize = 2;

const TwoMove = struct {
    index1: u16,
    val1: u8,
    index2: u16,
    val2: u8,
};

fn solve_line_exactly_two(
    state: *const State,
    line_state: LineState,
) !std.BoundedArray(TwoMove, max_line_solutions) {
    const indices = line_state.indices;
    assert(line_state.n_empty == 2);

    const cs1 = state.candidates[indices[0]];
    const cs2 = state.candidates[indices[1]];

    var n_solutions: usize = 0;
    var moves = std.BoundedArray(TwoMove, max_line_solutions).init(0) catch unreachable;

    var a: u8 = 1;
    while (a <= max_candidates) : (a += 1) {
        if (!cs1.is_candidate(a)) continue;
        const b = line_state.current_constraint - a;
        if (a == b) continue;
        if (!cs2.is_candidate(b)) continue;

        n_solutions += 1;
        if (n_solutions > max_line_solutions) {
            moves.len = 3;
            return moves;
        }

        try moves.append(.{
            .index1 = indices[0],
            .val1 = a,
            .index2 = indices[1],
            .val2 = b,
        });
    }

    return moves;
}

const ThreeMove = struct {
    index1: u16,
    val1: u8,
    index2: u16,
    val2: u8,
    index3: u16,
    val3: u8,
};

fn solve_line_exactly_three(
    state: *const State,
    line_state: LineState,
) !std.BoundedArray(ThreeMove, max_line_solutions) {
    const indices = line_state.indices;
    assert(line_state.n_empty == 3);

    const cs1 = state.candidates[indices[0]];
    const cs2 = state.candidates[indices[1]];
    const cs3 = state.candidates[indices[2]];
    var n_solutions: usize = 0;
    var moves = std.BoundedArray(ThreeMove, max_line_solutions).init(0) catch unreachable;

    var a: u8 = 1;
    while (a <= max_candidates) : (a += 1) {
        var b: u8 = 1;
        if (!cs1.is_candidate(a)) continue;
        const s = line_state.current_constraint - a;
        while (b <= @min(max_candidates, s)) : (b += 1) {
            if (!cs2.is_candidate(b)) continue;
            if (a == b) continue;
            const c = s - b;
            if (c == 0 or c > max_candidates) continue;
            if (!cs3.is_candidate(c)) continue;
            if (c == a or c == b) continue;
            assert(a + b + c == line_state.current_constraint);

            n_solutions += 1;
            if (n_solutions > max_line_solutions) {
                moves.len = 3;
                return moves;
            }

            try moves.append(.{
                .index1 = indices[0],
                .val1 = a,
                .index2 = indices[1],
                .val2 = b,
                .index3 = indices[2],
                .val3 = c,
            });
        }
    }

    return moves;
}

const FourMove = struct {
    index1: u16,
    val1: u8,
    index2: u16,
    val2: u8,
    index3: u16,
    val3: u8,
    index4: u16,
    val4: u8,
};

fn solve_line_exactly_four(
    state: *const State,
    line_state: LineState,
) !std.BoundedArray(FourMove, max_line_solutions) {
    const indices = line_state.indices;
    assert(line_state.n_empty == 4);

    const cs1 = state.candidates[indices[0]];
    const cs2 = state.candidates[indices[1]];
    const cs3 = state.candidates[indices[2]];
    const cs4 = state.candidates[indices[3]];

    var n_solutions: usize = 0;
    var moves = std.BoundedArray(FourMove, max_line_solutions).init(0) catch unreachable;

    var a: u8 = 1;
    while (a <= max_candidates) : (a += 1) {
        if (!cs1.is_candidate(a)) continue;
        var b: u8 = 1;
        const s = line_state.current_constraint - a;
        while (b <= @min(max_candidates, s)) : (b += 1) {
            if (!cs2.is_candidate(b)) continue;
            if (b == a) continue;
            const ss = s - b;
            var c: u8 = 1;
            while (c <= @min(max_candidates, ss)) : (c += 1) {
                if (!cs3.is_candidate(c)) continue;
                if (c == a or c == b) continue;
                const d = ss - c;
                if (d == 0 or d > max_candidates) continue;
                if (!cs4.is_candidate(d)) continue;
                if (d == a or d == b or d == c) continue;
                assert(a + b + c + d == line_state.current_constraint);

                n_solutions += 1;
                if (n_solutions > max_line_solutions) {
                    moves.len = 3;
                    return moves;
                }

                try moves.append(.{
                    .index1 = indices[0],
                    .val1 = a,
                    .index2 = indices[1],
                    .val2 = b,
                    .index3 = indices[2],
                    .val3 = c,
                    .index4 = indices[3],
                    .val4 = d,
                });
            }
        }
    }

    return moves;
}

const FiveMove = struct {
    index1: u16,
    val1: u8,
    index2: u16,
    val2: u8,
    index3: u16,
    val3: u8,
    index4: u16,
    val4: u8,
    index5: u16,
    val5: u8,
};

fn solve_line_exactly_five(
    state: *const State,
    line_state: LineState,
) !std.BoundedArray(FiveMove, max_line_solutions) {
    const indices = line_state.indices;
    assert(line_state.n_empty == 5);

    const cs1 = state.candidates[indices[0]];
    const cs2 = state.candidates[indices[1]];
    const cs3 = state.candidates[indices[2]];
    const cs4 = state.candidates[indices[3]];
    const cs5 = state.candidates[indices[4]];

    var n_solutions: usize = 0;
    var moves = std.BoundedArray(FiveMove, max_line_solutions).init(0) catch unreachable;

    var a: u8 = 1;
    while (a <= max_candidates) : (a += 1) {
        if (!cs1.is_candidate(a)) continue;
        var b: u8 = 1;
        const s = line_state.current_constraint - a;
        while (b <= @min(max_candidates, s)) : (b += 1) {
            if (!cs2.is_candidate(b)) continue;
            if (b == a) continue;
            const ss = s - b;
            var c: u8 = 1;
            while (c <= @min(max_candidates, ss)) : (c += 1) {
                if (!cs3.is_candidate(c)) continue;
                if (c == a or c == b) continue;
                const sss = ss - c;
                var d: u8 = 1;
                while (d <= @min(max_candidates, sss)) : (d += 1) {
                    if (!cs4.is_candidate(d)) continue;
                    if (d == a or d == b or d == c) continue;
                    const e = sss - d;
                    if (e == 0 or e > max_candidates) continue;
                    if (!cs5.is_candidate(e)) continue;
                    if (e == a or e == b or e == c or e == d) continue;
                    assert(a + b + c + d + e == line_state.current_constraint);

                    n_solutions += 1;
                    if (n_solutions > max_line_solutions) {
                        moves.len = 3;
                        return moves;
                    }

                    try moves.append(.{
                        .index1 = indices[0],
                        .val1 = a,
                        .index2 = indices[1],
                        .val2 = b,
                        .index3 = indices[2],
                        .val3 = c,
                        .index4 = indices[3],
                        .val4 = d,
                        .index5 = indices[4],
                        .val5 = e,
                    });
                }
            }
        }
    }

    return moves;
}

fn cloneAndMove(comptime T: type, allocator: Allocator, state: *State, move: T) !State {
    var clone = try state.clone(allocator, move.index1);
    switch (T) {
        TwoMove => {
            clone.candidates[move.index1].set_unique(move.val1);
            clone.candidates[move.index2].set_unique(move.val2);
        },
        ThreeMove => {
            clone.candidates[move.index1].set_unique(move.val1);
            clone.candidates[move.index2].set_unique(move.val2);
            clone.candidates[move.index3].set_unique(move.val3);
        },
        FourMove => {
            clone.candidates[move.index1].set_unique(move.val1);
            clone.candidates[move.index2].set_unique(move.val2);
            clone.candidates[move.index3].set_unique(move.val3);
            clone.candidates[move.index4].set_unique(move.val4);
        },
        FiveMove => {
            clone.candidates[move.index1].set_unique(move.val1);
            clone.candidates[move.index2].set_unique(move.val2);
            clone.candidates[move.index3].set_unique(move.val3);
            clone.candidates[move.index4].set_unique(move.val4);
            clone.candidates[move.index5].set_unique(move.val5);
        },
        else => @compileError("moveAndClone not implemented for type T"),
    }
    return clone;
}

fn search(allocator: Allocator, _stack: *ArrayList(State), state: State, aux_data: AuxData, opts: SearchOpts, run_context: *RunContext) !SearchResult {
    var stack = _stack.*;

    try stack.append(state);
    var rewind_clone: *State = undefined;
    if (has_gui) {
        rewind_clone = try allocator.create(State);
        rewind_clone.* = try state.clone(allocator, state.move_index);
        try run_context.rewinds.add(allocator, rewind_clone);
    }

    var iters: usize = 0;
    var current: State = undefined;
    var result = SearchResult{
        .allocator = allocator,
        .solution = null,
        .iters = undefined,
        .start_time = @as(i64, @intCast(time.nanoTimestamp())),
        .end_time = undefined,
    };

    if (has_gui) {
        @atomicStore(?*const State, &run_context.state, &state, .seq_cst);
    }

    var propagate_stack = try PropagateStack.init(allocator, aux_data.n_cells);
    defer propagate_stack.deinit();

    {
        var i: usize = 0;
        while (i < aux_data.n_cells) : (i += 1) {
            if (state.is_filled(i)) continue;

            if (state.candidates[i].count() == 1)
                propagate_stack.append(i) catch unreachable;
        }
    }

    const precomputed_lines = aux_data.precomputed_lines;
    search: while (stack.items.len > 0 and iters < opts.max_iters) {
        if (has_gui and !run_context.running.load(.seq_cst))
            break;
        if (false) break :search;
        // if (iters % 10 == 0)
        //     std.log.info("search stack {d} iters {d}\n", .{stack.items.len, iters});
        iters += 1;
        if (has_gui) {
            run_context.iters.store(iters, .seq_cst);
        }

        current = stack.pop();
        defer {
            if (has_gui) {
                if (run_context.paused == 1) {
                    run_context.pauseMutex.lock();
                    run_context.pauseCond.wait(&run_context.pauseMutex);
                    run_context.pauseMutex.unlock();
                }
                run_context.clearPropagations();
            }
            current.deinit(allocator);
        }

        if (has_gui) {
            std.time.sleep(run_context.sleep_time_ns / run_context.sleep_time_multiplier);
        }

        const consistent = try propagate(&propagate_stack, &current, precomputed_lines, run_context);

        if (has_gui) {
            rewind_clone = try allocator.create(State);
            rewind_clone.* = try current.clone(allocator, current.move_index);
            try run_context.rewinds.add(allocator, rewind_clone);
            @atomicStore(?*const State, &run_context.state, &current, .seq_cst);
            run_context.consistent.store(consistent, .seq_cst);
        }

        if (!consistent) continue;

        if (current.is_terminal(aux_data.n_cells)) {
            assert(current.is_solved(precomputed_lines));

            const solution_ptr = try allocator.create(State);
            solution_ptr.* = try current.clone(allocator, 0);
            result.solution = solution_ptr;
            result.iters = iters;
            result.end_time = @as(i64, @intCast(time.nanoTimestamp()));

            if (has_gui) {
                const draw_solution = try allocator.create(State);
                draw_solution.* = try current.clone(allocator, null);
                @atomicStore(?*const State, &run_context.state, draw_solution, .seq_cst);
            }

            return result;
        }

        // ================================= Exact line solving =================================
        //
        // Attempt to solve lines of with 2 to 5 empty cells exactly, looking for lines with either
        // 0, 1 or 2 solutions. A "solution" in this context is any valid set of moves that fills
        // all cells. If we encounter a line with more than 2 solutions we bail out early. So there
        // are three cases to consider:
        //
        // 0 solution: current board has no solutions, go next.
        // 1 solution: line has a forced move, make that move and continue.
        // 2 solution: line is a 50/50, if we have no better alternatives we'll take this since
        //             we at least have a 50% chance to get it right.
        //
        // Solving lines exactly is O(9^n), where `n` is the number of cells in the line, but it
        // turns out that solving (multiple) exponential subproblems is WAY faster than just doing
        // simple constraint propagation. I've determined experimentally that 5 empty is the limit
        // for this method, solving 6+ empty cell lines still yields reduced iteration counts,
        // but the all speed gains from reduced iterations are eaten up by the extra work needed to
        // solve the lines in the first place.
        //
        // We proceed as follows: we solve lines with 2, 3, 4 and 5 empty cells exactly and look
        // for lines with 0 or 1 solutions (these are dead ends and forced moves, respectively).
        // If we find no such lines, we then look at solutions with 2 moves in reverse order,
        // so 5, 4, 3, and 2. We look in this order because the more empty cells we can fill
        // the better.
        //
        // If none of this works, we fall back to a looking for the cell with the fewest candidates,
        // trying each move in turn.

        // 1. Solve lines of length 2 exactly.
        var two_move_one: ?TwoMove = null;
        var two_move_two: ?[max_line_solutions]TwoMove = null;
        for (0..precomputed_lines.lines.len) |index| {
            if (!current.nmoves.isSet(index * 4)) continue;
            // std.log.info("two move {d}", .{index});
            const line = precomputed_lines.lines[index];
            const line_state = computeLineState(&current, line);
            const moves = try solve_line_exactly_two(&current, line_state);

            switch (moves.len) {
                0 => continue :search, // current state is a dead end
                1 => {
                    two_move_one = moves.get(0);
                    break;
                },
                2 => if (two_move_two == null) {
                    two_move_two = moves.buffer;
                },
                else => {},
            }
        }

        // We found a length 2 line with 1 solution => forced move
        if (two_move_one) |move| {
            const clone = try cloneAndMove(TwoMove, allocator, &current, move);
            try stack.append(clone);
            continue :search;
        }

        // 2. Solve all lines of length 3 exactly.
        var three_move_one: ?ThreeMove = null;
        var three_move_two: ?[max_line_solutions]ThreeMove = null;
        for (0..precomputed_lines.lines.len) |index| {
            if (!current.nmoves.isSet(index * 4 + 1)) continue;
            // std.log.info("three move {d}", .{index});
            const line = precomputed_lines.lines[index];
            const line_state = computeLineState(&current, line);
            const moves = try solve_line_exactly_three(&current, line_state);

            switch (moves.len) {
                0 => continue :search, // current state is a dead end
                1 => {
                    three_move_one = moves.get(0);
                    break;
                },
                2 => if (three_move_two == null) {
                    three_move_two = moves.buffer;
                },
                else => {},
            }
        }

        // We found a length 3 line with 1 solution => forced move
        if (three_move_one) |move| {
            const clone = try cloneAndMove(ThreeMove, allocator, &current, move);
            try stack.append(clone);
            continue :search;
        }

        // 3. Solve all lines of length 4 exactly.
        var four_move_one: ?FourMove = null;
        var four_move_two: ?[max_line_solutions]FourMove = null;
        for (0..precomputed_lines.lines.len) |index| {
            if (!current.nmoves.isSet(index * 4 + 2)) continue;
            // std.log.info("four move {d}", .{index});
            const line = precomputed_lines.lines[index];
            const line_state = computeLineState(&current, line);
            const moves = try solve_line_exactly_four(&current, line_state);

            switch (moves.len) {
                0 => continue :search, // current state is a dead end
                1 => {
                    four_move_one = moves.get(0);
                    break;
                },
                2 => if (four_move_two == null) {
                    four_move_two = moves.buffer;
                },
                else => {},
            }
        }

        // We found a length 3 line with 1 solution => forced move
        if (four_move_one) |move| {
            // std.log.info("DID four MOVE OPT", .{});
            const clone = try cloneAndMove(FourMove, allocator, &current, move);
            try stack.append(clone);
            continue :search;
        }

        // 4. Solve all lines of length 5 exactly.
        var five_move_one: ?FiveMove = null;
        var five_move_two: ?[max_line_solutions]FiveMove = null;
        for (0..precomputed_lines.lines.len) |index| {
            if (!current.nmoves.isSet(index * 4 + 3)) continue;
            // std.log.info("five move {d}", .{index});
            const line = precomputed_lines.lines[index];
            const line_state = computeLineState(&current, line);
            const moves = try solve_line_exactly_five(&current, line_state);

            switch (moves.len) {
                0 => continue :search, // current state is a dead end
                1 => {
                    five_move_one = moves.get(0);
                    break;
                },
                2 => if (five_move_two == null) {
                    five_move_two = moves.buffer;
                },
                else => {},
            }
        }

        // We found a length 3 line with 1 solution => forced move
        if (five_move_one) |move| {
            const clone = try cloneAndMove(FiveMove, allocator, &current, move);
            try stack.append(clone);
            continue :search;
            // We found a length 5 line with a 50/50, add both moves and continue
        } else if (five_move_two) |moves| {
            for (moves) |move| {
                const clone = try cloneAndMove(FiveMove, allocator, &current, move);
                try stack.append(clone);
            }
            continue :search;
        }

        // We found a length 4 line with a 50/50, add both moves and continue
        if (four_move_two) |moves| {
            for (moves) |move| {
                const clone = try cloneAndMove(FourMove, allocator, &current, move);
                try stack.append(clone);
            }
            continue :search;
        }

        // We found a length 3 line with a 50/50, add both moves and continue
        if (three_move_two) |moves| {
            // std.log.info("DID THREE MOVE OPT", .{});
            for (moves) |move| {
                const clone = try cloneAndMove(ThreeMove, allocator, &current, move);
                try stack.append(clone);
            }
            continue :search;
        }

        // We found a length 2 line with a 50/50, add both moves and continue
        if (two_move_two) |moves| {
            // std.log.info("DID TWO MOVE OPT", .{});
            for (moves) |move| {
                const clone = try cloneAndMove(TwoMove, allocator, &current, move);
                try stack.append(clone);
            }
            continue :search;
        }

        // Exact line solving found nothing, fall back to cell with fewest candidates.
        var best_index: u16 = 0;
        var best_cell: Candidates = undefined;
        var best_count: u8 = 10;
        for (current.candidates, 0..) |cnds, i| {
            if (current.is_filled(i)) continue;
            const count = cnds.count();

            if (count > 1 and count < best_count) {
                best_index = @as(u16, @intCast(i));
                best_cell = cnds;
                best_count = count;
            }
        }

        var candidates = try std.BoundedArray(u8, 9).init(0);
        {
            var i: u8 = 1;
            while (i <= max_candidates) : (i += 1) {
                if (!best_cell.is_candidate(i)) continue;
                try candidates.append(i);
            }
        }

        for (candidates.slice()) |i| {
            var copy = try current.clone(allocator, best_index);
            copy.candidates[best_index].set_unique(i);
            try stack.append(copy);
        }
    }

    result.iters = iters;
    result.end_time = @as(i64, @intCast(time.nanoTimestamp()));
    return result;
}

const Description = struct {
    n_rows: usize,
    n_cols: usize,
    board: []const u8,
    row_constraints: []const u8,
    col_constraints: []const u8,
    solution: []const u8,
    state_index_to_board_index: []const usize,
};

fn parse_descriptions(allocator: Allocator, path: []const u8) ![]Description {
    const file = try fs.cwd().openFile(path, .{ .mode = .read_write });
    defer file.close();
    const stat = try file.stat();
    const contents = try file.reader().readAllAlloc(
        allocator,
        stat.size,
    );
    defer allocator.free(contents);

    var descriptions = ArrayList(Description).init(allocator);
    var rows = mem.split(u8, contents, "\n");

    // Remove comments
    while (rows.next()) |row| {
        if (row.len == 0 or row[0] != '#')
            break;
    }

    while (rows.index != null) {
        const desc = try parse_description(allocator, &rows);
        try descriptions.append(desc);
        _ = rows.next();
    }
    return descriptions.toOwnedSlice();
}

fn parse_description(allocator: Allocator, rows: *mem.SplitIterator(u8, .sequence)) !Description {
    var n_rows: usize = undefined;
    var n_cols: usize = undefined;
    {
        var n: usize = undefined;
        {
            const untrimmed_row = rows.next().?;
            const row = std.mem.trimRight(u8, untrimmed_row, &std.ascii.whitespace);
            var info = mem.split(u8, row, " ");
            var j: u8 = 0;
            while (info.next()) |c| {
                defer j += 1;
                switch (j) {
                    0 => n = try std.fmt.parseInt(usize, c, 10),
                    1 => n_rows = try std.fmt.parseInt(u8, c, 10),
                    2 => n_cols = try std.fmt.parseInt(u8, c, 10),
                    else => {},
                }
            }
            assert(j == 3);
        }
    }

    // debug.print("n {d} rows {d} cols {d}\n", .{n, n_rows, n_cols});

    const board_size = n_rows * n_cols;
    var board = try ArrayList(u8).initCapacity(allocator, board_size);
    var row_constraints = try ArrayList(u8).initCapacity(allocator, board_size);
    var col_constraints = try ArrayList(u8).initCapacity(allocator, board_size);
    var solution = try ArrayList(u8).initCapacity(allocator, board_size);
    defer solution.deinit();

    var section: usize = 0;
    var i: usize = 1;
    while (rows.next()) |untrimmed_row| {
        const row = std.mem.trimRight(u8, untrimmed_row, &std.ascii.whitespace);
        if (row.len == 0) continue;

        // std.log.info("i {d} section {d}\n", .{i, section});
        var list = switch (section) {
            0 => &board,
            1 => &row_constraints,
            2 => &col_constraints,
            3 => &solution,
            else => unreachable,
        };
        var inner_row = mem.split(u8, row, " ");
        while (inner_row.next()) |d| {
            const parsed = try std.fmt.parseInt(u8, d, 10);
            try list.append(parsed);
        }
        if (i % n_rows == 0) {
            section += 1;
            if (section > 3) break;
        }

        i += 1;
    }

    var state_index_to_board_index = std.ArrayList(usize).init(allocator);
    {
        for (board.items, 0..) |b, j| {
            if (b != 0) continue;
            try state_index_to_board_index.append(j);
        }
    }

    return Description{
        .n_rows = n_rows,
        .n_cols = n_cols,
        .board = try board.toOwnedSlice(),
        .row_constraints = try row_constraints.toOwnedSlice(),
        .col_constraints = try col_constraints.toOwnedSlice(),
        .solution = try solution.toOwnedSlice(),
        .state_index_to_board_index = try state_index_to_board_index.toOwnedSlice(),
    };
}

fn createKakuros(allocator: Allocator, descriptions: []const Description) ![]Kakuro {
    var kakuros = try allocator.alloc(Kakuro, descriptions.len);

    for (descriptions, 0..) |desc, i| {
        const n_cells = blk: {
            var n: usize = 0;
            for (desc.board) |c| {
                if (c != WALL) n += 1;
            }
            break :blk n;
        };

        const aux_data = AuxData{
            .board = desc.board,
            .n_rows = desc.n_rows,
            .n_cols = desc.n_cols,
            .n_cells = n_cells,
            .precomputed_lines = try compute_lines(
                allocator,
                desc,
                n_cells,
            ),
        };
        const state = try State.init(allocator, aux_data);
        kakuros[i] = Kakuro{
            .state = state,
            .aux_data = aux_data,
        };
    }

    return kakuros;
}

const Searcher = struct {
    const logger = std.log.scoped(.searcher);

    const Stats = struct {
        total_time: i64 = 0,
        total_iters: usize = 0,
    };

    const Self = @This();

    kakuro: Kakuro,
    stack: *ArrayList(State),
    max_iters_total: usize,
    max_iters_per_search: usize,
    max_retries: usize,
    run_context: *RunContext,

    fn do_search(self: *Self, allocator: Allocator) !?SearchResult {
        const root = self.kakuro.state;
        const aux_data = self.kakuro.aux_data;
        const stack = self.stack;

        var stats = Stats{};
        var max_iters = self.max_iters_per_search;
        while (true) {
            const opts = SearchOpts{
                .max_iters = max_iters,
            };

            var i: usize = 0;
            while (i < self.max_retries) : (i += 1) {
                var arena = ArenaAllocator.init(allocator);
                const arena_allocator = arena.allocator();
                // defer arena.deinit();

                const cloned = try root.clone(arena_allocator, null);
                var result = try search(arena_allocator, stack, cloned, aux_data, opts, self.run_context);
                self.stack.resize(0) catch unreachable;
                stats.total_iters += result.iters;
                stats.total_time += result.elapsed();

                if (result.solution) |solution| {
                    _ = solution;
                    logger.info("SOLVED >>> iters {d:^2} iters/s {d:^10.2} time {s}", .{ result.iters, result.iters_per_second(), fmt.fmtDurationSigned(result.elapsed()) });
                } else {
                    logger.info("{d}/{d} ### FAILED ### >>> iters {d} iters/s {d} time {s}", .{ i + 1, self.max_retries, result.iters, result.iters_per_second(), fmt.fmtDurationSigned(result.elapsed()) });
                }
                return result;
            } else {
                debug.print("TOTAL FAILURE\n", .{});
                const old_max_iters = max_iters;
                max_iters *= 2;
                if (max_iters > self.max_iters_total)
                    break;
                logger.info("max iters {d} -> {d}\n", .{ old_max_iters, max_iters });
                return null;
            }
        }
        unreachable;
    }
};

const Runner = struct {
    const logger = std.log.scoped(.runner);

    kakuros: []const Kakuro,
    allocator: Allocator,
    stack: *ArrayList(State),
    solution: ?State,
    run_context: *RunContext,
    search_results: []?SearchResult,

    const Self = @This();

    fn init(allocator: Allocator, kakuros: []Kakuro, run_context: *RunContext) !Self {
        const stack = try allocator.create(ArrayList(State));
        stack.* = ArrayList(State).init(allocator);
        return Self{
            .kakuros = kakuros,
            .allocator = allocator,
            .stack = stack,
            .solution = null,
            .run_context = run_context,
            .search_results = try allocator.alloc(?SearchResult, kakuros.len),
        };
    }

    fn runOne(self: *Self, index: usize) !void {
        logger.info("#{d} Running...", .{index});
        var searcher = create_searcher(self, index - 1);
        const search_result = try searcher.do_search(self.allocator);
        if (search_result) |sr| {
            if (sr.solution) |s| {
                self.solution = s.*;
            }
        }
        self.search_results[index - 1] = search_result;
    }

    fn runRange(self: *Self, start: usize, end: usize) !void {
        var index: usize = start;
        while (index < end) : (index += 1) {
            try self.runOne(index);
        }
    }

    fn runAll(self: *Self) !void {
        try self.runRange(1, self.kakuros.len + 1);
    }

    fn create_searcher(self: *Self, index: usize) Searcher {
        return Searcher{
            .kakuro = self.kakuros[index],
            .stack = self.stack,
            .max_iters_total = 15_000_000,
            .max_iters_per_search = 9_000_000,
            .max_retries = 50,
            .run_context = self.run_context,
        };
    }

    fn report(self: Self) !void {
        var w = std.io.getStdOut().writer();
        try w.print("index,iters,time_ns\n", .{});
        for (self.search_results, 0..) |result, i| {
            if (result) |r| {
                try w.print("{d},{d},{d}\n", .{ i + 1, r.iters, r.elapsed() });
            } else {
                try w.print("{d},{d},{d}\n", .{ i + 1, -1, -1 });
            }
        }
    }
};

fn display_board(state: State, aux_data: AuxData) void {
    const board = aux_data.board;
    const candidates = state.candidates;
    var cindex: usize = 0;
    for (board, 0..) |c, i| {
        if (c == WALL) {
            debug.print("  -", .{});
        } else {
            if (candidates[cindex].is_filled()) {
                debug.print("{d:3}", .{candidates[cindex].get_unique()});
            } else {
                debug.print("  X", .{});
            }
            cindex += 1;
        }
        if ((i + 1) % aux_data.n_cols == 0)
            debug.print("\n", .{});
    }
}

fn display_candidates(state: State, aux_data: AuxData) void {
    const buf_size: u8 = 9 * 2 + 2;
    const board = aux_data.board;
    var candidate_strs = testing.allocator.alloc([buf_size]u8, board.len) catch unreachable;
    defer testing.allocator.free(candidate_strs);

    var max_len: usize = 0;

    var cindex: usize = 0;
    var i: usize = 0;
    while (i < board.len) : (i += 1) {
        var store = [_]u8{' '} ** buf_size;
        var buf = io.fixedBufferStream(&store);
        var w = buf.writer();

        if (board[i] == WALL) {
            w.print("X", .{}) catch unreachable;
        } else {
            const cnds = state.candidates[cindex];
            w.writeByte('[') catch unreachable;
            {
                var j: u8 = 0;
                var n_written: u8 = 1;
                while (j < max_candidates) : (j += 1) {
                    if (cnds.is_candidate(j + 1)) {
                        w.print("{d}", .{j + 1}) catch unreachable;
                        if (n_written < cnds.count())
                            w.writeByte(' ') catch unreachable;
                        n_written += 1;
                    }
                }
            }
            w.writeByte(']') catch unreachable;
            cindex += 1;
        }

        const len = buf.getWritten().len;
        if (len > max_len)
            max_len = len;

        candidate_strs[i] = store;
    }

    var j: usize = 1;
    for (candidate_strs) |str| {
        debug.print("{s} ", .{str[0..max_len]});
        if (j % aux_data.n_cols == 0)
            debug.print("\n", .{});
        j += 1;
    }
}

fn printInfo(comptime T: type) void {
    @compileLog(T, @sizeOf(T), @alignOf(T));
}

fn printSizes() void {
    printInfo(Candidates);
    printInfo(State);
    printInfo(Line);
    printInfo(LinePair);
}

// ================================== GUI STUFF =================================
// ==============================================================================

// The first rewind is the current state,
// so this gives us current + 10 rewinds.
const max_rewinds: usize = 11;

const Queue = struct {
    len: usize,
    head: ?*Node,
    tail: ?*Node,

    const Self = @This();

    fn init() Self {
        return .{
            .len = 0,
            .head = null,
            .tail = null,
        };
    }

    fn add(self: *Self, allocator: Allocator, value: *State) !void {
        const tail = try allocator.create(Node);
        tail.* = .{
            .prev = self.tail,
            .next = null,
            .value = value,
        };

        if (self.tail) |t| {
            t.next = tail;
        }
        self.tail = tail;
        self.len = @min(self.len + 1, max_rewinds);
        if (self.len == 1) {
            self.head = tail;
        }
        if (self.len > max_rewinds) {
            var head = self.head.?;
            self.head = head.next;
            self.head.?.prev = null;
            head.deinit(allocator);
        }
    }

    fn get(self: Self, index: usize) ?*State {
        if (index > self.len - 1) return null;

        var i: usize = 0;
        var node = self.tail;
        while (i < index) : (i += 1) {
            if (node) |n| {
                node = n.prev;
            }
        }
        return node.?.value;
    }
};

const Node = struct {
    prev: ?*Node,
    next: ?*Node,
    value: *State,

    const Self = @This();

    fn deinit(self: *Self, allocator: Allocator) void {
        self.value.deinit(allocator);
        allocator.destroy(self.value);
        allocator.destroy(self);
    }
};

const Propagation = struct {
    index: usize,
    old: Candidates,
    new: Candidates,
};

const RunContext = if (build_options.mode == .gui) struct {
    running: std.atomic.Value(bool),
    rewinds: Queue,
    rewind_index: usize,
    state: ?*const State,
    consistent: std.atomic.Value(bool),
    iters: std.atomic.Value(usize),
    sleep_time_multiplier: usize,
    paused: u8,
    pauseMutex: std.Thread.Mutex,
    pauseCond: std.Thread.Condition,
    propagations: std.AutoArrayHashMap(usize, Propagation),
    propagations_mutex: std.Thread.Mutex,
    should_draw_propagation: u8,
    sleep_time_ns: usize,

    const Self = @This();

    fn markPropagation(self: *Self, index: usize, old: Candidates, new: Candidates) !void {
        self.propagations_mutex.lock();
        defer self.propagations_mutex.unlock();

        try self.propagations.put(index, .{
            .index = index,
            .old = old,
            .new = new,
        });
    }

    fn getPropagations(self: *Self, allocator: Allocator) !std.AutoArrayHashMap(usize, Propagation) {
        _ = allocator;
        self.propagations_mutex.lock();
        defer self.propagations_mutex.unlock();

        return try self.propagations.clone();
    }

    fn clearPropagations(self: *Self) void {
        self.propagations_mutex.lock();
        defer self.propagations_mutex.unlock();

        self.propagations.clearRetainingCapacity();
    }
} else void;

const cellSize: usize = 55;

const SolutionDrawMode = enum {
    none,
    solution,
    diff,
};

const screenWidth = 1200;
const screenHeight = 950;

fn numberKeyWasPressed() ?rl.KeyboardKey {
    const keys = [_]rl.KeyboardKey{
        .key_zero,
        .key_one,
        .key_two,
        .key_three,
        .key_four,
        .key_five,
        .key_six,
        .key_seven,
        .key_eight,
        .key_nine,
    };

    for (keys) |k| {
        if (rl.isKeyPressed(k))
            return k;
    }
    return null;
}

fn createRunContext(
    allocator: Allocator,
    kakuros: []Kakuro,
    drawIndex: usize,
    run_context: *RunContext,
    runner: *Runner,
    tid: *std.Thread,
    should_reset_camera: *bool,
) !void {
    run_context.* = RunContext{
        .running = std.atomic.Value(bool).init(true),
        .rewinds = Queue.init(),
        .rewind_index = 0,
        .consistent = std.atomic.Value(bool).init(true),
        .state = null,
        .iters = std.atomic.Value(usize).init(0),
        .sleep_time_multiplier = 1,
        .paused = 0,
        .pauseMutex = .{},
        .pauseCond = .{},
        .propagations = std.AutoArrayHashMap(usize, Propagation).init(allocator),
        .propagations_mutex = .{},
        .should_draw_propagation = 0,
        .sleep_time_ns = 500_000_000,
    };
    runner.* = try Runner.init(allocator, kakuros, run_context);
    tid.* = try std.Thread.spawn(.{}, Runner.runOne, .{ runner, drawIndex });
    should_reset_camera.* = true;
}

fn recreateRunContext(
    allocator: Allocator,
    kakuros: []Kakuro,
    drawIndex: usize,
    run_context: *RunContext,
    runner: *Runner,
    tid: *std.Thread,
    should_reset_camera: *bool,
) !void {
    @atomicStore(u8, &run_context.paused, 0, .seq_cst);
    run_context.pauseCond.signal();
    run_context.running.store(false, .seq_cst);
    std.Thread.join(tid.*);

    try createRunContext(allocator, kakuros, drawIndex, run_context, runner, tid, should_reset_camera);
}

// TODO: replace haphazard thread safety with something that actually works
fn runGui(allocator: Allocator, descriptions: []const Description) !void {
    // Initialization
    //--------------------------------------------------------------------------------------

    rl.initWindow(screenWidth, screenHeight, "zig-kakuro");
    rl.setTargetFPS(60);

    const useCamera = true;
    var drawIndex: usize = 1;
    var shouldDrawCandidates = false;
    var shouldDrawCandidateIndexes = false;
    var shouldDrawIndexOverlay = false;
    var shouldDrawHelpOverlay = false;
    var shouldDrawDebugOverlay = true;
    var indexOverlayInputStartTime: f64 = 0;
    var keyBuffer = std.BoundedArray(u8, 3).init(0) catch unreachable;
    var solutionDrawMode: SolutionDrawMode = .none;

    const kakuros = try createKakuros(allocator, descriptions);
    const iters = try allocator.create(usize);
    iters.* = 0;
    var run_context: RunContext = undefined;
    var runner: Runner = undefined;
    var tid: std.Thread = undefined;
    var should_reset_camera = false;
    try createRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);

    var camera = rl.Camera2D{
        .target = .{ .x = 0, .y = 0 },
        .offset = .{ .x = 0, .y = 0 },
        .rotation = 0.0,
        .zoom = 0.8,
    };

    var prev_mouse_position = rl.getMousePosition();

    while (!rl.windowShouldClose()) {
        should_reset_camera = false;

        if (rl.isKeyPressed(.key_m)) {
            std.log.info("PRESSED 'M'", .{});
            shouldDrawHelpOverlay = !shouldDrawHelpOverlay;
        }

        if (rl.isKeyPressed(.key_d)) {
            std.log.info("PRESSED 'D'", .{});
            shouldDrawDebugOverlay = !shouldDrawDebugOverlay;
        }

        if (rl.isKeyPressed(.key_z)) {
            std.log.info("PRESSED 'Z'", .{});
            should_reset_camera = true;
        }

        if (rl.isKeyPressed(.key_u)) {
            if (drawIndex > 0) drawIndex -= 1;
            std.log.info("PRESSED 'U', index {d}", .{drawIndex});
            try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
        }

        if (rl.isKeyPressed(.key_i)) {
            if (drawIndex < descriptions.len) drawIndex += 1;
            std.log.info("PRESSED 'I', index {d}", .{drawIndex});

            try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
        }

        if (rl.isKeyPressed(.key_h)) {
            run_context.sleep_time_multiplier = @max(run_context.sleep_time_multiplier / 2, 1);
            std.log.info("PRESSED 'H', speed {d}", .{run_context.sleep_time_multiplier});
        }

        if (rl.isKeyPressed(.key_l)) {
            run_context.sleep_time_multiplier *= 2;
            std.log.info("PRESSED 'L', speed {d}", .{run_context.sleep_time_multiplier});
        }

        if (rl.isKeyPressed(.key_c)) {
            shouldDrawCandidates = !shouldDrawCandidates;
            std.log.info("PRESSED 'C', should draw candidates {}", .{shouldDrawCandidates});
        }

        if (rl.isKeyPressed(.key_x)) {
            shouldDrawCandidateIndexes = !shouldDrawCandidateIndexes;
            std.log.info("PRESSED 'X', should draw candidate indexes {}", .{shouldDrawCandidateIndexes});
        }

        if (rl.isKeyPressed(.key_o)) {
            _ = @atomicRmw(u8, &run_context.should_draw_propagation, .Xor, 1, .seq_cst);
            std.log.info("PRESSED 'O', should draw propagation {}", .{run_context.should_draw_propagation});
        }

        if (rl.isKeyPressed(.key_s)) {
            solutionDrawMode = switch (solutionDrawMode) {
                .none => .solution,
                .solution => .diff,
                .diff => .none,
            };
            std.log.info("PRESSED 'S', should draw solution {}", .{solutionDrawMode});
        }

        if (rl.isKeyPressed(.key_p)) {
            std.log.info("PRESSED 'P', paused {d}", .{run_context.paused});
            run_context.rewind_index = 0;
            const old = @atomicRmw(u8, &run_context.paused, .Xor, 1, .seq_cst);
            if (old == 1) {
                run_context.pauseCond.signal();
            }
        }

        if (rl.isKeyPressed(.key_r)) {
            std.log.info("PRESSED 'R'", .{});
            try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
        }

        if (rl.isKeyPressed(.key_left)) {
            std.log.info("PRESSED 'LEFT'", .{});
            const rewind_index = @min(
                @min(run_context.rewind_index + 1, max_rewinds - 1),
                run_context.iters.load(.seq_cst),
            );
            run_context.rewind_index = rewind_index;
            run_context.state = run_context.rewinds.get(run_context.rewind_index);
        }

        if (rl.isKeyPressed(.key_right)) {
            std.log.info("PRESSED 'RIGHT'", .{});
            run_context.rewind_index -|= 1;
            run_context.state = run_context.rewinds.get(run_context.rewind_index);
        }

        if (numberKeyWasPressed()) |key| {
            const number = @as(u8, @intCast(@intFromEnum(key)));
            std.log.info("PRESSED '{d}'", .{number});
            if (!shouldDrawIndexOverlay) {
                shouldDrawIndexOverlay = true;
                indexOverlayInputStartTime = rl.getTime();
            }
            try keyBuffer.append(number);
        }

        if (shouldDrawIndexOverlay and (keyBuffer.len == 3 or rl.getTime() - indexOverlayInputStartTime > 0.7)) {
            defer {
                keyBuffer = std.BoundedArray(u8, 3).init(0) catch unreachable;
                shouldDrawIndexOverlay = false;
            }

            const index = std.fmt.parseInt(usize, keyBuffer.constSlice(), 10) catch unreachable;
            if (index > 0 and index <= kakuros.len) {
                drawIndex = index;
                try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
                should_reset_camera = true;
            }
        }

        const mouse_position = rl.getMousePosition();
        defer prev_mouse_position = mouse_position;

        camera.zoom += rl.getMouseWheelMove() * 0.05;
        if (rl.isMouseButtonDown(rl.MouseButton.mouse_button_left)) {
            const delta_x = mouse_position.x - prev_mouse_position.x;
            const delta_y = mouse_position.y - prev_mouse_position.y;
            camera.target = rl.Vector2{
                .x = camera.target.x - delta_x,
                .y = camera.target.y - delta_y,
            };
        }

        // Draw
        //----------------------------------------------------------------------------------
        const desc = descriptions[drawIndex - 1];
        if (should_reset_camera) {
            resetCamera(desc, &camera);
        }

        rl.beginDrawing();
        rl.clearBackground(Color.white);
        if (useCamera)
            rl.beginMode2D(camera);

        const board = desc.board;
        const row_constraints = desc.row_constraints;
        const col_constraints = desc.col_constraints;
        const n_cols = desc.n_cols;
        const n_rows = desc.n_rows;

        for (board, 0..) |s, index| {
            // std.log.info("index {d} cell {d}", .{index, s});
            const row = index / n_cols;
            const col = index % n_cols;
            if (s == 0) {
                drawEmptyCell(col + 1, row + 1);
            } else {
                drawOutOfBounds(col + 1, row + 1);
            }
        }

        var i: usize = 0;
        while (i < n_cols) : (i += 1) {
            drawOutOfBounds(i, 0);
        }

        i = 0;
        while (i < n_rows) : (i += 1) {
            drawOutOfBounds(0, i);
        }

        var prev: usize = 0;
        for (row_constraints, 0..) |constraint, index| {
            const row = index / n_cols;
            const col = index % n_cols;
            if (prev != constraint) {
                if (constraint != 0) {
                    const colConstraint = col_constraints[index];
                    drawNumberedBox(col, row + 1, constraint, colConstraint != 0);
                }
            }
            prev = constraint;
        }

        prev = 0;
        var c: usize = 0;
        while (c < n_cols) : (c += 1) {
            var r: usize = 0;
            while (r < n_rows) : (r += 1) {
                const index = r * n_cols + c;
                const constraint = col_constraints[index];
                if (prev != constraint) {
                    if (constraint != 0) {
                        const rowConstraint = row_constraints[index];
                        const shouldFill = r == 0 or rowConstraint == 0;
                        drawNumberedBoxInvert(c + 1, r, constraint, shouldFill);
                    }
                }
                prev = constraint;
            }
        }

        switch (solutionDrawMode) {
            .none => if (run_context.state) |state| {
                drawState(desc, state);
                if (state.move_index) |index| {
                    const board_index = desc.state_index_to_board_index[index];
                    const row = board_index / desc.n_cols;
                    const col = board_index % desc.n_cols;
                    drawCell(col + 1, row + 1, rl.fade(Color.blue, 0.5));
                }
            },
            .solution => drawSolution(desc),
            .diff => if (run_context.state) |state| {
                drawState(desc, state);
                drawSolutionDiff(desc, state);
                if (state.move_index) |index| {
                    const board_index = desc.state_index_to_board_index[index];
                    const row = board_index / desc.n_cols;
                    const col = board_index % desc.n_cols;
                    drawCell(col + 1, row + 1, rl.fade(Color.blue, 0.5));
                }
            },
        }

        var propagations = try runner.run_context.getPropagations(allocator);
        defer propagations.deinit();

        if (shouldDrawCandidates) {
            if (run_context.state) |state| {
                drawCandidates(desc, state, propagations);
            }
        }

        if (shouldDrawCandidateIndexes) {
            if (run_context.state) |state| {
                drawCandidateIndexes(desc, state);
            }
        }

        drawPropagations(desc, propagations);

        if (useCamera)
            rl.endMode2D();

        if (shouldDrawDebugOverlay) {
            drawDebugOverlay(allocator, &run_context, drawIndex, solutionDrawMode, shouldDrawCandidates, kakuros.len);
        }

        if (shouldDrawIndexOverlay) {
            drawIndexOverlay(keyBuffer.constSlice());
        }

        if (shouldDrawHelpOverlay) {
            drawHelpOverlay();
        } else {
            const help = "Press 'M' for help";
            const font_size = 44;
            rl.drawText(help[0..], screenWidth - computeTextWidth(help[0..], font_size) - 20, screenHeight - 70, font_size, Color.red);
        }

        rl.endDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    rl.closeWindow(); // Close window and OpenGL context
    //--------------------------------------------------------------------------------------
}

fn resetCamera(desc: Description, camera: *rl.Camera2D) void {
    const cell_size_float = @as(f32, @floatFromInt(cellSize));
    const board_width = @as(f32, @floatFromInt(desc.n_cols)) * cell_size_float;
    const board_height = @as(f32, @floatFromInt(desc.n_rows)) * cell_size_float;
    const width_scale_factor = screenWidth / board_width;
    const height_scale_factor = screenHeight / board_height;
    camera.zoom = @min(@min(width_scale_factor, height_scale_factor), 1.0);
    camera.target = .{ .x = 0, .y = 0 };
    camera.offset = .{ .x = 0, .y = 0 };
}

fn drawPropagations(desc: Description, propagations: std.AutoArrayHashMap(usize, Propagation)) void {
    for (propagations.keys()) |p| {
        const board_index = desc.state_index_to_board_index[p];
        const row = board_index / desc.n_cols;
        const col = board_index % desc.n_cols;
        drawCell(col + 1, row + 1, rl.fade(Color.yellow, 0.3));
    }
}

// TODO: do this without allocation
fn drawDebugOverlay(
    allocator: Allocator,
    run_context: *const RunContext,
    drawIndex: usize,
    solutionDrawMode: SolutionDrawMode,
    shouldDrawCandidates: bool,
    n_kakuros: usize,
) void {
    const mouse_position = rl.getMousePosition();
    const Printer = struct {
        allocator: Allocator,
        const Self = @This();
        fn printLine(self: Self, comptime str: []const u8, args: anytype) [:0]const u8 {
            return std.fmt.allocPrintZ(self.allocator, str, args) catch unreachable;
        }
    };
    const printer = Printer{ .allocator = allocator };
    const strs = [_][:0]const u8{
        printer.printLine("index: {d}/{d}", .{ drawIndex, n_kakuros }),
        printer.printLine("iters: {d}", .{run_context.iters.load(.seq_cst) - run_context.rewind_index}),
        printer.printLine("speed: {d}x", .{run_context.sleep_time_multiplier}),
        printer.printLine("rewind: {d}", .{run_context.rewind_index}),
        printer.printLine("solution: {s}", .{@tagName(solutionDrawMode)}),
        printer.printLine("candidates: {}", .{shouldDrawCandidates}),
        printer.printLine("consistent: {}", .{run_context.consistent.load(.seq_cst)}),
        printer.printLine("paused: {}", .{run_context.paused != 0}),
        printer.printLine("x {d} y {d}", .{ @as(isize, @intFromFloat(mouse_position.x)), @as(isize, @intFromFloat(mouse_position.y)) }),
    };

    const font_size = 36;
    const offset = font_size + 4;
    const padding = 20;
    const width = 300 + padding * 2;
    const x_pos = screenWidth - width;
    const rec = rl.Rectangle{
        .x = @as(f32, @floatFromInt(x_pos)),
        .y = 0,
        .width = @as(f32, @floatFromInt(width)),
        .height = @as(f32, @floatFromInt(offset * strs.len + 5)),
    };
    rl.drawRectangleRec(rec, rl.fade(Color.black, 0.8));

    var y_pos: c_int = 0;
    for (strs) |str| {
        rl.drawText(str, x_pos + padding, y_pos, font_size, Color.red);
        y_pos += offset;
        allocator.free(str);
    }
}

fn drawHelpOverlay() void {
    const rec = rl.Rectangle{
        .x = 0,
        .y = 0,
        .width = screenWidth,
        .height = screenHeight,
    };
    rl.drawRectangleRec(rec, rl.fade(Color.black, 0.8));

    const fontSize: u32 = 44;
    const padding: c_int = 20;
    var y_pos: c_int = padding;

    const strs = [_][:0]const u8{
        "M - Show this menu",
        "[NUMBER] - Goto Kakuro",
        "P - Pause",
        "R - Reset",
        "I - Solve next board",
        "U - Solve previous board",
        "L - Increase solver speed",
        "H - Decrease solver speed",
        "O - Show propagation",
        "C - Show/hide candidates",
        "X - Show/hide indices",
        "S - Show/hide/diff solution",
        "D - Show/hide debug overlay",
        "Z - Reset camera",
        "LEFT - Rewind",
        "RIGHT - Go forward",
    };

    for (strs) |str| {
        rl.drawText(str, padding, y_pos, fontSize, Color.white);
        y_pos += fontSize + padding / 2;
    }
}

fn drawIndexOverlay(asciiNumbers: []const u8) void {
    const rec = rl.Rectangle{
        .x = 0,
        .y = 0,
        .width = screenWidth,
        .height = screenHeight,
    };
    rl.drawRectangleRec(rec, rl.fade(Color.black, 0.8));

    var buf: [30:0]u8 = undefined;
    const str = std.fmt.bufPrintZ(&buf, "{s}", .{asciiNumbers}) catch unreachable;
    const fontSize: u32 = 188;
    const textSize = computeTextWidth(str, fontSize);
    rl.drawText(str, (screenWidth / 2) - @divFloor(textSize, 2), @divFloor(screenHeight, 2) - (fontSize - 2), fontSize, Color.white);
}

fn computeTextWidth(text: [:0]const u8, fontSize: c_int) c_int {
    return rl.measureText(text, fontSize);
}

fn drawState(desc: Description, state: *const State) void {
    var stateIndex: usize = 0;
    for (desc.board, 0..) |s, i| {
        if (s != 0) continue;

        const candidate = state.candidates[stateIndex];
        if (candidate.count() == 1) {
            const row = i / desc.n_cols;
            const col = i % desc.n_cols;
            const val = candidate.get_unique();
            drawNumber(col + 1, row + 1, val);
        }
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawSolution(desc: Description) void {
    for (desc.solution, 0..) |s, i| {
        if (s == 0) continue;

        const row = i / desc.n_cols;
        const col = i % desc.n_cols;
        drawNumber(col + 1, row + 1, s);
    }
}

fn drawSolutionDiff(desc: Description, state: *const State) void {
    var stateIndex: usize = 0;
    for (desc.solution, 0..) |s, i| {
        if (s == 0) continue;

        const candidate = state.candidates[stateIndex];
        if (candidate.count() == 1) {
            const row = i / desc.n_cols;
            const col = i % desc.n_cols;
            const val = candidate.get_unique();
            const rec = rectangleForPoint(col + 1, row + 1);
            if (val == s) {
                rl.drawRectangleRec(rec, rl.fade(Color.green, 0.3));
            } else {
                rl.drawRectangleRec(rec, rl.fade(Color.red, 0.3));
            }
        }
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawCandidates(desc: Description, state: *const State, propagations: ?std.AutoArrayHashMap(usize, Propagation)) void {
    var stateIndex: usize = 0;
    for (desc.board, 0..) |s, i| {
        if (s != 0) continue;

        const candidate = state.candidates[stateIndex];
        if (candidate.count() == 1) {
            stateIndex += 1;
            if (stateIndex == state.candidates.len) break;
            continue;
        }
        const row = i / desc.n_cols;
        const col = i % desc.n_cols;
        const old = blk: {
            if (propagations) |ps| {
                break :blk if (ps.get(stateIndex)) |p| p.old else null;
            } else break :blk null;
        };
        drawCandidatesForCell(col + 1, row + 1, candidate, old);
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawCandidatesForCell(x: usize, y: usize, candidates: Candidates, old: ?Candidates) void {
    const fontSize = 20;
    const unit = cellSize / 3;
    var i: u8 = 1;
    const fx = @as(f32, @floatFromInt(x * cellSize));
    const fy = @as(f32, @floatFromInt(y * cellSize));
    const rec = rl.Rectangle{ .x = fx, .y = fy, .width = cellSize, .height = cellSize };
    const count = candidates.count();
    if (count == 0) {
        rl.drawRectangleRec(rec, rl.fade(Color.red, 0.3));
    }
    var buf: [2:0]u8 = undefined;
    while (i <= max_candidates) : (i += 1) {
        if (candidates.is_candidate(i)) {
            const text = std.fmt.bufPrintZ(&buf, "{d}", .{i}) catch unreachable;
            const nudgeX = ((i - 1) % 3) * unit + 2;
            const nudgeY = ((i - 1) / 3) * unit + 1;
            const innerX = @as(c_int, @intCast(x * cellSize + nudgeX));
            const innerY = @as(c_int, @intCast(y * cellSize + nudgeY));
            rl.drawText(text, innerX, innerY, fontSize, Color.gray);
        } else if (!candidates.is_candidate(i) and old != null and old.?.is_candidate(i)) {
            const text = std.fmt.bufPrintZ(&buf, "{d}", .{i}) catch unreachable;
            const nudgeX = ((i - 1) % 3) * unit + 2;
            const nudgeY = ((i - 1) / 3) * unit + 1;
            const innerX = @as(c_int, @intCast(x * cellSize + nudgeX));
            const innerY = @as(c_int, @intCast(y * cellSize + nudgeY));
            rl.drawText(text, innerX, innerY, fontSize, Color.gray);
            const innerFx = @as(f32, @floatFromInt(x * cellSize + nudgeX));
            const innerFy = @as(f32, @floatFromInt(y * cellSize + nudgeY));
            rl.drawLineEx(
                rl.Vector2{ .x = innerFx, .y = innerFy },
                rl.Vector2{ .x = innerFx + 10, .y = innerFy + 16 },
                3,
                Color.red,
            );
        }
    }
}

fn drawCandidateIndexes(desc: Description, state: *const State) void {
    var stateIndex: usize = 0;
    const unit = cellSize / 3;
    for (desc.board, 0..) |s, i| {
        if (s != 0) continue;

        const x = i % desc.n_cols + 1;
        const y = i / desc.n_cols + 1;

        var buf: [30:0]u8 = undefined;
        const text = std.fmt.bufPrintZ(&buf, "{d}", .{stateIndex}) catch unreachable;
        const nudgeX = 2 * unit - 5;
        const nudgeY = 1;
        const innerX = @as(c_int, @intCast(x * cellSize + nudgeX));
        const innerY = @as(c_int, @intCast(y * cellSize + nudgeY));

        rl.drawText(text, innerX, innerY, 12, Color.gray);
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawNumber(x: usize, y: usize, number: usize) void {
    const fontSize = 40;
    var buf: [2:0]u8 = undefined;
    const text = std.fmt.bufPrintZ(&buf, "{d}", .{number}) catch unreachable;
    const textSize = rl.measureTextEx(rl.getFontDefault(), text, fontSize, 0);
    const paddingX = @divFloor(@as(c_int, @intCast(cellSize)) - @divFloor(@as(c_int, @intFromFloat(textSize.x)), 2), 2);
    const paddingY = @divFloor(@as(c_int, @intCast(cellSize)) - @divFloor(@as(c_int, @intFromFloat(textSize.y)), 2), 4);
    rl.drawText(text, @as(c_int, @intCast(x * cellSize)) + paddingX, @as(c_int, @intCast(y * cellSize)) + paddingY, fontSize, Color.black);
}

fn drawEmptyCell(x: usize, y: usize) void {
    drawCell(x, y, Color.white);
}

fn drawOutOfBounds(x: usize, y: usize) void {
    drawCell(x, y, Color.black);
}

fn drawCell(x: usize, y: usize, color: rl.Color) void {
    const rec = rectangleForPoint(x, y);
    rl.drawRectangleRec(rec, color);
    rl.drawRectangleLinesEx(rec, 1, Color.black);
}

fn rectangleForPoint(x: usize, y: usize) rl.Rectangle {
    return rl.Rectangle{
        .x = @as(f32, @floatFromInt(x * cellSize)),
        .y = @as(f32, @floatFromInt(y * cellSize)),
        .width = cellSize,
        .height = cellSize,
    };
}

fn drawNumberedBox(x: usize, y: usize, number: usize, fillLower: bool) void {
    const max = @as(f32, @floatFromInt(cellSize));
    const posX = @as(f32, @floatFromInt(x * cellSize));
    const posY = @as(f32, @floatFromInt(y * cellSize));
    rl.drawTriangle(
        .{ .x = max + posX, .y = max + posY },
        .{ .x = max + posX, .y = 0 + posY },
        .{ .x = 0 + posX, .y = 0 + posY },
        Color.white,
    );

    // const nudge = 5;
    const centerX = @as(c_int, @intFromFloat((max + max + 0 + posX * 3) / 3));
    const centerY = @as(c_int, @intFromFloat((max + 0 + 0 + posY * 3) / 3));
    var buf: [30:0]u8 = undefined;
    const text = std.fmt.bufPrintZ(&buf, "{d}", .{number}) catch unreachable;
    rl.drawText(text, centerX - 5, centerY - 13, 20, Color.black);

    if (fillLower) {
        rl.drawTriangle(
            .{ .x = max + posX, .y = max + posY },
            .{ .x = 0 + posX, .y = 0 + posY },
            .{ .x = 0 + posX, .y = max + posY },
            Color.black,
        );
    }

    const rec = rectangleForPoint(x, y);
    rl.drawRectangleLinesEx(rec, 1, Color.black);
}

fn drawNumberedBoxInvert(x: usize, y: usize, number: usize, fillUpper: bool) void {
    const max = @as(f32, @floatFromInt(cellSize));
    const posX = @as(f32, @floatFromInt(x * cellSize));
    const posY = @as(f32, @floatFromInt(y * cellSize));
    if (fillUpper) {
        rl.drawTriangle(
            .{ .x = max + posX, .y = max + posY },
            .{ .x = max + posX, .y = 0 + posY },
            .{ .x = 0 + posX, .y = 0 + posY },
            Color.black,
        );
    } else {
        rl.drawLineEx(
            rl.Vector2{ .x = posX, .y = posY },
            rl.Vector2{ .x = posX + max, .y = posY + max },
            3,
            Color.black,
        );
    }

    rl.drawTriangle(
        .{ .x = max + posX, .y = max + posY },
        .{ .x = 0 + posX, .y = 0 + posY },
        .{ .x = 0 + posX, .y = max + posY },
        Color.white,
    );

    const centerY = @as(c_int, @intFromFloat((max + max + 0 + posY * 3) / 3));
    var buf: [30:0]u8 = undefined;
    const text = std.fmt.bufPrintZ(&buf, "{d}", .{number}) catch unreachable;
    rl.drawText(text, @as(c_int, @intCast(x * cellSize)) + 5, centerY - 5, 20, Color.black);

    const rec = rectangleForPoint(x, y);
    rl.drawRectangleLinesEx(rec, 1, Color.black);
}

// ==================================== MAIN ====================================
// ==============================================================================
//

fn solve(allocator: Allocator, descriptions: []const Description, report: bool) !void {
    const kakuros = try createKakuros(allocator, descriptions);
    var run_context = {};
    var runner = try Runner.init(allocator, kakuros, &run_context);
    try runner.runAll();
    if (report)
        try runner.report();
}

pub fn main() !void {
    std.log.info("mode {s}", .{@tagName(build_options.mode)});

    const argv = std.os.argv;
    var boards_path: []const u8 = "boards.txt";
    var report = false;
    if (argv.len > 1) {
        if (std.mem.eql(u8, std.mem.span(argv[1]), "--boards")) {
            boards_path = std.mem.span(argv[2]);
        }
        else if (std.mem.eql(u8, std.mem.span(argv[1]), "--report")) {
            report = true;
        }
    }

    const allocator = std.heap.c_allocator;
    const descriptions = try parse_descriptions(allocator, boards_path);

    switch (build_options.mode) {
        .gui => try runGui(allocator, descriptions),
        .solve => try solve(allocator, descriptions, report),
    }
}
