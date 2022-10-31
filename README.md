# zig-kakuro

<p align="center">
  <img width="600" height="600" src="solve.gif">
</p>


A [Kakuro](https://en.wikipedia.org/wiki/Kakuro) solver written in Zig with a GUI for interactive solving.

For now, the solver uses an intentionally simple algorithm and focuses on an efficient implementation.

The repo bundles 950 Kakuro puzzles from [Otto and Angela Janko's puzzle site](https://www.janko.at/Raetsel/Kakuro/), an excellent resource for logic puzzles of all kinds. Currently, the zig-kakuro can solve 941 of 950 puzzles, or 99%. Of course, the goal is to reach 100% 😎

## Building

1. Download Zig 0.9.1 for your platform [here](https://ziglang.org/download/) (but see (#platform-support) below).
1. Clone this repo to your local machine.
1. In the repo folder, run `git submodule update --init --recursive`
1. Run zig-kakuro in one of two ways:
  - Run `zig build gui` to run the GUI solver
  - Run `zig build solve` to run the solver on all the included Kakuros.

### Platform support

|                | Solver  | GUI |
|----------------|---------|-----|
| macOS x86      | ✅      | ✅  |
| macOS aarch64  | ✅      | ❌  |
| Windows 10 x86 | ✅      | 🐛  |
| Linux          | ❓      | ❓  |

If you run into any bugs or build issues, please [open an issue](https://github.com/schmee/zig-kakuro/issues/new).

The GUI issues are all caused by [the incomplete stage 1 Zig C ABI](https://github.com/ziglang/zig/issues/1481), and will hopefully go away once raylib-zig and zig-kakuro upgrades to Zig 0.10.

## Acknowledgements

- https://www.janko.at/: for the puzzles bundled in `boards.txt`. An amazing website for logic puzzles of all kinds. The puzzles are licensed under Creative Commons 3.0, and are used for non-commercial purpose with attribution, see https://www.janko.at/Raetsel/Creative-Commons.htm for more information.
- [raylib](https://www.raylib.com/): used to build the GUI.
- [raylib-zig](https://github.com/Not-Nik/raylib-zig): Zig bindings for Raylib, makes it a breeze to get started with Zig + Raylib!

## License

zig-kakuro is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
