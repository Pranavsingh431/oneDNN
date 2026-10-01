/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_X64_IR_DUMP_HPP
#define CPU_X64_IR_DUMP_HPP

// Developer-only debug output of the IR pipeline.
//
// A kernel built from the IR prints one text block per generated kernel when
// `ONEDNN_VERBOSE` holds `x64ir=<level>` in a dev-mode build. Levels are
// cumulative, so `x64ir=N` prints everything at or below `N`:
//   1 - summary: kernel name, ISA, descriptor, IR counts, code size, and
//       creation timings
//   2 - nothing beyond level 1 yet
//   3 - the IR listing
//
// The level is independent of `debuginfo=` and of `all`. `all` reads as the
// highest debuginfo level, so gating on debuginfo would make `all` print the
// listing of every kernel.
//
// The whole block is formatted into one string and printed with a single stdio
// call, so blocks of kernels created concurrently do not interleave. The block
// is framed for extraction:
//
//   x64ir: begin #<seq> <kernel name> isa=<isa>
//   ...
//   x64ir: end #<seq>
//
// `<seq>` counts the printed blocks in the process. It is not the
// `ONEDNN_JIT_DUMP` file index, which is private to the JIT code registration.
// A block is matched to a dump file by kernel name and code size instead.
//
// The formatting functions are compiled in every build so that the unit tests
// run without dev mode. Only `verbose_level()` depends on dev mode.

#include <chrono>
#include <string>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Returns the level of the last `x64ir=<level>` token in `verbose_value`, a
// comma-separated `ONEDNN_VERBOSE` value. Returns 0 when there is no such
// token. The level is parsed with `atoi()`, so non-numeric text reads as 0.
// Negative levels are clamped to 0.
//
// Export for testing.
int DNNL_API parse_x64ir_level(const std::string &verbose_value);

// Returns the level set by `ONEDNN_VERBOSE`. The variable is read once per
// process. Returns 0 in builds without dev mode.
int verbose_level();

// Measures the time between consecutive `lap()` calls. A disabled timer reads
// no clock and returns 0, so a kernel that prints nothing pays nothing.
class lap_timer_t {
public:
    explicit lap_timer_t(bool enabled);
    // Returns microseconds since construction or the previous `lap()`.
    double lap();

private:
    bool enabled_;
    std::chrono::steady_clock::time_point last_;
};

// Kernel-level facts the IR does not hold. The kernel fills them in its
// `generate()`.
//
//   name      - kernel name, as returned by `name()`
//   isa       - ISA the kernel is generated for
//   desc      - one line describing the kernel's descriptor. The format is up
//               to the kernel.
//   code_size - total code size in bytes at the end of `generate()`,
//               including static data
//   build_us  - time to build the IR, in microseconds
//   alloc_us  - time to configure and allocate registers, in microseconds
//   emit_us   - time from the end of allocation to the end of `emit()`, in
//               microseconds
struct kernel_dump_t {
    const char *name = "";
    cpu_isa_t isa = isa_undef;
    std::string desc;
    size_t code_size = 0;
    double build_us = 0;
    double alloc_us = 0;
    double emit_us = 0;
};

// Returns the listing of `ir` before allocation. The listing has one line per
// operation, prefixed with the operation index and indented by loop depth.
// It is preceded by one line per vec data type that lists the vregs holding
// it.
//
// Operands print as follows:
//   r<id>, v<id>, m<id> - gpr, vec, and mask vreg with id `<id>`
//   [r<id>+<disp>]      - memory operand. `[param+<disp>]` reads the kernel
//                         argument struct. A converting access is prefixed
//                         with the data type in memory (`bf16[r3+0]`).
//   L<id>               - label with id `<id>`
//
// Export for testing.
std::string DNNL_API to_string(const ir_t &ir);

// Returns the whole block for one kernel at `level`, framed with sequence
// number `seq`.
//
// Export for testing.
std::string DNNL_API format_kernel_dump(
        int level, int seq, const kernel_dump_t &info, const ir_t &ir);

// Formats the block for one kernel with the next sequence number and prints it
// to stdout with a single call.
void print_kernel_dump(int level, const kernel_dump_t &info, const ir_t &ir);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
