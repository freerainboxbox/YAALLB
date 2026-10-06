/*
 * Does this ds4 tree expose the drafter-graph accessor?
 *
 * ds4_engine_spec_graph_memory_estimate() (and its ds4_spec_graph_memory) is
 * not part of released ds4: it exists only in trees that carry it, and it is the
 * one source of a session's DSpark/MTP graph scratch, whose sizes follow the
 * support checkpoint's stage and target-layer counts rather than any file size.
 * tools/ds4_estimate.c reports those terms where the tree can answer them and
 * says "unknown" where it cannot, so the same estimator builds against any ds4.
 *
 * Whether the accessor exists is a property of the tree's ds4.h, and a missing
 * or reshaped accessor is a *compile* error (a typedef cannot be probed at
 * runtime), so the question is asked where it can be answered: ds4-estimate.mk
 * compiles this file with -fsyntax-only and defines
 * DS4_HAVE_SPEC_GRAPH_ACCESSOR when that succeeds.
 *
 * It is only ever syntax-checked, never linked or run. Naming every field the
 * estimator prints is what makes a reshaped struct a probe failure instead of a
 * silently wrong build.
 */

#include "ds4.h"

int ds4_spec_probe(void);

int ds4_spec_probe(void) {
    const ds4_spec_graph_memory s = ds4_engine_spec_graph_memory_estimate(
            (ds4_engine *)0, /*ctx_size=*/1, /*prefill_chunk=*/1);
    return (int)(s.dspark_capture_bytes + s.verifier_scratch_bytes +
                 s.host_scratch_bytes + s.total_bytes +
                 s.dspark_capture_stages);
}
