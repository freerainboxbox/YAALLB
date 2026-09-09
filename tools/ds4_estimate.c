/*
 * ds4-estimate: projected ds4 memory footprint for one serve configuration.
 *
 * YAALLB schedules models against a VRAM budget and must know what a spawned
 * `ds4-server` will occupy *before* it is spawned. ds4 owns the only correct
 * answer: the context/KV/scratch term depends on the model shape recorded in
 * the GGUF (Flash vs Pro vs GLM), the backend, the prefill chunk, and the SSD
 * streaming mode, and it is computed by ds4 itself for its own startup logs.
 * This program asks ds4 for it and prints the components as one JSON object,
 * so YAALLB never re-derives ds4's shape-dependent arithmetic in Python.
 *
 * Build it against a built ds4 tree (objects already compiled):
 *
 *   make -C <ds4_dir> -f <yaallb>/tools/ds4-estimate.mk
 *
 * Usage:
 *   ds4-estimate --model GGUF [--backend metal|cuda|rocm|cpu] --ctx N
 *                [--prefill-chunk N] [--ssd-streaming]
 *                [--mtp-model GGUF] [--vision GGUF]
 *
 * Relative --model/--mtp-model/--vision paths resolve against the current
 * working directory, exactly as they do for ds4-server.
 *
 * ds4 takes a process-wide instance lock while an engine is open. When
 * DS4_LOCK_FILE is unset this program takes a private lock file so it never
 * refuses to run next to a live ds4-server (and never blocks one).
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ds4.h"

#define DS4_ESTIMATE_SCHEMA_VERSION 1

static void usage(void) {
    fprintf(stderr,
            "usage: ds4-estimate --model GGUF "
            "[--backend metal|cuda|rocm|cpu] --ctx N "
            "[--prefill-chunk N] [--ssd-streaming] "
            "[--mtp-model GGUF] [--vision GGUF]\n");
}

/* Bytes of a mapped GGUF (weights YAALLB must budget as resident). A missing
 * file is reported as 0 and left to ds4, which fails the engine open. */
static uint64_t file_bytes(const char *path) {
    if (!path || !path[0]) return 0;
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (uint64_t)st.st_size;
}

static const char *need_value(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-estimate: missing value for %s\n", opt);
        usage();
        exit(2);
    }
    return argv[++(*i)];
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    /* ds4 folds ROCm into its CUDA backend id, as ds4-server does. */
    if (!strcmp(s, "cuda") || !strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-estimate: invalid --backend: %s\n", s);
    exit(2);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *mtp_model = NULL;
    const char *vision = NULL;
    const char *backend_name = NULL;
    int ctx = 0;
    uint32_t prefill_chunk = 0;
    bool ssd_streaming = false;
    char private_lock[128] = {0};

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--model")) {
            model = need_value(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-model")) {
            mtp_model = need_value(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vision")) {
            vision = need_value(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--backend")) {
            backend_name = need_value(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx")) {
            ctx = atoi(need_value(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--prefill-chunk")) {
            prefill_chunk = (uint32_t)atoi(need_value(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--ssd-streaming")) {
            ssd_streaming = true;
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "ds4-estimate: unknown option: %s\n", arg);
            usage();
            return 2;
        }
    }
    if (!model || !model[0]) {
        fprintf(stderr, "ds4-estimate: --model GGUF is required\n");
        usage();
        return 2;
    }
    if (ctx <= 0) {
        fprintf(stderr, "ds4-estimate: --ctx N must be positive\n");
        usage();
        return 2;
    }

    /* Mirror ds4-server's default_server_backend() when --backend is omitted.
     * DS4_NO_GPU is private to the ds4 build, so a tree compiled with
     * `make cpu` must be told: build this program with -DDS4_ESTIMATE_CPU_ONLY
     * (see ds4-estimate.mk). */
    ds4_backend backend;
    if (backend_name) {
        backend = parse_backend(backend_name);
    } else {
#if defined(DS4_ESTIMATE_CPU_ONLY)
        backend = DS4_BACKEND_CPU;
#elif defined(__APPLE__)
        backend = DS4_BACKEND_METAL;
#else
        backend = DS4_BACKEND_CUDA;
#endif
    }

    if (!getenv("DS4_LOCK_FILE")) {
        snprintf(private_lock, sizeof(private_lock),
                 "/tmp/ds4-estimate-%ld.lock", (long)getpid());
        setenv("DS4_LOCK_FILE", private_lock, 1);
    }

    /*
     * inspect_only opens the GGUF (mmap + metadata, which is what selects the
     * active model shape the estimator needs) and returns before any weights
     * prefetch, vocab load, or graph allocation. --mtp-model/--vision support
     * GGUFs are metadata-opened too, so their sizes are reported without
     * loading them.
     */
    ds4_engine_options opt = {0};
    opt.model_path = model;
    opt.mtp_path = mtp_model;
    opt.vision_path = vision;
    opt.backend = backend;
    opt.context_size = ctx;
    opt.prefill_chunk = prefill_chunk;
    opt.ssd_streaming = ssd_streaming;
    opt.inspect_only = true;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        fprintf(stderr, "ds4-estimate: ds4_engine_open failed for %s\n", model);
        if (private_lock[0]) unlink(private_lock);
        return 1;
    }

    /* Ask ds4 with its own effective prefill chunk (0 keeps its default). */
    const uint32_t effective_chunk = ds4_engine_prefill_chunk(engine);
    const ds4_context_memory m = ds4_context_memory_estimate_with_prefill_mode(
            backend, ctx, effective_chunk, ssd_streaming);
    const uint64_t context_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;

    /* model_name/backend_name are ds4's own fixed shape/backend strings, so
     * they need no JSON escaping; nothing here is model- or path-derived. */
    printf("{\"version\":%d,"
           "\"model_name\":\"%s\","
           "\"backend\":\"%s\","
           "\"ctx\":%d,"
           "\"layers\":%d,"
           "\"model_bytes\":%llu,"
           "\"support_bytes\":%llu,"
           "\"vision_bytes\":%llu,"
           "\"prefill_chunk\":%u,"
           "\"prefill_cap\":%u,"
           "\"raw_cap\":%u,"
           "\"comp_cap\":%u,"
           "\"raw_bytes\":%llu,"
           "\"compressed_bytes\":%llu,"
           "\"scratch_bytes\":%llu,"
           "\"context_bytes\":%llu,"
           "\"has_mtp\":%s,"
           "\"mtp_draft_tokens\":%d}\n",
           DS4_ESTIMATE_SCHEMA_VERSION,
           ds4_engine_model_name(engine),
           ds4_backend_name(backend),
           ctx,
           ds4_engine_layer_count(engine),
           (unsigned long long)ds4_engine_model_bytes(engine),
           (unsigned long long)file_bytes(mtp_model),
           (unsigned long long)file_bytes(vision),
           effective_chunk,
           m.prefill_cap,
           m.raw_cap,
           m.comp_cap,
           (unsigned long long)m.raw_bytes,
           (unsigned long long)m.compressed_bytes,
           (unsigned long long)m.scratch_bytes,
           (unsigned long long)context_bytes,
           ds4_engine_has_mtp(engine) ? "true" : "false",
           ds4_engine_mtp_draft_tokens(engine));

    ds4_engine_close(engine);
    if (private_lock[0]) unlink(private_lock);
    return 0;
}
