#include "shared/utils.h"
#include "shared/most_common.h"
#include "shared/lrs_test.h"
#include "non_iid/collision_test.h"
#include "non_iid/lz78y_test.h"
#include "non_iid/multi_mmc_test.h"
#include "non_iid/lag_test.h"
#include "non_iid/multi_mcw_test.h"
#include "non_iid/compression_test.h"
#include "non_iid/markov_test.h"

#include <pthread.h>
#include <getopt.h>
#include <limits.h>
#include <sys/sysinfo.h>

#include "generic.h"
typedef struct DATA_FOR_THREADS {
    int counter;
    char *indir;
    char *outdir;
} DATA_FOR_THREADS;


data_t global_data;
bool global_initial_entropy, global_all_bits;
pthread_mutex_t __lock;
static int __verbose;

[[noreturn]] void print_usage() {
    printf("Usage is: ea_non_iid [-i|-c] [-a|-t] [-v] [-l <index>,<samples> ] <file_name> [bits_per_symbol]\n\n");
    printf("\t <file_name>: Must be relative path to a binary file with at least 1 million entries (samples).\n");
    printf("\t [bits_per_symbol]: Must be between 1-8, inclusive. By default this value is inferred from the data.\n");
    printf("\t [-i|-c]: '-i' for initial entropy estimate, '-c' for conditioned sequential dataset entropy estimate. The initial entropy estimate is the default.\n");
    printf("\t [-a|-t]: '-a' produces the 'H_bitstring' assessment using all read bits, '-t' truncates the bitstring used to produce the `H_bitstring` assessment to %d bits. Test all data by default.\n",
           MIN_SIZE);
    printf("\t Note: When testing binary data, no `H_bitstring` assessment is produced, so the `-a` and `-t` options produce the same results for the initial assessment of binary data.\n");
    printf("\t -v: Optional verbosity flag for more output. Can be used multiple times.\n");
    printf("\t -l <index>,<samples>\tRead the <index> substring of length <samples>.\n");
    printf("\n");
    printf("\t Samples are assumed to be packed into 8-bit values, where the least significant 'bits_per_symbol'\n");
    printf("\t bits constitute the symbol.\n");
    printf("\n");
    printf("\t -i: Initial Entropy Estimate (Section 3.1.3)\n");
    printf("\n");
    printf("\t\t Computes the initial entropy estimate H_I as described in Section 3.1.3\n");
    printf("\t\t (not accounting for H_submitter) using the entropy estimators specified in\n");
    printf("\t\t Section 6.3.  If 'bits_per_symbol' is greater than 1, the samples are also\n");
    printf("\t\t converted to bitstrings and assessed to create H_bitstring; for multi-bit symbols,\n");
    printf("\t\t two entropy estimates are computed: H_original and H_bitstring.\n");
    printf("\t\t Returns min(H_original, bits_per_symbol X H_bitstring). The initial entropy\n");
    printf("\t\t estimate H_I = min(H_submitter, H_original, bits_per_symbol X H_bitstring).\n");
    printf("\n");
    printf("\t -c: Conditioned Sequential Dataset Entropy Estimate (Section 3.1.5.2)\n");
    printf("\n");
    printf("\t\t Computes the entropy estimate per bit h' for the conditioned sequential dataset if the\n");
    printf("\t\t conditioning function is non-vetted. The samples are converted to a bitstring.\n");
    printf("\t\t Returns h' = min(H_bitstring).\n");
    printf("\n");
    exit(-1);
}

void *func(void *params) {
    bool initial_entropy, all_bits;
    int verbose = 0;
    char *file_path;
    double H_original, H_bitstring, ret_min_entropy;
    data_t data;
    int opt;
    double bin_t_tuple_res = -1.0, bin_lrs_res = -1.0;
    double t_tuple_res = -1.0, lrs_res = -1.0;
    unsigned long subsetIndex = ULONG_MAX;
    unsigned long subsetSize = 0;
    unsigned long long inint;
    char *nextOption;
    DATA_FOR_THREADS *thread_data = (DATA_FOR_THREADS *)params;
    verbose = __verbose;
    __verbose = 0;

    // collect data
    int i = thread_data->counter;
    pthread_mutex_unlock(&__lock);
    if (i > 1000) {
        return NULL;
    }

    initial_entropy = global_initial_entropy;
    all_bits = global_all_bits;
    data.word_size = global_data.word_size;

    asprintf(&file_path, "data/%s/%05d.bin", thread_data->indir, i);
    puts(file_path);
    if (verbose > 0) printf("Opening file: '%s'\n", file_path);

    if (!read_file_subset(file_path, &data, subsetIndex, subsetSize)) {
        printf("Error reading file.\n");
        print_usage();
    }

    if (verbose > 0)
        printf("Loaded %ld samples of %d distinct %d-bit-wide symbols\n", data.len, data.alph_size, data.word_size);

    if (data.alph_size <= 1) {
        printf("Symbol alphabet consists of 1 symbol. No entropy awarded...\n");
        free_data(&data);
        exit(-1);
    }

    if (!all_bits && (data.blen > MIN_SIZE)) data.blen = MIN_SIZE;

    if ((verbose > 0) && ((data.alph_size > 2) || !initial_entropy))
        printf("Number of Binary Symbols: %ld\n", data.blen);
    if (data.len < MIN_SIZE) printf("\n*** Warning: data contains less than %d samples ***\n\n", MIN_SIZE);
    if (verbose > 0) {
        if (data.alph_size < (1 << data.word_size)) printf("\nSymbols have been translated.\n");
    }

    // The maximum min-entropy is -log2(1/2^word_size) = word_size
    // The maximum bit string min-entropy is 1.0
    H_original = data.word_size;
    H_bitstring = 1.0;

    if (verbose > 0) {
        printf("\nRunning non-IID tests...\n\n");
        printf("Running Most Common Value Estimate...\n");
    }

    // Section 6.3.1 - Estimate entropy with Most Common Value
    if (((data.alph_size > 2) || !initial_entropy)) {
        ret_min_entropy = most_common(data.bsymbols, data.blen, 2, verbose, "Bitstring");
        if (verbose > 0) printf("\tMost Common Value Estimate (bit string) = %f / 1 bit(s)\n", ret_min_entropy);
        H_bitstring = min(ret_min_entropy, H_bitstring);
    }
    log_to_file(&ret_min_entropy, true, i, thread_data->outdir);

    if (initial_entropy) {
        ret_min_entropy = most_common(data.symbols, data.len, data.alph_size, verbose, "Literal");
        if (verbose > 0)
            printf("\tMost Common Value Estimate = %f / %d bit(s)\n", ret_min_entropy, data.word_size);
        H_original = min(ret_min_entropy, H_original);
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (verbose > 0) printf("\nRunning Entropic Statistic Estimates (bit strings only)...\n");

    // Section 6.3.2 - Estimate entropy with Collision Test (for bit strings only)
    if (((data.alph_size > 2) || !initial_entropy)) {
        ret_min_entropy = collision_test(data.bsymbols, data.blen, verbose, "Bitstring");
        if (verbose > 0) printf("\tCollision Test Estimate (bit string) = %f / 1 bit(s)\n", ret_min_entropy);
        H_bitstring = min(ret_min_entropy, H_bitstring);
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (initial_entropy && (data.alph_size == 2)) {
        ret_min_entropy = collision_test(data.symbols, data.len, verbose, "Literal");
        if (verbose > 0) printf("\tCollision Test Estimate = %f / 1 bit(s)\n", ret_min_entropy);
        H_original = min(ret_min_entropy, H_original);
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    // Section 6.3.3 - Estimate entropy with Markov Test (for bit strings only)
    if (((data.alph_size > 2) || !initial_entropy)) {
        ret_min_entropy = markov_test(data.bsymbols, data.blen, verbose, "Bitstring");
        if (verbose > 0) printf("\tMarkov Test Estimate (bit string) = %f / 1 bit(s)\n", ret_min_entropy);
        H_bitstring = min(ret_min_entropy, H_bitstring);
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (initial_entropy && (data.alph_size == 2)) {
        ret_min_entropy = markov_test(data.symbols, data.len, verbose, "Literal");
        if (verbose > 0) printf("\tMarkov Test Estimate = %f / 1 bit(s)\n", ret_min_entropy);
        H_original = min(ret_min_entropy, H_original);
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    // Section 6.3.4 - Estimate entropy with Compression Test (for bit strings only)
    if (((data.alph_size > 2) || !initial_entropy)) {
        ret_min_entropy = compression_test(data.bsymbols, data.blen, verbose, "Bitstring");
        if (ret_min_entropy >= 0) {
            if (verbose > 0) printf("\tCompression Test Estimate (bit string) = %f / 1 bit(s)\n", ret_min_entropy);
            H_bitstring = min(ret_min_entropy, H_bitstring);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (initial_entropy && (data.alph_size == 2)) {
        ret_min_entropy = compression_test(data.symbols, data.len, verbose, "Literal");
        if (verbose > 0) printf("\ttCompression Test Estimate = %f / 1 bit(s)\n", ret_min_entropy);
        H_original = min(ret_min_entropy, H_original);
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (verbose > 0) printf("\nRunning Tuple Estimates...\n");

    // Section 6.3.5 - Estimate entropy with t-Tuple Test

    if (((data.alph_size > 2) || !initial_entropy)) {
        SAalgs(data.bsymbols, data.blen, 2, bin_t_tuple_res, bin_lrs_res, verbose, "Bitstring");
        if (bin_t_tuple_res >= 0.0) {
            if (verbose > 0) printf("\tT-Tuple Test Estimate (bit string) = %f / 1 bit(s)\n", bin_t_tuple_res);
            H_bitstring = min(bin_t_tuple_res, H_bitstring);
        }
    }
    log_to_file(&bin_t_tuple_res, false, i, thread_data->outdir);

    if (initial_entropy) {
        SAalgs(data.symbols, data.len, data.alph_size, t_tuple_res, lrs_res, verbose, "Literal");
        if (t_tuple_res >= 0.0) {
            if (verbose > 0) printf("\tT-Tuple Test Estimate = %f / %d bit(s)\n", t_tuple_res, data.word_size);
            H_original = min(t_tuple_res, H_original);
        }
    }
    log_to_file(&t_tuple_res, false, i, thread_data->outdir);

    // Section 6.3.6 - Estimate entropy with LRS Test
    if (((data.alph_size > 2) || !initial_entropy)) {
        if (verbose > 0) printf("\tLRS Test Estimate (bit string) = %f / 1 bit(s)\n", bin_lrs_res);
        H_bitstring = min(bin_lrs_res, H_bitstring);
    }
    log_to_file(&bin_lrs_res, false, i, thread_data->outdir);


    if (initial_entropy) {
        if (verbose > 0) printf("\tLRS Test Estimate = %f / %d bit(s)\n", lrs_res, data.word_size);
        H_original = min(lrs_res, H_original);
    }
    log_to_file(&lrs_res, false, i, thread_data->outdir);

    if (verbose > 0) printf("\nRunning Predictor Estimates...\n");

    if (((data.alph_size > 2) || !initial_entropy)) {
        // Section 6.3.7 - Estimate entropy with Multi Most Common in Window Test
        ret_min_entropy = multi_mcw_test(data.bsymbols, data.blen, 2, verbose, "Bitstring");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tMulti Most Common in Window (MultiMCW) Prediction Test Estimate (bit string) = %f / 1 bit(s)\n", ret_min_entropy);
            H_bitstring = min(ret_min_entropy, H_bitstring);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (initial_entropy) {
        ret_min_entropy = multi_mcw_test(data.symbols, data.len, data.alph_size, verbose, "Literal");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tMulti Most Common in Window (MultiMCW) Prediction Test Estimate = %f / %d bit(s)\n",
                       ret_min_entropy, data.word_size);
            H_original = min(ret_min_entropy, H_original);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    // Section 6.3.8 - Estimate entropy with Lag Prediction Test
    if (((data.alph_size > 2) || !initial_entropy)) {
        ret_min_entropy = lag_test(data.bsymbols, data.blen, 2, verbose, "Bitstring");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tLag Prediction Test Estimate (bit string) = %f / 1 bit(s)\n", ret_min_entropy);
            H_bitstring = min(ret_min_entropy, H_bitstring);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (initial_entropy) {
        ret_min_entropy = lag_test(data.symbols, data.len, data.alph_size, verbose, "Literal");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tLag Prediction Test Estimate = %f / %d bit(s)\n", ret_min_entropy, data.word_size);
            H_original = min(ret_min_entropy, H_original);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    // Section 6.3.9 - Estimate entropy with Multi Markov Model with Counting Test (MultiMMC)
    if (((data.alph_size > 2) || !initial_entropy)) {
        ret_min_entropy = multi_mmc_test(data.bsymbols, data.blen, 2, verbose, "Bitstring");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tMulti Markov Model with Counting (MultiMMC) Prediction Test Estimate (bit string) = %f / 1 bit(s)\n",
                       ret_min_entropy);
            H_bitstring = min(ret_min_entropy, H_bitstring);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (initial_entropy) {
        ret_min_entropy = multi_mmc_test(data.symbols, data.len, data.alph_size, verbose, "Literal");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tMulti Markov Model with Counting (MultiMMC) Prediction Test Estimate = %f / %d bit(s)\n",
                       ret_min_entropy, data.word_size);
            H_original = min(ret_min_entropy, H_original);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    // Section 6.3.10 - Estimate entropy with LZ78Y Test
    if (((data.alph_size > 2) || !initial_entropy)) {
        ret_min_entropy = LZ78Y_test(data.bsymbols, data.blen, 2, verbose, "Bitstring");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tLZ78Y Prediction Test Estimate (bit string) = %f / 1 bit(s)\n", ret_min_entropy);
            H_bitstring = min(ret_min_entropy, H_bitstring);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (initial_entropy) {
        ret_min_entropy = LZ78Y_test(data.symbols, data.len, data.alph_size, verbose, "Literal");
        if (ret_min_entropy >= 0) {
            if (verbose > 0)
                printf("\tLZ78Y Prediction Test Estimate = %f / %d bit(s)\n", ret_min_entropy, data.word_size);
            H_original = min(ret_min_entropy, H_original);
        }
    }
    log_to_file(&ret_min_entropy, false, i, thread_data->outdir);

    if (verbose > 0) {
        printf("\n");
        if (initial_entropy) {
            printf("H_original: %f\n", H_original);
            if (data.alph_size > 2) {
                printf("H_bitstring: %f\n\n", H_bitstring);
                printf("min(H_original, %d X H_bitstring): %f\n\n", data.word_size,
                       min(H_original, data.word_size * H_bitstring));
            }
        } else printf("h': %f\n", H_bitstring);
    } else {
        double h_assessed = data.word_size;

        if ((data.alph_size > 2) || !initial_entropy) {
            h_assessed = min(h_assessed, H_bitstring * data.word_size);
            if (verbose > 0) printf("H_bitstring = %.17g\n", H_bitstring);
        }

        if (initial_entropy) {
            h_assessed = min(h_assessed, H_original);
            if (verbose > 0) printf("H_original: %.17g\n", H_original);
        }

        if (verbose > 0) printf("Assessed min entropy: %.17g\n", h_assessed);
    }

    free_data(&data);
}

int main(int argc, char *argv[]) {

    if (pthread_mutex_init(&__lock, NULL) != 0) {
        printf("\n mutex init has failed\n");
        return 1;
    }

    int verbose = 0;
    char *file_path;
    double H_original, H_bitstring, ret_min_entropy;
    int opt;
    double bin_t_tuple_res = -1.0, bin_lrs_res = -1.0;
    double t_tuple_res = -1.0, lrs_res = -1.0;
    unsigned long subsetIndex = ULONG_MAX;
    unsigned long subsetSize = 0;
    unsigned long long inint;
    char *nextOption;


    global_data.word_size = 0;

    global_initial_entropy = true;
    global_all_bits = true;

    while ((opt = getopt(argc, argv, "icatvl:")) != -1) {
        switch (opt) {
            case 'i':
                global_initial_entropy = true;
                break;
            case 'c':
                global_initial_entropy = false;
                break;
            case 'a':
                global_all_bits = true;
                break;
            case 't':
                global_all_bits = false;
                break;
            case 'v':
                verbose++;
                break;
            case 'l':
                inint = strtoull(optarg, &nextOption, 0);
                if ((inint > ULONG_MAX) || (errno == EINVAL) || (nextOption == NULL) || (*nextOption != ',')) {
                    print_usage();
                }
                subsetIndex = inint;

                nextOption++;

                inint = strtoull(nextOption, NULL, 0);
                if ((inint > ULONG_MAX) || (errno == EINVAL)) {
                    print_usage();
                }
                subsetSize = inint;
                break;
            default:
                print_usage();
        }
    }

    argc -= optind;
    argv += optind;

    // Parse args
    if ((argc != 1) && (argc != 2)) {
        printf("Incorrect usage.\n");
        print_usage();
    }
    // get filename
    file_path = argv[0];

    if (argc == 2) {
        // get bits per word
        inint = atoi(argv[1]);
        if (inint < 1 || inint > 8) {
            printf("Invalid bits per symbol.\n");
            print_usage();
        } else {
            global_data.word_size = inint;
        }
    }

    int counter[1000] = {0};
    for (int i = 1; i <= 1000; i++) {
        counter[i-1] = i;
    }

    pthread_t thread_id1, thread_id2, thread_id3, thread_id4, thread_id5, thread_id6, thread_id7, thread_id8, thread_id9, thread_id10, thread_id11, thread_id12, thread_id13, thread_id14;

    DATA_FOR_THREADS params;

#if __BINARY_DATA__
    asprintf(&(params.outdir), "%s", "result_binary");
    asprintf(&(params.indir), "%s", "data_binary");

#if __MULTIPLE_THREADS__
    for (int i = 1; i <= 1000; i+=14) {
#else
    for (int i = 1; i <= 1000; i+=1) {
#endif

        // create and start
        pthread_mutex_lock(&__lock);
        __verbose = 1;
        params.counter = i;
        pthread_create(&thread_id1, NULL, func, (void *)&params);

#if __MULTIPLE_THREADS__
        pthread_mutex_lock(&__lock);
        params.counter = i+1;
        pthread_create(&thread_id2, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+2;
        pthread_create(&thread_id3, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+3;
        pthread_create(&thread_id4, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+4;
        pthread_create(&thread_id5, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+5;
        pthread_create(&thread_id6, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+6;
        pthread_create(&thread_id7, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+7;
        pthread_create(&thread_id8, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+8;
        pthread_create(&thread_id9, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+9;
        pthread_create(&thread_id10, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+10;
        pthread_create(&thread_id11, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+11;
        pthread_create(&thread_id12, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+12;
        pthread_create(&thread_id13, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+13;
        pthread_create(&thread_id14, NULL, func, (void *)&params);
#endif

        // wait for complete
        pthread_join(thread_id1, NULL);
#if __MULTIPLE_THREADS__
        pthread_join(thread_id2, NULL);
        pthread_join(thread_id3, NULL);
        pthread_join(thread_id4, NULL);
        pthread_join(thread_id5, NULL);
        pthread_join(thread_id6, NULL);
        pthread_join(thread_id7, NULL);
        pthread_join(thread_id8, NULL);
        pthread_join(thread_id9, NULL);
        pthread_join(thread_id10, NULL);
        pthread_join(thread_id11, NULL);
        pthread_join(thread_id12, NULL);
        pthread_join(thread_id13, NULL);
        pthread_join(thread_id14, NULL);
#endif
    }
#endif // __BINARY_DATA__

#if __4BIT_DATA__

    // data 4bit
    asprintf(&(params.outdir), "%s", "result_4bit");
    asprintf(&(params.indir), "%s", "data_4bit");

#if __MULTIPLE_THREADS__
    for (int i = 1; i <= 1000; i+=14) {
#else
    for (int i = 1; i <= 1000; i+=1) {
#endif

        // create and start
        pthread_mutex_lock(&__lock);
        __verbose = 1;
        params.counter = i;
        pthread_create(&thread_id1, NULL, func, (void *)&params);

#if __MULTIPLE_THREADS__
        pthread_mutex_lock(&__lock);
        params.counter = i+1;
        pthread_create(&thread_id2, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+2;
        pthread_create(&thread_id3, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+3;
        pthread_create(&thread_id4, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+4;
        pthread_create(&thread_id5, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+5;
        pthread_create(&thread_id6, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+6;
        pthread_create(&thread_id7, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+7;
        pthread_create(&thread_id8, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+8;
        pthread_create(&thread_id9, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+9;
        pthread_create(&thread_id10, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+10;
        pthread_create(&thread_id11, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+11;
        pthread_create(&thread_id12, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+12;
        pthread_create(&thread_id13, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+13;
        pthread_create(&thread_id14, NULL, func, (void *)&params);
#endif

        // wait for complete
        pthread_join(thread_id1, NULL);
#if __MULTIPLE_THREADS__
        pthread_join(thread_id2, NULL);
        pthread_join(thread_id3, NULL);
        pthread_join(thread_id4, NULL);
        pthread_join(thread_id5, NULL);
        pthread_join(thread_id6, NULL);
        pthread_join(thread_id7, NULL);
        pthread_join(thread_id8, NULL);
        pthread_join(thread_id9, NULL);
        pthread_join(thread_id10, NULL);
        pthread_join(thread_id11, NULL);
        pthread_join(thread_id12, NULL);
        pthread_join(thread_id13, NULL);
        pthread_join(thread_id14, NULL);
#endif
    }
#endif // __4BIT_DATA__

#if __8BIT_DATA__

    // data 8bit
    asprintf(&(params.outdir), "%s", "result_8bit");
    asprintf(&(params.indir), "%s", "data_8bit");

#if __MULTIPLE_THREADS__
    for (int i = 1; i <= 1000; i+=14) {
#else
    for (int i = 1; i <= 1000; i+=1) {
#endif

        // create and start
        pthread_mutex_lock(&__lock);
        __verbose = 1;
        params.counter = i;
        pthread_create(&thread_id1, NULL, func, (void *)&params);

#if __MULTIPLE_THREADS__
        pthread_mutex_lock(&__lock);
        params.counter = i+1;
        pthread_create(&thread_id2, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+2;
        pthread_create(&thread_id3, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+3;
        pthread_create(&thread_id4, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+4;
        pthread_create(&thread_id5, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+5;
        pthread_create(&thread_id6, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+6;
        pthread_create(&thread_id7, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+7;
        pthread_create(&thread_id8, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+8;
        pthread_create(&thread_id9, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+9;
        pthread_create(&thread_id10, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+10;
        pthread_create(&thread_id11, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+11;
        pthread_create(&thread_id12, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+12;
        pthread_create(&thread_id13, NULL, func, (void *)&params);

        pthread_mutex_lock(&__lock);
        params.counter = i+13;
        pthread_create(&thread_id14, NULL, func, (void *)&params);
#endif

        // wait for complete
        pthread_join(thread_id1, NULL);
#if __MULTIPLE_THREADS__
        pthread_join(thread_id2, NULL);
        pthread_join(thread_id3, NULL);
        pthread_join(thread_id4, NULL);
        pthread_join(thread_id5, NULL);
        pthread_join(thread_id6, NULL);
        pthread_join(thread_id7, NULL);
        pthread_join(thread_id8, NULL);
        pthread_join(thread_id9, NULL);
        pthread_join(thread_id10, NULL);
        pthread_join(thread_id11, NULL);
        pthread_join(thread_id12, NULL);
        pthread_join(thread_id13, NULL);
        pthread_join(thread_id14, NULL);
#endif
    }
#endif // __8BIT_DATA__
    return 0;
}