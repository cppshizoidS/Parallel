#include <algorithm>
#include <iostream>
#include <cstdio>
#include <string>
#include <chrono>
#include <vector>
#include <functional>

using namespace std;

struct comp_result
{
    int    value;
    string title;
    float  time;
};

using hrc       = chrono::high_resolution_clock;
using fdur      = chrono::duration<float>;
using fn_solver = function<int(string, string)>;
using fn_bench  = function<comp_result()>;

__device__ int d_count;

const int THREADS_PER_BLOCK = 64;

void
save_char_position(string& str, vector<vector<int>>& pos, char c, int charPos)
{
    if (pos.empty()) {
        pos = vector<vector<int>>(str.size());
    }

    for (int i = 0; i < str.size(); i++) {
        if (c == str[i]) {
            pos[i].push_back(charPos);
        }
    }
}

void
count_substr(
    vector<vector<int>>& pos,
    int* count,
    int prev_char_pos = -1,
    int cur_char      = 0
) {
    if (cur_char == pos.size()) {
        (*count)++;
        return;
    }

    for (auto& p : pos[cur_char]) {
        if (prev_char_pos < p) {
            count_substr(pos, count, p, cur_char + 1);
        }
    }
}

__device__ bool
check_combination(int* pos, int* sizes, int arraySize, long long combination)
{
    int prev_char_pos = -1;
    int sizesSum      = 0;

    for (int i = 0; i < arraySize; i++) {
        long long p = combination % sizes[i];

        if (prev_char_pos > pos[sizesSum + p]) {
            return false;
        }

        combination   /= sizes[i];
        prev_char_pos  = pos[sizesSum + p];
        sizesSum      += sizes[i];
    }

    return true;
}

__global__ void
count_substr(
    int* pos,
    long long combinations,
    int* sizes,
    int array_size,
    int combinations_per_thread
) {
    __shared__ int block_combinations;
    __shared__ int total_block_combinations;

    int id = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (id == 0) {
        d_count = 0;
    }

    if (threadIdx.x == 0) {
        block_combinations       = 0;
        total_block_combinations = 0;
    }

    int thread_combinations = 0;

    long long start_combination = id * combinations_per_thread;
    if (start_combination >= combinations) {
        return;
    }

    for (
        long long i = start_combination;
        i < combinations && i < start_combination + combinations_per_thread;
        i++
    ) {
        if (check_combination(pos, sizes, array_size, i)) {
            thread_combinations++;
        }
    }

    atomicAdd(&block_combinations, thread_combinations);
    __syncthreads();

    if (atomicExch(&total_block_combinations, 1) == 0) {
        atomicAdd(&d_count, block_combinations);
    }
}

void
get_blocks_and_threads(long long elems_count, int* blocks, int* threads)
{
    *blocks  = (elems_count - 1) / THREADS_PER_BLOCK + 1;
    *threads = (elems_count < THREADS_PER_BLOCK
        ? elems_count : THREADS_PER_BLOCK);
}

int
solve_parallel(string src, string sub)
{
    int *d_pos, *d_sizes;
    int blocks, threads, count;
    int thread_combinations = 100;
    int max_blocks          = 4000;
    int total_size          = 0;
    int offset              = 0;
    long long combinations  = 1;

    vector<int> sizes;
    vector<vector<int>> pos;

    for (int i = 0; i < src.size(); i++) {
        save_char_position(sub, pos, src[i], i);
    }

    for (auto& p : pos) {
        if (p.empty()) {
            return 0;
        }
    }

    for (auto& p : pos) {
        sizes.push_back(p.size());
    }

    for (auto& p : pos) {
        total_size   += p.size();
        combinations *= p.size();
    }

    cudaMalloc(&d_pos, total_size * sizeof(int));

    for (int i = 0; i < pos.size(); i++) {
        cudaMemcpy(
            d_pos + offset,
            &pos[i].front(),
            pos[i].size() * sizeof(int),
            cudaMemcpyHostToDevice
        );
        offset += pos[i].size();
    }

    int bytes = pos.size() * sizeof(int);
    cudaMalloc(&d_sizes, bytes);
    cudaMemcpy(d_sizes, &sizes.front(), bytes, cudaMemcpyHostToDevice);
  
    get_blocks_and_threads(
        combinations / thread_combinations + 1, &blocks, &threads);
  
    while (blocks > max_blocks) {
        thread_combinations *= 10;
        get_blocks_and_threads(
            combinations / thread_combinations + 1, &blocks, &threads);
    }
  
    count_substr <<< blocks, threads >>> (
        d_pos, combinations, d_sizes, sizes.size(), thread_combinations);
    cudaDeviceSynchronize();
  
    cudaMemcpyFromSymbol(&count, d_count, sizeof(int));
  
    cudaFree(d_pos);
    cudaFree(d_sizes);
    return count;
}

int
solve_sequential(string src, string sub)
{
    int count = 0;
    vector<vector<int>> pos;

    for (int i = 0; i < src.size(); i++) {
        save_char_position(sub, pos, src[i], i);
    }

    for (auto& p : pos) {
        if (p.empty()) {
            return 0;
        }
    }

    count_substr(pos, &count);
    return count;
}

void
read_str(string* dest)
{
    cout << "enter string: " << flush;
    cin >> *dest;
}

bool
is_valid(const string& src, const string& sub)
{ return src.size() >= sub.size(); }

void
print_result(const comp_result& result)
{
    cout
        << "[" << result.title << "]:\n"
        << "time:  " << result.time  << "s\n"
        << "count: " << result.value << endl;
}

fn_bench
get_solver_bench(
    const string title,
    const string src,
    const string sub,
    fn_solver solver
) {
    return [title, src, sub, solver]() -> comp_result {
        hrc::time_point start{hrc::now()};
        
        int count = solver(src, sub);

        fdur time = chrono::duration_cast<fdur>(hrc::now() - start);
        return comp_result{count, title, time.count()};
    };
}

int
main(int argc, char** argv)
{
    int command;
    string src, sub;
    fn_bench bench;

    for (;;) {
        cout <<
            "\n"
            "source string: '" << src << "'\n"
            "substring:     '" << sub << "'\n"
            "1. set source string\n"
            "2. set substring\n"
            "3. run sequential\n"
            "4. run parallel\n"
            "0. exit\n"
            "==> " << flush;
        cin >> command;

        switch (command) {
        case 1:
            read_str(&src);
            break;
        case 2:
            read_str(&sub);
            break;
        case 3: // sequential
        case 4: // parallel
            if (!is_valid(src, sub)) {
                cout <<
                    "error: source string length must be greater "
                    "than substring length" << flush;
                continue;
            }

            if (command == 3) {
                bench = get_solver_bench(
                    "sequential", src, sub, solve_sequential);
            } else {
                bench = get_solver_bench("parallel", src, sub, solve_parallel);
            }
            print_result(bench());
            break;
        case 0:
            cout << "exit…\n";
            return 0;
        default:
            cout << "wrong input!\n";
            continue;
        }
    }
}

