#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>

#include <mpi.h>

#include "my_mpi.hpp"

using namespace std;
using hrc  = chrono::high_resolution_clock;
using fdur = chrono::duration<float>;

struct range
{
    size_t index;
    size_t len;
};

struct comm_stats
{
    int rank;
    int size;
};

struct comp_result
{
    const char* title;
    fdur time;
};


range
merge(vector<int>& src, range left, range right)
{
    range united_range{left.index, left.len + right.len};

    int* data = &src[left.index];
    vector<int> items;
    items.reserve(united_range.len);

    while (left.len > 0 || right.len > 0) {
        if (left.len == 0) {
            std::memcpy(data, items.data(), items.size() * sizeof(int));
            break;
        } else if (right.len == 0) {
            std::memcpy(
                data + items.size(),
                &src[left.index],
                left.len * sizeof(int)
            );
            std::memcpy(data, items.data(), items.size() * sizeof(int));
            break;
        }

        range& biggest_range = (src[left.index] > src[right.index]
            ? right : left);

        items.push_back(src[biggest_range.index]);
        biggest_range.index++;
        biggest_range.len--;
    }

    return united_range;
}

void
merge_sort(vector<int>& src, range sort_range)
{
    if (sort_range.len <= 1) {
        return;
    }

    vector<range> ranges((sort_range.len >> 1) + (sort_range.len & 1));
    for (size_t n = 0; n < ranges.capacity(); n++) {
        ranges[n] = range{n << 1, 2};
    }
    if (sort_range.len & 1) {
        ranges.push_back(range{sort_range.len - 2, 1});
    }

    for (const auto& r : ranges) {
        std::sort(&src[r.index], &src[r.index + r.len]);
    }

    while (ranges.size() > 1) {
        for (size_t i = 0; i + 1 < ranges.size(); i += 2) {
            ranges[i >> 1] = merge(src, ranges[i], ranges[i + 1]);
        }

        // odd check
        if (ranges.size() & 1) {
            int last     = (ranges.size() - 2) >> 1;
            ranges[last] = merge(src, ranges[last], ranges.back());
        }

        ranges.resize(ranges.size() >> 1);
    }
}

range
get_range(const std::vector<int>& src, int procs_total, int rank)
{
    size_t count = src.size() / procs_total; // число элементов для процесса
    range part{(size_t)rank * count, count}; // диапазон сортировки для процесса

    if (part.index + part.len > src.size())
        part.len = src.size() - part.index; // вычисление актуальной длины

    return part;
}

void
merge_sort_parallel(vector<int>& src, comm_stats comm)
{
    // вычисление длины диапазона сортировки для текущего процесса
    size_t len = (comm.rank == 0
        ? get_range(src, comm.size, comm.rank).len : src.size());
    range sub{0, len};

    merge_sort(src, sub);

    // recvcounts - массив длин диапазонов для процессов
    // displs     - массив индексов для процессов
    vector<int> recvcounts, displs;

    for (int i = 0; i < comm.size; i++) {
        recvcounts.push_back(get_range(src, comm.size, i).len);
        displs.push_back(i * recvcounts[0]);
    }

    vector<int> tmp(src.size());
    mpi_gatherv(src, len, tmp, recvcounts, displs);

    if (comm.rank == 0) {
        for (int i = 1; i < comm.size; i++) {
            range processPart = get_range(tmp, comm.size, i); // получение диапазона для слияния
            sub               = merge(tmp, sub, processPart); // слияние отсортированных диапазонов
        }
        src = tmp;
    }
}

int
save(
    const char* filename,
    comp_result result,
    const std::vector<int>& data,
    const char* mode = "a"
) {
    FILE* file = std::fopen(filename, mode);

    if (!file) {
        std::fprintf(stderr, "fail to open %s (error %d)\n", filename, errno);
        return errno;
    }

    std::fprintf(
        file, "[%s] computing time: %fs\n", result.title, result.time.count());

    std::fputs("data: ", file);
    for (const auto& item : data) {
        std::fprintf(file, "%d ", item);
    }

    std::fputs("\n\n", file);
    std::fclose(file);

    return 0;
}

int
main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Using:\n\t%s <size>\n", argv[0]);
        return -1;
    }

    const char* OUTPUT = "results.txt";
    const int size     = std::atoi(argv[1]);

    if (size <= 0) {
        printf("size must be greater than 0\n");
        return -1;
    }

    MPI_Init(NULL, NULL);
    comm_stats world;

    MPI_Comm_size(MPI_COMM_WORLD, &world.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);

    vector<int> set_1, set_2;

    if (world.rank == 0) {
        set_1.resize(size);

        std::iota(set_1.begin(), set_1.end(), -(size >> 1)); // наполнение массива
        std::shuffle( // перетасовка
            set_1.begin(), set_1.end(), std::mt19937{std::random_device{}()});

        set_2 = set_1;

        for (int i = 1; i < world.size; i++) {
            range processPart = get_range(set_2, world.size, i); // получения диапазона сортировки для процесса
            int size = processPart.len;

            mpi_send(&size, 1, i); // указание дочернему процессу длины диапазона для сортировки
            mpi_send(set_2.data() + processPart.index, size, i); // передача дочернему процессу данных для сортировки
        }
    } else {
        int len;
        mpi_recv(&len, 1); // получение от родителя длины диапазона

        set_2.resize(len); // выделение памяти для сортировки своего диапазона
        mpi_recv(set_2.data(), len); // получение данных для сортировки
    }

    hrc::time_point start;
    fdur time_default, time_mpi;

    // rank 0 - main process; sequential
    if (world.rank == 0) {
        start = hrc::time_point{hrc::now()};

        merge_sort(set_1, { 0, (size_t)size });

        time_default = chrono::duration_cast<fdur>(hrc::now() - start);
    }

    // parallel
    MPI_Barrier(MPI_COMM_WORLD);
    start = hrc::time_point{hrc::now()};

    merge_sort_parallel(set_2, world);
    time_mpi = chrono::duration_cast<fdur>(hrc::now() - start);

    if (world.rank == 0) {
        save(OUTPUT, {"default", time_default}, set_1, "w");
        save(OUTPUT, {"mpi", time_mpi}, set_2);
    }

    MPI_Finalize();
    return 0;
}

