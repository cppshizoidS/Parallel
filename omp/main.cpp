#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include <omp.h>

struct match;
struct computing_result;

using matches      = std::array<match, 6>;
using compute_func = computing_result (*)(uint64_t);


struct match
{
    uint8_t  value;
    uint64_t count;
};

struct check_result
{
    bool  status;
    match value;
};

struct computing_result
{
    uint64_t value;
    double   time;
    match    degree;
};


matches
get_primes(uint64_t N)
{
    match p0{0, 0},
          p1{1, 0},
          p2{2, 0},
          p3{3, 0},
          p5{5, 0},
          p7{7, 0};

	for (; N > 0; N /= 10) {
		uint8_t dig = N % 10;

        if (dig == 0) {
            ++p0.count;
        } else if (dig == 1) {
            ++p1.count;
        } else if (dig == 2) {
            ++p2.count;
        } else if (dig == 3) {
            ++p3.count;
        } else if (dig == 4) {
            p2.count += 2;
        } else if (dig == 5) {
            ++p5.count;
        } else if (dig == 6) {
            ++p2.count;
            ++p3.count;
        } else if (dig == 7) {
            ++p7.count;
        } else if (dig == 8) {
            p2.count += 3;
        } else if (dig == 9) {
            p3.count += 2;
        }
    }

    return {p0, p1, p2, p3, p5, p7};
}

bool
compare_matches(const match& lhs, const match& rhs)
{
    if (rhs.count == 0) {
        return true;
    }
    if (lhs.count == 0) {
        return false;
    }
    return lhs.count < rhs.count;
}

check_result
check_primes_degree(const matches& primes)
{
    // если в числе хоть один ноль - условие не выполняется
    if (primes[0].count > 0) {
        return {false, {}};
    }

    // проверка на количество разнообразных цифр в числе:
    // если она одна, и она больше одного, то число является степенью
	uint8_t variants_count{};
	match rarest{}, degree{};

    for (const auto& prime : primes) {
        // если число встречалось, увеличиваем
        if (prime.count > 0) {
            variants_count++;
        }
    }

    rarest = *std::min_element(primes.begin(), primes.end(), compare_matches);

    // если какое-то из простых чисел встречалась один раз,
    // то заданное число не может быть степенью
	if (rarest.count == 1) {
        return {false, {}};
    }

    // если все простые числа в заданном числе одинаковые
	if (variants_count == 1) { 
        return {true, rarest};
	}

    degree = rarest;
    for (const auto& prime : primes) {
        if ((prime.count == 0) || (prime.value == rarest.value)) {
            continue;
        }

        if (prime.count % rarest.count > 0) {
            return {false, {}};
        } else {
            degree.value *= std::pow(prime.value, prime.count / rarest.count);
        }
    }

	return {true, degree};
}

check_result
check_number_degree(uint64_t N)
{ return check_primes_degree(get_primes(N)); }

computing_result
compute_linear(uint64_t N)
{
	double time = omp_get_wtime();

	for (uint64_t n = N + 1; n < ULLONG_MAX; ++n) {
        auto check = check_number_degree(n);

		if (check.status) {
			return {n, omp_get_wtime() - time, check.value};
		}
	}

	return {0, omp_get_wtime() - time};
}

computing_result
compute_parallel(uint64_t N)
{
	const uint8_t threads_count = 4;
	omp_set_num_threads(threads_count);

	bool found  = false;
	double time = omp_get_wtime();
	computing_result result{ULLONG_MAX, 0, {}};

#pragma omp parallel
    for (
        uint64_t n = N + 1 + omp_get_thread_num();
        !found;
        n += threads_count
    ) {
        auto check = check_number_degree(n);

        if (check.status) {
#pragma omp critical
            if (n < result.value) {
                result.value  = n;
                result.degree = check.value;
                found         = true;
            }
        }
    }

    result.time = omp_get_wtime() - time;

	return result;
}

int
main()
{
	char mode;
	uint64_t n;
    const char* label;
    compute_func f;

    printf(
        "Task #39\nType 0 to exit at anytime\n\n"
        "Enter N: "
    );

    std::cin >> n;

	while (true) {
        printf("mode ([s]equential/[p]arallel): ");
        std::cin >> mode;

        if (mode == '0') {
            printf("\nExiting…\n");
            return 0;
        } else if (mode == 's') {
            f     = compute_linear;
            label = "Sequential";
        } else if (mode == 'p') {
            f     = compute_parallel;
            label = "Parallel";
        }

        auto result   = f(n);
        match& degree = result.degree;

        printf(
            "\n[RESULTS]\n"
            "%s: %lu -> %d^%lu (%fs)\n\n",
            label, result.value, degree.value, degree.count, result.time
        );
	}
}

