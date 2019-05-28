#include <omp.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <vector>

//#define MEM_STATIC

// Constants
typedef double my_type;
auto constexpr num_runs = 50;
auto constexpr x = 512;
auto constexpr n = x * x;
auto constexpr static_size = 64;

#ifdef MEM_STATIC
int initialize_array(my_type matrix[n][n])
#else
int initialize_array(my_type*& matrix)
#endif
{
	if (matrix == nullptr)
	{
		std::cout << "Could not allocate memory for matrix!";
		return 1;
	}

	for (auto i = 0; i < n; ++i)
			matrix[i] = rand() % 10000;

	return 0;
}

int main()
{
	srand(time(NULL));

#ifdef MEM_STATIC
	my_type matrix[n][n];
#else
	my_type* matrix = nullptr;
	matrix = (my_type*)malloc(sizeof(my_type) * n);
#endif

#pragma region Initialization
	{
		std::cout << "Initialization - ";

		const auto t0 = std::chrono::high_resolution_clock::now();
		if (initialize_array(matrix))
			return 1;

		const auto t1 = std::chrono::high_resolution_clock::now();
		const auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

		std::cout << "Init Duration =  " << duration_ms << "(ms)";
		std::cout << std::endl;
	}
#pragma endregion

	my_type sum = 0;
	my_type avg = 0;

	int num_threads = omp_get_max_threads();
	omp_set_num_threads(num_threads);

	std::cout << "---------------------------" << std::endl;

#pragma region Sequential
	{
		std::cout << "Sequential -		";

		double sum_duration = 0;

		for (auto r = 0; r < num_runs; ++r)
		{
			const auto t0 = std::chrono::high_resolution_clock::now();

			sum = 0;
			avg = 0;

			for (auto i = 0; i < n; ++i)
					sum += matrix[i];

			const auto t1 = std::chrono::high_resolution_clock::now();
			sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		}

		avg = sum / (n);

		const auto duration_ms = (sum_duration / num_runs) / 1000.0;
		std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

		std::cout << "Avg =  " << avg;
		std::cout << std::endl;
	}
#pragma endregion

	std::cout << "---------------------------" << std::endl;

#pragma region Simple Parallel

	{
		std::cout << "Simple Parallel -		";

		double sum_duration = 0;

		for (auto r = 0; r < num_runs; ++r)
		{
			const auto t0 = std::chrono::high_resolution_clock::now();

			sum = 0;
			avg = 0;

#pragma omp parallel 
			{
				const auto id = omp_get_thread_num();

				if (id == 0)
					num_threads = omp_get_num_threads();

				double _temp_sum = 0;
				for (int i = (id / num_threads) * n; i < ((id + 1) / num_threads) * n; ++i)
						_temp_sum += matrix[i];
#pragma omp atomic
				sum += _temp_sum;
			}

			avg = sum / (n);
			const auto t1 = std::chrono::high_resolution_clock::now();
			sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		}

		const auto duration_ms = (sum_duration / num_runs) / 1000.0;
		std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

		std::cout << "Avg =  " << avg;
		std::cout << std::endl;
	}

#pragma endregion

	std::cout << "---------------------------" << std::endl;

#pragma region Parallel For Reduction

	{
		std::cout << "Parallel For Reduction -		";

		double sum_duration = 0;

		for (auto r = 0; r < num_runs; ++r)
		{
			const auto t0 = std::chrono::high_resolution_clock::now();

			sum = 0;
			avg = 0;

#pragma omp parallel for reduction(+: sum)
			for (int i = 0; i < n; ++i)
					sum += matrix[i];

			avg = sum / (n);
			const auto t1 = std::chrono::high_resolution_clock::now();
			sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		}

		const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
		std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

		std::cout << "Avg =  " << avg;
		std::cout << std::endl;
	}

#pragma endregion

	std::cout << "---------------------------" << std::endl;

#pragma region Parallel Simple Array - False Sharing
	{
		std::cout << "Parallel For Simple Array - False Sharing -		";

		double sum_duration = 0;

		for (auto r = 0; r < num_runs; ++r)
		{
			const auto t0 = std::chrono::high_resolution_clock::now();

			sum = 0;
			avg = 0;

			auto sums = std::vector<my_type>(num_threads);

#pragma omp parallel 
			{
				const auto id = omp_get_thread_num();

				for (int i = (id / num_threads) * n; i < ((id + 1) / num_threads) * n; ++i)
						sums[id] += matrix[i];
			}


			for (auto s : sums) sum += s;

			avg = sum / (n);
			const auto t1 = std::chrono::high_resolution_clock::now();
			sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		}

		const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
		std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

		std::cout << "Avg =  " << avg;
		std::cout << std::endl;
	}

#pragma endregion

	std::cout << "---------------------------" << std::endl;

#pragma region Parallel For Reduction Static Schedule

	{
		std::cout << "Parallel For Reduction Static (" << static_size << ") - ";

		double sum_duration = 0;

		for (auto r = 0; r < num_runs; ++r)
		{
			const auto t0 = std::chrono::high_resolution_clock::now();

			sum = 0;
			avg = 0;

#pragma omp parallel for reduction(+: sum) schedule(static, static_size)
			for (int i = 0; i < n; ++i)
					sum += matrix[i];

			avg = sum / (n);
			const auto t1 = std::chrono::high_resolution_clock::now();
			sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		}

		const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
		std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

		std::cout << "Avg =  " << avg;
		std::cout << std::endl;
	}

#pragma endregion

	std::cout << "---------------------------" << std::endl;

#pragma region Parallel For Reduction Dynamic Schedule

	{
		std::cout << "Parallel For Reduction Dynamic -		";

		double sum_duration = 0;

		for (auto r = 0; r < num_runs; ++r)
		{
			const auto t0 = std::chrono::high_resolution_clock::now();

			sum = 0;
			avg = 0;

#pragma omp parallel for reduction(+: sum) schedule(dynamic)
			for (int i = 0; i < n; ++i)
					sum += matrix[i];

			avg = sum / (n);
			const auto t1 = std::chrono::high_resolution_clock::now();
			sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		}

		const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
		std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

		std::cout << "Avg =  " << avg;
		std::cout << std::endl;
	}

#pragma endregion

	std::cout << "---------------------------" << std::endl << std::endl;

	std::cout << "Max Threads Num = " << omp_get_max_threads() << std::endl << std::endl;
	std::cout << "Number of Threads = " << num_threads << std::endl << std::endl;
	std::cout << "2D Array Size = " << x << " * " << x << std::endl << std::endl;
	std::cout << "Number Of Each Method Run = " << num_runs << std::endl << std::endl;

	std::cout << "---------------------------" << std::endl;

	std::cout << "Profiling Finished Successfully";

#ifndef MEM_STATIC
	free(matrix);
#endif

	std::cin >> sum;

	return 0;
}