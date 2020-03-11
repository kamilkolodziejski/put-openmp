#include <omp.h>
#include <stdio.h>
#include <string>
#include <omp.h>
#include <limits>
#include <time.h>
#include <fstream>
#include <cstdio>

// ustawienie maksymalnej wartosci int 
const int N = std::numeric_limits<int>::max() - 1;
// liczba watkow 
const int NUM_THREADS = 4;
// koniec zakresu znajdywania liczb pierwszych
const int MAX = (int)(N / 2);
// poczatek zakresu znajdywania liczb pierwszych
const int MIN = 2;
// rozmiar potrzebnej tablicy
const int targetArrayCount = MAX + 1;

// tablica sluzy do przechowywania 
bool** privPrimes = new bool* [NUM_THREADS];

class MeasuredSieveOfEratosthenes
{
protected:
    virtual void Calculate(bool* isPrimeNumber) = 0;
public:
    void MarkPrimitivesAndPrintTime(bool* isPrimeNumber)
    {
        clock_t start, end;

        start = clock();
        Calculate(isPrimeNumber);
        end = clock();

        printf("%s - %.4f\n", GetName().c_str(), (((double)end - (double)start) / 1000.0));
    }

    virtual std::string GetName() = 0;
};

class SingleThreadSieveOfEratosthenes : public MeasuredSieveOfEratosthenes
{
protected:
    void Calculate(bool* isPrimeNumber);
public:
    std::string GetName()
    {
        return "Single thread Sieve of Eratosthenes";
    }
};

class ParallelMarkMultiplesOfPrimes : public MeasuredSieveOfEratosthenes
{
protected:
    void Calculate(bool* isPrimeNumber);
public:
    std::string GetName()
    {
        return "Parallel Sieve of Eratosthenes";
    }
};

class ParallelMarkMultiplesOfPrimesNoWaitThreads : public MeasuredSieveOfEratosthenes
{
protected:
    void Calculate(bool* isPrimeNumber);
public:
    std::string GetName()
    {
        return "Parallel Sieve of Eratosthenes with nowait";
    }
};

class ParallelMarkMultiplesOfPrimesWithPrivateArray : public MeasuredSieveOfEratosthenes
{
protected:
    void Calculate(bool* isPrimeNumber);
public:
    std::string GetName()
    {
        return "Parallel Sieve of Eratosthenes with private array";
    }
};

void FillWithTrue(bool* array);

void PrintPrimeNumbers(bool* isPrimeNumber);

bool CompareArrays(bool* a, bool* b);

void WriteResultToFile(std::string filename, bool* primesArray);

int main()
{
    omp_set_num_threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; ++i)
        privPrimes[i] = new bool[targetArrayCount];

    bool* arr = new bool[targetArrayCount];

    printf("Max = %d\n", MAX);

    // petla do wielokrotnego uruchamiania obliczen - aktualnie ustawiona na 1 wykonanie
    for (int x = 0; x < 1; ++x) {

        FillWithTrue(arr);
        // algorytm opisywany w sprawozdaniu, reszta implementacji w celach pogladowych
        ParallelMarkMultiplesOfPrimes().MarkPrimitivesAndPrintTime(arr);

        //for (int i = 0; i < NUM_THREADS; ++i)
        //    FillWithTrue(privPrimes[i]);
        //FillWithTrue(arr);
        //ParallelMarkMultiplesOfPrimesWithPrivateArray().MarkPrimitivesAndPrintTime(arr);
        //FillWithTrue(arr);
        //ParallelMarkMultiplesOfPrimesNoWaitThreads().MarkPrimitivesAndPrintTime(arr);

        //writeResultToFile("output.txt", b);

        //printf("\n%s\n\n", compareArrays(a, b) ? "True" : "False");
    }


    for (int i = 0; i < NUM_THREADS; ++i)
        delete[] privPrimes[i];
    delete[] privPrimes;
    delete[] arr;
    return 0;
}

void SingleThreadSieveOfEratosthenes::Calculate(bool* isPrimeNumber)
{
    for (int i = 2; i <= sqrt(MAX); ++i)
        if (isPrimeNumber[i])
        {
            int x = i * i;
            while (x <= MAX)
            {
                isPrimeNumber[x] = false;
                x += i;
            }
        }
}

void ParallelMarkMultiplesOfPrimes::Calculate(bool* isPrimeNumber)
{
    const int n = (int)sqrt(MAX);
    int i;
    int* primes = new int[n];
    int primesCounter = 0;
    for (i = 2; i <= n; ++i)
        if (isPrimeNumber[i])
        {
            for (int j = i * i; j <= n; j += i)
                isPrimeNumber[j] = false;

            primes[primesCounter++] = i;
        }

#pragma omp parallel for schedule(static, 1)
    for (i = 0; i < primesCounter; ++i)
    {
        int p = primes[i];
        for (int j = p * p; j <= MAX; j += p)
            if (j >= MIN)
                if (isPrimeNumber[j])
                    isPrimeNumber[j] = false;
    }

}

void ParallelMarkMultiplesOfPrimesNoWaitThreads::Calculate(bool* isPrimeNumber)
{
    const int n = (int)sqrt(MAX);
    int i;
    int* primes = new int[n];
    int primesCounter = 0;
    for (i = 2; i <= n; ++i)
        if (isPrimeNumber[i])
        {
            for (int j = i * i; j <= n; j += i)
                isPrimeNumber[j] = false;

            primes[primesCounter++] = i;
        }

#pragma omp parallel  
    {
#pragma omp for nowait private(i) schedule(static, 1)
        for (i = 0; i < primesCounter; ++i)
        {
            int p = primes[i];
            for (int j = p * p; j <= MAX; j += p)
                if (j >= MIN && isPrimeNumber)
                    isPrimeNumber[j] = false;
        }
    }
}

void ParallelMarkMultiplesOfPrimesWithPrivateArray::Calculate(bool* isPrimeNumber)
{
    const int n = (int)sqrt(MAX);

    int i;
    int* primes = new int[n];
    int primesCounter = 0;
    for (i = 2; i <= n; ++i)
        if (isPrimeNumber[i])
        {
            for (int j = i * i; j <= n; j += i)
                isPrimeNumber[j] = false;
            primes[primesCounter++] = i;
        }

#pragma omp parallel// private(i)
    {
        int thread = omp_get_thread_num();
#pragma omp for schedule(static, 1) 
        for (i = 0; i < primesCounter; ++i)
        {
            int p = primes[i];
            for (int j = p * p; j <= MAX; j += p)
                if (j >= MIN)
                    if (privPrimes[thread][j])
                        privPrimes[thread][j] = false;
        }

        #pragma omp single
        for (i = 0; i < targetArrayCount; ++i)
            for (int j = 0; j < NUM_THREADS; ++j)
                isPrimeNumber[i] = isPrimeNumber[i] && privPrimes[j][i];
                
    }

}

void FillWithTrue(bool* array)
{
    for (long i = 2; i < MAX; ++i)
    {
        array[i] = true;
    }
}

void PrintPrimeNumbers(bool* isPrimeNumber)
{
    for (long i = 2; i <= MAX; ++i)
    {
        if (isPrimeNumber[i])
            printf("%d ", i);
    }
    printf("\n");
}

bool CompareArrays(bool* a, bool* b)
{
    for (int i = 0; i <= MAX; ++i)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

void WriteResultToFile(std::string filename, bool* primesArray)
{
    FILE* file;
    if (fopen_s(&file, filename.c_str(), "w") != 0)
    {
        printf("Error while opening %s file\n", filename.c_str());
        return;
    }
    else
        if (file != 0)
        {
            int primesCounter = 0;
            for (int i = MIN; i <= MAX; ++i)
                if (primesArray[i])
                    primesCounter++;

            fprintf_s(file, "%d %d %d\n", MIN, MAX, primesCounter);

            int printNumbersCounter = 0;
            for (int i = MIN; i <= MAX; ++i)
            {
                if (primesArray[i])
                {
                    fprintf_s(file, "%d ", i);
                    if (++printNumbersCounter % 10 == 0)
                        fprintf(file, "\n");
                }
            }
            fclose(file);
        }
}