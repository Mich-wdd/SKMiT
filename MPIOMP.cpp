#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>
#include <mpi.h>
#include <omp.h>

using namespace std;

struct Point {
    double x;
    double y;
};

struct Cell {
    Point uL;
    Point dR;
};

struct GridSize {
    int m;
    int n;
};

const int M = 160, N = 180;

const int iterationsNumber = 1000;
const double A1 = -5, B1 = 5, A2 = -2, B2 = 6;
double delta = 0.000001;

const Point A = { 3.0, 0.0 };
const Point B = { 0.0, 4.0 };
const Point C = { -3.0, 0.0 };

double calculateEpsilon(const double h1, const double h2) {
    if (h1 > h2) return pow(h1, 2);
    else return pow(h2, 2);
}

double calculateH(const double B, const double A, const int gridSize) {
    return (B - A) / gridSize;
}

void printPoint(Point p) {
    cout << "[" << p.x << ", " << p.y << "] ";
}

void printGrid(vector<Point>& grid, GridSize& gridSize) {
    for (int i = 0; i < (gridSize.m + 1) * (gridSize.n + 1); i++) {
        printPoint(grid[i]);
        cout << endl;
    }
    return;
}

void print_vector_double(const vector<double>& vector, const GridSize& gridSize) {
    for (int i = 0; i < (gridSize.m + 1) * (gridSize.n + 1); i++) {
        cout << vector[i] << " ";
        if ((i + 1) % (gridSize.m + 1) == 0) cout << endl;
    }
    return;
}

void print_operatorA(double(&operatorA)[(M + 1) * (N + 1)][(M + 1) * (N + 1)], const GridSize& gridSize) {
    for (int i = 0; i < (gridSize.m + 1) * (gridSize.n + 1); i++) {
        for (int j = 0; j < (gridSize.m + 1) * (gridSize.n + 1); j++) {
            cout << operatorA[i][j] << " ";
        }
        cout << endl;
    }
    return;
}

void createGrid(vector<Point>& grid, const GridSize& gridSize, const int& world_size, const int& world_rank, const double& h1, const double& h2) {
    int total_rows = gridSize.m + 1;
    int total_coloms = gridSize.n + 1;
    int usual_rows_per_process = total_rows / world_size;
    int rows_per_process = usual_rows_per_process;
    int remainder = total_rows % world_size;

    //Обработка последнего процесса
    if (world_rank == world_size - 1) {
        rows_per_process += remainder; // добавляем остаток
    }

    int start_row = world_rank * (total_rows / world_size);

    vector<Point> local_grid(rows_per_process * total_coloms);

    vector<int> recvcounts(world_size);
    vector<int> displs(world_size);

    // Инициализация recvcounts и displs
    for (int i = 0; i < world_size; i++) {
        recvcounts[i] = (i == world_size - 1) ? (usual_rows_per_process + remainder) * total_coloms * sizeof(Point) : usual_rows_per_process * total_coloms * sizeof(Point);
        displs[i] = i * usual_rows_per_process * total_coloms * sizeof(Point);
    }

#pragma omp parallel for
    for (int i = 0; i < rows_per_process; i++) {
        cout << "world_rank: " << world_rank << ", omp_get_thread_num() :" << omp_get_thread_num() << endl;
        int index = start_row + i;
        for (int j = 0; j < total_coloms; j++) {
            local_grid[i * total_coloms + j] = Point{ A1 + index * h1, A2 + j * h2 };
        }
    }

    MPI_Gatherv(local_grid.data(), rows_per_process * total_coloms * sizeof(Point), MPI_BYTE,
        grid.data(), recvcounts.data(), displs.data(), MPI_BYTE,
        0, MPI_COMM_WORLD);
}

double yTriangle(const double x) {
    return (-4.0 / 3.0) * abs(x) + 4;
}

bool isPointInTriangle(const Point P) {
    if (P.x > A.x || P.x < C.x || P.y > B.y || P.y < A.y) return false;
    if (yTriangle(P.x) < P.y) return false;
    return true;
}

double getRandomDouble(std::mt19937& generator, const double min, const double max) {
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(generator);
}

double getClosureArea(Cell& cell, const double& h1, const double& h2) {
    if (cell.uL.x > A.x || cell.dR.x < C.x || cell.uL.y < A.y || cell.dR.y > B.y) return 0.0;
    if (yTriangle(cell.uL.x) < cell.uL.y && yTriangle(cell.dR.x) < cell.dR.y) return 0.0;
    int counter = 0;
    std::random_device rd;
    std::mt19937 generator(rd());
#pragma omp parallel for reduction(+:counter)
    for (int k = 0; k < iterationsNumber; k++) {
        double xk = getRandomDouble(generator, cell.uL.x, cell.dR.x);
        double yk = getRandomDouble(generator, cell.uL.y, cell.dR.y);
        if (isPointInTriangle(Point{ xk, yk })) {
            counter++;
        }
    }
    if (counter == iterationsNumber) return h1 * h2;
    return h1 * h2 * counter / iterationsNumber;
}

void createF(vector<double>& F, const vector<Point>& local_grid, const GridSize& local_grid_size, const GridSize& grid_size, const int& world_size, const int& world_rank, const double& h1, const double& h2) {
    int local_rows = local_grid_size.m + 1;
    int local_coloms = local_grid_size.n + 1;

    int total_rows = grid_size.m + 1;
    int total_coloms = grid_size.n + 1;
    int usual_rows_per_process = total_rows / world_size;
    int rows_per_process = usual_rows_per_process;
    int remainder = total_rows % world_size;

    if (world_rank == world_size - 1) {
        rows_per_process += remainder; // добавляем остаток
    }

    vector<int> recvcounts(world_size);
    vector<int> displs(world_size);
    for (int i = 0; i < world_size; i++) {
        recvcounts[i] = (i == world_size - 1) ? (usual_rows_per_process + remainder) * total_coloms : usual_rows_per_process * total_coloms;
        displs[i] = i * usual_rows_per_process * total_coloms;
    }

    vector<double> local_F(total_rows * total_coloms);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < local_coloms; j++) {
            Point up_left = Point{ local_grid[i * local_coloms + j].x - h1 / 2, local_grid[i * local_coloms + j].y + h2 / 2 };
            Point down_right = Point{ local_grid[i * local_coloms + j].x + h1 / 2, local_grid[i * local_coloms + j].y - h2 / 2 };
            Cell rect = { up_left, down_right };
            local_F[i * local_coloms + j] = getClosureArea(rect, h1, h2) / (h1 * h2);
        }
    }

    MPI_Gatherv(local_F.data(), local_rows * local_coloms, MPI_DOUBLE,
        F.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);
}

double getCoefA(const Point& p, const double h, const double eps) {
    Point point1 = { p.x - h * 0.5, p.y - h * 0.5 };
    Point point2 = { p.x - h * 0.5, p.y + h * 0.5 };
    if (point1.y > B.y || point2.y < A.y) return 1 / eps;
    int counter = 0;
    std::random_device rd;
    std::mt19937 generator(rd());
#pragma omp parallel for reduction(+:counter)
    for (int k = 0; k < iterationsNumber; k++) {
        double yk = getRandomDouble(generator, point1.y, point2.y);
        if (isPointInTriangle(Point{ point1.x, yk })) counter++;
    }
    if (counter == iterationsNumber) return 1;
    double l = h * counter / iterationsNumber;
    return (l / h) + ((1 - l / h) / eps);
}

double getCoefB(const Point& p, const double h, const double eps) {
    Point point1 = { p.x - h * 0.5, p.y - h * 0.5 };
    Point point2 = { p.x + h * 0.5, p.y - h * 0.5 };
    if (point1.x > A.x || point2.x < C.x) return 1 / eps;
    int counter = 0;
    std::random_device rd;
    std::mt19937 generator(rd());
#pragma omp parallel for reduction(+:counter)
    for (int k = 0; k < iterationsNumber; k++) {
        double x_t = getRandomDouble(generator, point1.x, point2.x);
        if (isPointInTriangle(Point{ x_t, point1.y })) counter++;
    }
    if (counter == iterationsNumber) return 1;
    double l = h * counter / iterationsNumber;
    return (l / h) + ((1 - l / h) / eps);
}

void createOperator(double(&operatorA)[(M + 1) * (N + 1)][(M + 1) * (N + 1)], const vector<Point>& grid, const GridSize& gridSize, const double& h1, const double& h2, const double& eps) {
#pragma omp parallel for collapse(2)
    for (int i = 1; i < gridSize.m; i++) {
        for (int j = 1; j < gridSize.n; j++) {
            Point grid_point_Ip1J = grid[(i + 1) * (gridSize.n + 1) + j];
            Point grid_point_IJ = grid[i * (gridSize.n + 1) + j];
            Point grid_point_IJp1 = grid[i * (gridSize.n + 1) + (j + 1)];
            double aIp1J = getCoefA(grid_point_Ip1J, h2, eps);
            double aIJ = getCoefA(grid_point_IJ, h2, eps);
            double bIJp1 = getCoefB(grid_point_IJp1, h1, eps);
            double bIJ = getCoefB(grid_point_IJ, h1, eps);
            operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j] = (aIp1J + aIJ) / pow(h1, 2) + (bIJp1 + bIJ) / pow(h2, 2);
            if (i == 1) {
                operatorA[i * (gridSize.n + 1) + j][(i + 1) * (gridSize.n + 1) + j] = -aIp1J / pow(h1, 2);
                if (j == 1) {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j + 1] = -bIJp1 / pow(h2, 2);
                }
                else if (j == gridSize.n - 1) {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j - 1] = -bIJ / pow(h2, 2);
                }
                else {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j - 1] = -bIJ / pow(h2, 2);
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j + 1] = -bIJp1 / pow(h2, 2);
                }
            }
            else if (i == gridSize.m - 1) {
                operatorA[i * (gridSize.n + 1) + j][(i - 1) * (gridSize.n + 1) + j] = -aIJ / pow(h1, 2);
                if (j == 1) {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j + 1] = -bIJp1 / pow(h2, 2);
                }
                else if (j == gridSize.n - 1) {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j - 1] = -bIJ / pow(h2, 2);
                }
                else {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j - 1] = -bIJ / pow(h2, 2);
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j + 1] = -bIJp1 / pow(h2, 2);
                }
            }
            else {
                operatorA[i * (gridSize.n + 1) + j][(i - 1) * (gridSize.n + 1) + j] = -aIJ / pow(h1, 2);
                operatorA[i * (gridSize.n + 1) + j][(i + 1) * (gridSize.n + 1) + j] = -aIp1J / pow(h1, 2);
                if (j == 1) {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j + 1] = -bIJp1 / pow(h2, 2);
                }
                else if (j == gridSize.n - 1) {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j - 1] = -bIJ / pow(h2, 2);
                }
                else {
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j - 1] = -bIJ / pow(h2, 2);
                    operatorA[i * (gridSize.n + 1) + j][i * (gridSize.n + 1) + j + 1] = -bIJp1 / pow(h2, 2);
                }
            }
        }
    }
}



double getScalar(const vector<double>& vector1, const vector<double>& vector2, const double h1, const double h2) {
    double scalar = 0;
#pragma omp parallel for reduction(+:scalar)
    for (int i = 0; i < vector1.size(); i++) {
        scalar += vector1[i] * vector2[i];
    }
    return h1 * h2 * scalar;
}

void getNorm(double& global_scalar, const vector<double>& vector, const double h1, const double h2) {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double local_scalar = getScalar(vector, vector, h1, h2);
    MPI_Reduce(&local_scalar, &global_scalar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        global_scalar = sqrt(global_scalar);
    }
    MPI_Bcast(&global_scalar, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void subtraction(vector<double>& difference, const vector<double>& decrease, const vector<double>& subtract) {
    int local_items = decrease.size();
#pragma omp parallel for
    for (int i = 0; i < local_items; i++) {
        difference[i] = decrease[i] - subtract[i];
    }
}

void doOperator(double(&mathOperator)[(M + 1) * (N + 1)][(M + 1) * (N + 1)], const vector<double>& w, vector<double>& result, const GridSize& gridSize) {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int total_items = (M + 1) * (N + 1);
    int usual_items_per_process = total_items / world_size;
    int items_per_process = usual_items_per_process;
    int remainder = total_items % world_size;
    if (world_rank == world_size - 1) {
        items_per_process += remainder; // добавляем остаток
    }

    vector<int> recvcounts(world_size);
    vector<int> displs(world_size);

    // Инициализация recvcounts и displs
    for (int i = 0; i < world_size; i++) {
        recvcounts[i] = (i == world_size - 1) ? (usual_items_per_process + remainder) : usual_items_per_process;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
    }

    int start = displs[world_rank];
#pragma omp parallel for
    for (int i = start; i < start + items_per_process; i++) {
        if (i < gridSize.n + 2) {
            result[i - start] = 0.0;
        }
        else {
            result[i - start] = mathOperator[i][i] * w[i]
                + mathOperator[i][i - 1] * w[i - 1]
                + mathOperator[i][i + 1] * w[i + 1]
                + mathOperator[i][i - gridSize.n - 1] * w[i - gridSize.n - 1]
                + mathOperator[i][i + gridSize.n + 1] * w[i + gridSize.n + 1];
        }
    }
    return;
}

vector<double> multiply(vector<double>& vector, const double tau) {
#pragma omp parallel for
    for (int i = 0; i < vector.size(); i++) {
        vector[i] *= tau;
    }
    return vector;
}

void printTime(double& start_time) {
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    cout << "Total time: " << elapsed_time << " sec" << endl;
    cout << "Total time: " << elapsed_time / 60.0 << " min" << endl;
}

void fastestDescent(vector<double>& result, double(&operatorA)[(M + 1) * (N + 1)][(M + 1) * (N + 1)], const vector<double>& F, const double delta, const double h1, const double h2, const GridSize& gridSize) {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int counter;
    if (world_rank == 0) counter = 0;

    int total_items = (int)F.size();

    vector<double> wK(total_items, 0.0);
    vector<double> rK(total_items);

    double local_tau, ArKNorm, wKDiffNorm;

    double start_time;
    if (world_rank == 0) start_time = MPI_Wtime();


    int usual_items_per_process = total_items / world_size;
    int items_per_process = usual_items_per_process;
    int remainder = total_items % world_size;
    if (world_rank == world_size - 1) {
        items_per_process += remainder;
    }

    vector<double> local_rK(items_per_process);
    vector<double> local_wK(items_per_process);
    vector<double> local_ArK(items_per_process);
    vector<double> local_wKp1(items_per_process);
    vector<double> local_wKDiff(items_per_process);

    vector<double> local_F(items_per_process);

    vector<int> recvcounts(world_size);
    vector<int> displs(world_size);
    for (int i = 0; i < world_size; i++) {
        recvcounts[i] = (i == world_size - 1) ? (usual_items_per_process + remainder) : usual_items_per_process;
        displs[i] = i * usual_items_per_process;
    }

    MPI_Scatterv(F.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        local_F.data(), recvcounts[world_rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    do {

        doOperator(operatorA, wK, local_rK, gridSize);

        subtraction(local_rK, local_rK, local_F);

        MPI_Gatherv(local_rK.data(), items_per_process, MPI_DOUBLE,
            rK.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        MPI_Bcast(rK.data(), total_items, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        doOperator(operatorA, rK, local_ArK, gridSize);

        getNorm(ArKNorm, local_ArK, h1, h2);

        local_tau = getScalar(local_ArK, local_rK, h1, h2) / pow(ArKNorm, 2);
        double global_tau;
        if (world_rank == 0) global_tau = 0.0;
        MPI_Reduce(&local_tau, &global_tau, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        multiply(local_rK, global_tau);

        MPI_Scatterv(wK.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            local_wK.data(), recvcounts[world_rank], MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        subtraction(local_wKp1, local_wK, local_rK);
        subtraction(local_wKDiff, local_wKp1, local_wK);

        getNorm(wKDiffNorm, local_wKDiff, h1, h2);

        if (world_rank == 0) {
            if (counter % 500 == 0) {
                cout << "diffNorm: " << wKDiffNorm << ", delta: " << delta << endl;
            }
            if (counter % 5000 == 0) {
                printTime(start_time);
            }
        }

        if (wKDiffNorm < delta) {
            if (world_rank == 0) {
                cout << "Stop on: " << counter << endl;
                printTime(start_time);
            }
            break;
        }

        MPI_Gatherv(local_wKp1.data(), recvcounts[world_rank], MPI_DOUBLE,
            wK.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        MPI_Bcast(wK.data(), total_items, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == 0) counter++;

    } while (true);

    MPI_Gatherv(local_wKp1.data(), recvcounts[world_rank], MPI_DOUBLE,
        result.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    GridSize gridSize = { M, N };

    vector<Point> grid;
    vector<double> F((gridSize.m + 1) * (gridSize.n + 1));
    double operatorA[(M + 1) * (N + 1)][(M + 1) * (N + 1)];
    vector<double> result;

    double h1 = 0.0;
    double h2 = 0.0;
    double eps = 0.0;

    double start_time;

    int total_rows = gridSize.m + 1;
    int total_coloms = gridSize.n + 1;
    int rows_per_process = total_rows / world_size;
    int remainder = total_rows % world_size;

    if (world_rank == 0) {
        h1 = calculateH(B1, A1, gridSize.m);
        h2 = calculateH(B2, A2, gridSize.n);
        eps = calculateEpsilon(h1, h2);

        // Распространяем h1 и h2 на все процессы
        for (int i = 1; i < world_size; ++i) {
            MPI_Send(&h1, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&h2, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&eps, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        grid.resize((gridSize.m + 1) * (gridSize.n + 1));
        result.resize((gridSize.m + 1) * (gridSize.n + 1));
    }
    else {
        // Получаем h1 и h2 от главного процесса
        grid.resize((gridSize.m + 1) * (gridSize.n + 1));
        MPI_Recv(&h1, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&h2, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&eps, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    createGrid(grid, gridSize, world_size, world_rank, h1, h2);

    if (world_rank == 0) {
        cout << "Grid successfully created" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    vector<int> recvcounts(world_size);
    vector<int> displs(world_size);
    vector<int> itemcounts(world_size);
    for (int i = 0; i < world_size; i++) {
        itemcounts[i] = (i == world_size - 1) ? (rows_per_process + remainder) * total_coloms : rows_per_process * total_coloms;
        recvcounts[i] = itemcounts[i] * sizeof(Point);
        displs[i] = i * rows_per_process * total_coloms * sizeof(Point);
    }

    vector<Point> local_grid(itemcounts[world_rank]);

    MPI_Scatterv(grid.data(), recvcounts.data(), displs.data(), MPI_BYTE,
        local_grid.data(), recvcounts[world_rank], MPI_BYTE,
        0, MPI_COMM_WORLD);

    GridSize localGridSize = { (world_rank == world_size - 1) ? rows_per_process + remainder - 1 : rows_per_process - 1,  total_coloms - 1 };

    if (world_rank == 0) {
        cout << "start create F" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    createF(F, local_grid, localGridSize, gridSize, world_size, world_rank, h1, h2);

    if (world_rank == 0) {
        cout << "F successfully created" << endl;
    }

    MPI_Bcast(F.data(), (gridSize.m + 1) * (gridSize.n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "Start create Operator A" << endl << endl;
    }

    MPI_Bcast(grid.data(), (gridSize.m + 1) * (gridSize.n + 1) * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

    createOperator(operatorA, grid, gridSize, h1, h2, eps);

    cout << "Operator A successfully created on process: " << world_rank << endl << endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "Start for grid size: " << gridSize.m << ", " << gridSize.n << endl;
        start_time = MPI_Wtime();
    }

    double delta = 0.000003;
    fastestDescent(result, operatorA, F, delta, h1, h2, gridSize);

    if (world_rank == 0) {
        cout << "fastestDescent successfully " << endl << endl;
    }

    if (world_rank == 0) {
        printTime(start_time);
    }

    MPI_Finalize();

    return 0;
}