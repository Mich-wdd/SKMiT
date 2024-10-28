#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>
#include <omp.h>

using namespace std;

struct Point {
    double x;
    double y;
};

void printPoint(Point p) {
    cout << "x: " << p.x << " y: " << p.y << endl;
}

struct Cell {
    Point uL;
    Point dR;
};

struct GridSize {
    int m;
    int n;
};

void printCell(Cell cell) {
    printPoint(cell.uL);
    printPoint(cell.dR);
}

const int iterationsNumber = 1000;
const double A1 = -4, B1 = 4, A2 = -1, B2 = 5;
double delta = 0.000001;

const Point A = { 3.0, 0.0 };
const Point B = { 0.0, 4.0 };
const Point C = { -3.0, 0.0 };

void printTime(std::chrono::steady_clock::time_point start, GridSize gridSize) {
    auto diff = chrono::steady_clock::now() - start;
    cout << endl << "The final time for the size = " << gridSize.m << ", " << gridSize.n << ": " << (double)chrono::duration <double, milli>(diff).count() / 60000.0 << " ms" << endl;
}

double yTriangle(const double x) {
    return (-4.0 / 3.0) * abs(x) + 4;
}

bool isPointInTriangle(const Point P) {
    if (P.x > A.x || P.x < C.x || P.y > B.y || P.y < A.y) return false;
    if (yTriangle(P.x) < P.y) return false;
    return true;
}

double getRandomDouble(const double min, const double max) {
    std::default_random_engine generator;
    std::uniform_real_distribution<> distribution(min, max);
    return distribution(generator);
}

double getCoefA(const vector<vector<Point>>& grid, const int i, const int j, const double h, const double eps) {
    Point point1 = { grid[i][j].x - h * 0.5, grid[i][j].y - h * 0.5 };
    Point point2 = { grid[i][j].x - h * 0.5, grid[i][j].y + h * 0.5 };
    if (point1.y > B.y || point2.y < A.y) return 1 / eps;
    int counter = 0;
    int k;
#pragma omp parallel for reduction(+:counter) private(k) schedule(static)
    for (k = 0; k < iterationsNumber; k++) {
        double yk = getRandomDouble(point1.y, point2.y);
        if (isPointInTriangle(Point{ point1.x, yk })) counter++;
    }
    if (counter == iterationsNumber) return 1;
    double l = h * counter / iterationsNumber;
    return (l / h) + ((1 - l / h) / eps);
}

double getCoefB(const vector<vector<Point>>& grid, const int i, const int j, const double h, const double eps) {
    Point point1 = { grid[i][j].x - h * 0.5, grid[i][j].y - h * 0.5 };
    Point point2 = { grid[i][j].x + h * 0.5, grid[i][j].y - h * 0.5 };
    if (point1.x > A.x || point2.x < C.x) return 1 / eps;
    int counter = 0;
    int k;
#pragma omp parallel for reduction(+:counter) private(k) schedule(static)
    for (k = 0; k < iterationsNumber; k++) {
        double x_t = getRandomDouble(point1.x, point2.x);
        if (isPointInTriangle(Point{ x_t, point1.y })) counter++;
    }
    if (counter == iterationsNumber) return 1;
    double l = h * counter / iterationsNumber;
    return (l / h) + ((1 - l / h) / eps);
}

double getClosureArea(const Cell cell, const double h1, const double h2) {
    if (cell.uL.x > A.x || cell.dR.x < C.x || cell.uL.y < A.y || cell.dR.y > B.y) return 0.0;
    int counter = 0;
    int k;
#pragma omp parallel for reduction(+:counter) private(k) schedule(static)
    for (k = 0; k < iterationsNumber; k++) {
        double xk = getRandomDouble(cell.uL.x, cell.dR.x);
        double yk = getRandomDouble(cell.dR.y, cell.uL.y);
        if (isPointInTriangle(Point{ xk, yk })) {
            counter++;
        }
    }
    return h1 * h2 * (counter / iterationsNumber);
}

double getScalar(const vector<double>& vector1, const vector<double>& vector2, const double h1, const double h2) {
    double scalar = 0;
    int i;
#pragma omp parallel for reduction(+:scalar) private(i) shared(vector1, vector2) schedule(static)
    for (i = 0; i < vector1.size(); i++) {
        scalar += vector1[i] * vector2[i];
    }
    return h1 * h2 * scalar;
}

double getNorm(const vector<double>& vector, const double h1, const double h2) {
    return sqrt(getScalar(vector, vector, h1, h2));
}

vector<double> subtraction(const vector<double>& decrease, const vector<double>& subtract) {
    vector<double> difference(decrease.size());
    int i;
#pragma omp parallel for private(i) shared(difference, decrease, subtract) schedule(static)
    for (i = 0; i < decrease.size(); i++) {
        difference[i] = decrease[i] - subtract[i];
    }
    return difference;
}

vector<double> multiply(vector<double>& vector, const double tau) {
    int i;
#pragma omp parallel for private(i) shared(vector) schedule(static)
    for (i = 0; i < vector.size(); i++) {
        vector[i] *= tau;
    }
    return vector;
}

void doOperator(const vector<vector<double>>& mathOperator, const vector<double>& w, vector<double>& result, const GridSize gridSize) {
    int i;
#pragma omp parallel for private(i) shared(mathOperator, w, result) schedule(static)
    for (i = gridSize.n + 2; i < mathOperator.size() - gridSize.n - 2; i++) {
        result[i] = mathOperator[i][i] * w[i]
            + mathOperator[i][i - 1] * w[i - 1]
            + mathOperator[i][i + 1] * w[i + 1]
            + mathOperator[i][i - gridSize.n - 1] * w[i - gridSize.n - 1]
            + mathOperator[i][i + gridSize.n + 1] * w[i + gridSize.n + 1];
    }
    return;
}

vector<double> fastestDescent(const vector<vector<double>>& operatorA, const vector<double>& F, const double delta, const double h1, const double h2, const GridSize gridSize) {
    int counter = 0;
    int matrixSize = (int)F.size();
    vector<double> wK(matrixSize, 0.0);
    vector<double> wKp1(matrixSize, 0.0);
    vector<double> rK(matrixSize);
    vector<double> ArK(matrixSize);
    vector<double> wKDiff(matrixSize);

    double tau, ArKNorm, wKDiffNorm;

    do {
        doOperator(operatorA, wK, rK, gridSize);
        rK = subtraction(rK, F);
        doOperator(operatorA, rK, ArK, gridSize);
        ArKNorm = getNorm(ArK, h1, h2);
        tau = getScalar(ArK, rK, h1, h2) / pow(ArKNorm, 2);
        multiply(rK, tau);
        wKp1 = subtraction(wK, rK);
        wKDiff = subtraction(wKp1, wK);
        wKDiffNorm = getNorm(wKDiff, h1, h2);
        if (counter % 500 == 0) {
            cout << "count: " << counter << ", diffNorm: " << wKDiffNorm << ", delta: " << delta << endl;
        }
        if (wKDiffNorm < delta) {
            cout << "Stop on: " << counter << ", diffNorm: " << wKDiffNorm << endl;
            break;
        }
        else wK = wKp1;
        counter++;
    } while (true);

    return wKp1;
}

vector<vector<Point>> createGrid(const GridSize gridSize, const double h1, const double h2) {
    vector<vector<Point>> grid(gridSize.m + 1, vector<Point>(gridSize.n + 1, Point{ 0.0, 0.0 }));
    int i, j;
#pragma omp parallel for private(i, j) shared(grid) schedule(static) collapse(2)
    for (i = 0; i < gridSize.m + 1; i++) {
        for (j = 0; j < gridSize.n + 1; j++) {
            grid[i][j] = Point{ A1 + i * h1, A2 + j * h2 };
        }
    }
    return grid;
}

vector<double> createF(const vector<vector<Point>>& grid, const GridSize gridSize, const double h1, const double h2) {
    vector<double> F((gridSize.m + 1) * (gridSize.n + 1), 0.0);
    int i, j;
#pragma omp parallel for private(i, j) shared(F, grid) schedule(static) collapse(2)
    for (i = 1; i < gridSize.m; i++) {
        for (j = 1; j < gridSize.n; j++) {
            Point up_left = Point{ grid[i][j].x - h1 / 2, grid[i][j].y + h2 / 2 };
            Point down_right = Point{ grid[i][j].x + h1 / 2, grid[i][j].y - h2 / 2 };
            Cell rect = { up_left, down_right };
            F[i * (gridSize.n + 1) + j] = getClosureArea(rect, h1, h2) / (h1 * h2);
        }
    }
    return F;
}

vector<vector<double>> createOperator(const vector<vector<Point>>& grid, const GridSize gridSize, const double h1, const double h2, const double eps) {
    vector<vector<double>> operatorA((gridSize.m + 1) * (gridSize.n + 1), vector<double>((gridSize.m + 1) * (gridSize.n + 1), 0.0));
    int i, j;
#pragma omp parallel for private(i, j) shared(operatorA, grid) schedule(static) collapse(2)
    for (i = 1; i < gridSize.m; i++) {
        for (j = 1; j < gridSize.n; j++) {
            double aIp1J = getCoefA(grid, i + 1, j, h2, eps);
            double aIJ = getCoefA(grid, i, j, h2, eps);
            double bIJp1 = getCoefB(grid, i, j + 1, h1, eps);
            double bIJ = getCoefB(grid, i, j, h1, eps);
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
    return operatorA;
}

double calculateEpsilon(const double h1, const double h2) {
    if (h1 > h2) return pow(h1, 2);
    else return pow(h2, 2);
}

double calculateH(const double B, const double A, const int gridSize) {
    return (B - A) / gridSize;
}

int main()
{
    vector<GridSize> gridsSizes = { GridSize{160, 180} };

    int num_of_threads = 32;
    omp_set_num_threads(num_of_threads);

    for (int i = 0; i < gridsSizes.size(); i++) {
        GridSize gridSize = gridsSizes[i];
        const double h1 = calculateH(B1, A1, gridSize.m);
        const double h2 = calculateH(B2, A2, gridSize.n);
        const double eps = calculateEpsilon(h1, h2);

        cout << "Start for grid size: " << gridSize.m << ", " << gridSize.n << endl;
        auto start = chrono::steady_clock::now();

        vector<vector<Point>> grid = createGrid(gridSize, h1, h2);

        cout << "Grid successfully created" << endl;

        vector<double> F = createF(grid, gridSize, h1, h2);

        cout << "Right part F successfully created" << endl;

        vector <vector<double>> operatorA = createOperator(grid, gridSize, h1, h2, eps);

        cout << "Operator A successfully created" << endl << endl;

        delta = 0.000002;
        auto result = fastestDescent(operatorA, F, delta, h1, h2, gridSize);

        printTime(start, gridSize);
    }
    return 0;
}