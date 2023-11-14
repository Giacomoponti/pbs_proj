#include <vector>

const int U_FIELD = 0;
const int V_FIELD = 1;
const int S_FIELD = 2;
class Fluid {
public:
    Fluid(float density, int numX, int numY, float h)
        : density(density), numX(numX + 2), numY(numY + 2), numCells(numX * numY),
          h(h), u(numCells), v(numCells), newU(numCells), newV(numCells),
          p(numCells), s(numCells), m(numCells), newM(numCells), overRelaxation(1){
        m.assign(numCells, 1.0);
    }

    void integrate(float dt, float gravity) {
        int n = numY;
        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0)
                    v[i * n + j] += gravity * dt;
            }
        }
    }

    void solveIncompressibility(int numIters, float dt) {
        int n = numY;
        float cp = density * h / dt;

        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 1; i < numX - 1; i++) {
                for (int j = 1; j < numY - 1; j++) {
                    if (s[i * n + j] == 0.0)
                        continue;

                    float sx0 = s[(i - 1) * n + j];
                    float sx1 = s[(i + 1) * n + j];
                    float sy0 = s[i * n + j - 1];
                    float sy1 = s[i * n + j + 1];
                    float s_sum = sx0 + sx1 + sy0 + sy1;

                    if (s_sum == 0.0)
                        continue;

                    float div = u[(i + 1) * n + j] - u[i * n + j] +
                                v[i * n + j + 1] - v[i * n + j];

                    float p_val = -div / s_sum;
                    p_val *= overRelaxation;
                    p[i * n + j] += cp * p_val;

                    u[i * n + j] -= sx0 * p_val;
                    u[(i + 1) * n + j] += sx1 * p_val;
                    v[i * n + j] -= sy0 * p_val;
                    v[i * n + j + 1] += sy1 * p_val;
                }
            }
        }
    }

    void extrapolate() {
        int n = numY;
        for (int i = 0; i < numX; i++) {
            u[i * n + 0] = u[i * n + 1];
            u[i * n + numY - 1] = u[i * n + numY - 2];
        }
        for (int j = 0; j < numY; j++) {
            v[0 * n + j] = v[1 * n + j];
            v[(numX - 1) * n + j] = v[(numX - 2) * n + j];
        }
    }

    float sampleField(float x, float y, int field) {
        int n = numY;
        float h1 = 1.0 / h;
        float h2 = 0.5 * h;

        x = std::max(std::min(x, numX * h), h);
        y = std::max(std::min(y, numY * h), h);

        float dx = 0.0;
        float dy = 0.0;

        std::vector<float>* f;

        switch (field) {
        case U_FIELD:
            f = &u;
            dy = h2;
            break;
        case V_FIELD:
            f = &v;
            dx = h2;
            break;
        case S_FIELD:
            f = &m;
            dx = h2;
            dy = h2;
            break;
        }

        int x0 = std::min(static_cast<int>(floor((x - dx) * h1)), numX - 1);
        float tx = ((x - dx) - x0 * h) * h1;
        int x1 = std::min(x0 + 1, numX - 1);

        int y0 = std::min(static_cast<int>(floor((y - dy) * h1)), numY - 1);
        float ty = ((y - dy) - y0 * h) * h1;
        int y1 = std::min(y0 + 1, numY - 1);

        float sx = 1.0 - tx;
        float sy = 1.0 - ty;

        float val = sx * sy * (*f)[x0 * n + y0] +
                    tx * sy * (*f)[x1 * n + y0] +
                    tx * ty * (*f)[x1 * n + y1] +
                    sx * ty * (*f)[x0 * n + y1];

        return val;
    }

    float avgU(int i, int j) {
        int n = numY;
        float u_val = (u[i * n + j - 1] + u[i * n + j] +
                       u[(i + 1) * n + j - 1] + u[(i + 1) * n + j]) *
                      0.25;
        return u_val;
    }

    float avgV(int i, int j) {
        int n = numY;
        float v_val = (v[(i - 1) * n + j] + v[i * n + j] +
                       v[(i - 1) * n + j + 1] + v[i * n + j + 1]) *
                      0.25;
        return v_val;
    }

    void advectVel(float dt) {
        newU = u;
        newV = v;

        int n = numY;
        float h2 = 0.5 * h;

        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY; j++) {
                // u component
                if (s[i * n + j] != 0.0 && s[(i - 1) * n + j] != 0.0 && j < numY - 1) {
                    float x = i * h;
                    float y = j * h + h2;
                    float u_val = u[i * n + j];
                    float v_val = avgV(i, j);
                    x = x - dt * u_val;
                    y = y - dt * v_val;
                    u_val = sampleField(x, y, U_FIELD);
                    newU[i * n + j] = u_val;
                }
                // v component
                if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0 && i < numX - 1) {
                    float x = i * h + h2;
                    float y = j * h;
                    float u_val = avgU(i, j);
                    float v_val = v[i * n + j];
                    x = x - dt * u_val;
                    y = y - dt * v_val;
                    v_val = sampleField(x, y, V_FIELD);
                    newV[i * n + j] = v_val;
                }
            }
        }

        u = newU;
        v = newV;
    }

    void advectSmoke(float dt) {
        newM = m;

        int n = numY;
        float h2 = 0.5 * h;

        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0) {
                    float u_val = (u[i * n + j] + u[(i + 1) * n + j]) * 0.5;
                    float v_val = (v[i * n + j] + v[i * n + j + 1]) * 0.5;
                    float x = i * h + h2 - dt * u_val;
                    float y = j * h + h2 - dt * v_val;

                    newM[i * n + j] = sampleField(x, y, S_FIELD);
                }
            }
        }

        m = newM;
    }

private:
    float density;
    int numX;
    int numY;
    int numCells;
    float h;
    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> newU;
    std::vector<float> newV;
    std::vector<float> p;
    std::vector<float> s;
    std::vector<float> m;
    std::vector<float> newM;
    float overRelaxation; // You may want to adjust this value
};

int main() {
    // Example usage of the Fluid class
    Fluid myFluid(1.0, 5, 5, 0.1);
    myFluid.integrate(0.1, 9.8);
    myFluid.solveIncompressibility(10, 0.1);
    myFluid.extrapolate();
    myFluid.advectVel(0.1);
    myFluid.advectSmoke(0.1);

    return 0;
}
