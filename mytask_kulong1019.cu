/*Program Title: Open Simulation of Artificial Sun Library*/
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <fstream>

#define PI 3.141592653589793
#define DOUBLE_MAX 1.7976931348623157E+308
#define blocks 16380
#define blockSize 512
#define EPS 1E-12

#define FREE_PATH 2.03646753

#define CUDA_CHECK_ERROR(err)           \
if ((err) != cudaSuccess) {          \
    printf("Cuda error: %s\n", cudaGetErrorString(err));    \
    printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}

using namespace std;

class Timer {
private:
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1> >;
    std::chrono::time_point<clock_t> m_beg;

public:
    Timer() : m_beg(clock_t::now()) {}
    void reset() { m_beg = clock_t::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
    }
};

/*두립자의 충돌시점계산함수 */
__device__ double search_D_t(double x1, double y1, double x2, double y2, double vx1, double vy1, double vx2, double vy2, double R) {
    double A, B, C, D;
    double t1, t2;

    A = (vx1 - vx2)*(vx1 - vx2) + (vy1 - vy2)*(vy1 - vy2);
    B = 2*((x1-x2)*(vx1 - vx2) + (y1-y2)*(vy1 - vy2));
    C = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) - 4*R*R;
    D = B*B - 4*A*C;

    if (D < 0)  {return DOUBLE_MAX;}

    t1 = (-B - (double)sqrt(D)) / 2 / A;
    t2 = (-B + (double)sqrt(D)) / 2 / A;

    if (abs(t1) < EPS) { t1 = 0; }
    if (abs(t2) < EPS) { t2 = 0; }

    if (t1 < 0 && t2 > 0)               {return t2;}
    if (t1 > 0 && t2 < 0)               {return t1;}
    if (t1 > 0 && t2 > 0 && t1 < t2)    {return t1;}
    if (t1 > 0 && t2 > 0 && t2 < t1)    {return t2;}
    return DOUBLE_MAX;
}

/*두립자사이거리계산 */
__device__ double search_r(double x1, double y1, double x2, double y2) {
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

/*속도스칼라계산 */
__host__ __device__ double search_V(double vx, double vy) {
    return sqrt(vx*vx + vy*vy);
}

/*공간에서 자유비행후 충돌시점계산 */
__global__ void kernel_pwa(double *x, double *y, double *vx, double *vy, int *m, int *l, double *tmin0, double limit, double R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double t;
    int mm, ll;
    mm = m[index];
    ll = l[index];

    for(int i = 0; i < 5; i++) {
        switch(i) {
            case 0: t = search_D_t(x[mm], y[mm], x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
            case 1: t = search_D_t(x[mm]-2*limit, y[mm], x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
            case 2: t = search_D_t(x[mm]+2*limit, y[mm], x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
            case 3: t = search_D_t(x[mm], y[mm]+2*limit, x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
            case 4: t = search_D_t(x[mm], y[mm]-2*limit, x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
        }
        if (t != DOUBLE_MAX) { break; }
    }
    tmin0[index] = t;
    __syncthreads();
}

/*주기적인 경계조건에 따른 충돌시점계산 */
__global__ void kernel_pwv(double *x, double *y, double *vx, double *vy, int *m, int *l, double *tmin0, double limit, double R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double t;
    int mm, ll;
    mm = m[index];
    ll = l[index];
    
    for(int i = 0; i < 3; i++) {
        switch(i) {
            case 0: t = search_D_t(x[mm], y[mm], x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
            case 1: t = search_D_t(x[mm]-2*limit, y[mm], x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
            case 2: t = search_D_t(x[mm]+2*limit, y[mm], x[ll], y[ll], vx[mm], vy[mm], vx[ll], vy[ll],R); break;
        }
        if (t != DOUBLE_MAX) { break; }
    }
    tmin0[index] = t;
    __syncthreads();
}

/*벽체와의 충돌시점계산 */
__global__ void kernel_hwh(int R, double *y, double *vy, double *time, double limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (limit - abs(y[index]) > 10*FREE_PATH) {
        time[index] = DOUBLE_MAX;
    } else {
        time[index] = ( limit - R - y[index] * vy[index] / abs(vy[index]) ) / abs(vy[index]);
    }
    __syncthreads();
}

__device__ double atomicMin_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void cuda_minimum(double* in, unsigned int size, double* out) {
    __shared__ double buf [blockSize];
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        buf[threadIdx.x] = in[index];
    else
        buf[threadIdx.x] = in[0];
    index += blocks * blockSize;
    while (index < size) {
        buf[threadIdx.x] = fmin(buf[threadIdx.x], in[index]);
        index += blocks * blockSize;
    }
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            buf[threadIdx.x] = min(buf[threadIdx.x], buf[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMin_double(out, buf[0]);
    }
    __syncthreads();
}

__global__ void search_min_index(double *times, double min_t, int *min_index) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (times[index] == min_t) {
        min_index[0] = index;
    }
}  

__global__ void dXdY_limit(double *x, double *y, double *vx, double *vy, double time, double lim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    x[index] = x[index] + vx[index]*time; 
    y[index] = y[index] + vy[index]*time;

    if(x[index] < -lim) { x[index]=x[index]+2*lim; }
    if(x[index] > lim) { x[index]=x[index]-2*lim; }
    if(y[index] < -lim) { y[index]=y[index]+2*lim; }
    if(y[index] > lim) { y[index]=y[index]-2*lim; }
    __syncthreads();
}

/*충돌한 두립자의 속도계산 - 꿀롱 힘 고려*/
__device__ void dVX_dVY(double *x, double *y, double *vx, double *vy, int p1, int p2, double R, double dt) {
    double vx1_, vx2_, vy1_, vy2_;

    // 충돌 시 속도 변화 (탄성 충돌)
    double cosa = (x[p2]-x[p1])/2/R;
    double sina = (y[p2]-y[p1])/2/R;

    vx1_ = vx[p1]*cosa+vy[p1]*sina;
    vy1_ = -vx[p1]*sina+vy[p1]*cosa;

    vx2_ = vx[p2]*cosa+vy[p2]*sina;
    vy2_ = -vx[p2]*sina+vy[p2]*cosa;

    vx[p1] = vx2_*cosa-vy1_*sina; 
    vy[p1] = vx2_*sina+vy1_*cosa; 

    vx[p2] = vx1_*cosa-vy2_*sina; 
    vy[p2] = vx1_*sina+vy2_*cosa;
    
    /*꿀롱 힘 고려 - 충돌 후 추가적인 속도 변화*/
    const double kCoulomb = 8.9875517873681764e9f;
    double q = 1.0e-9f;
    double mass = 1.0f;
    
    double axp1 = 0.0, ayp1 = 0.0;
    double axp2 = 0.0, ayp2 = 0.0;
    
    // 성능을 위해 근접한 립자들만 계산
    for(int j = 0; j < 50; j++) {
        if(p1 != j) {
            double dx = x[p1] - x[j];
            double dy = y[p1] - y[j];
            double r2 = dx*dx + dy*dy;
            
            if(r2 < 1e-12) r2 = 1e-12;
            
            double r = sqrt(r2);
            double force = kCoulomb * q * q / (r2);
            
            axp1 += force * (dx/r) / mass;
            ayp1 += force * (dy/r) / mass;
        }
        
        if(p2 != j) {
            double dx = x[p2] - x[j];
            double dy = y[p2] - y[j];
            double r2 = dx*dx + dy*dy;
            
            if(r2 < 1e-12) r2 = 1e-12;
            
            double r = sqrt(r2);
            double force = kCoulomb * q * q / (r2);
            
            axp2 += force * (dx/r) / mass;
            ayp2 += force * (dy/r) / mass;
        }
    }
    
    // 꿀롱 힘에 의한 속도 변화
    vx[p1] += axp1 * dt;
    vy[p1] += ayp1 * dt;
    vx[p2] += axp2 * dt;
    vy[p2] += ayp2 * dt;
}

/*꿀롱 힘만을 고려한 지속적인 힘 계산 (충돌 사이에도 적용)*/
__global__ void applyCoulombForces(double *x, double *y, double *vx, double *vy, double dt, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= N) return;
    
    const double kCoulomb = 8.9875517873681764e9f;
    double q = 1.0e-9f;
    double mass = 1.0f;
    
    double ax = 0.0, ay = 0.0;
    
    // 모든 다른 립자들에 대한 꿀롱 힘 계산
    for(int j = 0; j < N; j++) {
        if(index != j) {
            double dx = x[index] - x[j];
            double dy = y[index] - y[j];
            double r2 = dx*dx + dy*dy;
            
            if(r2 < 1e-12) r2 = 1e-12;
            
            double r = sqrt(r2);
            double force = kCoulomb * q * q / (r2);
            
            // 힘의 방향 단위 벡터
            ax += force * (dx/r) / mass;
            ay += force * (dy/r) / mass;
        }
    }
    
    // 속도 갱신
    vx[index] += ax * dt;
    vy[index] += ay * dt;
}

/*주기적인 경계조건에 의한 충돌후 계의 모든 립자들의 속도계산 */
__global__ void dist_impulses_pwv(double *x, double *y, double *vx, double *vy, int *min_i, int *m, int *l, double lim, double R, double dt) {
    double cor_r;
    int min_idx;
    int sample = 0;

    min_idx = min_i[0];

    for (int k = 0; k < 3; k++) {
        switch(k) {
            case 0: cor_r = search_r(x[m[min_idx]], y[m[min_idx]], x[l[min_idx]], y[l[min_idx]]); break;
            case 1: cor_r = search_r(x[m[min_idx]] - 2*lim, y[m[min_idx]], x[l[min_idx]], y[l[min_idx]]); break;
            case 2: cor_r = search_r(x[m[min_idx]] + 2*lim, y[m[min_idx]], x[l[min_idx]], y[l[min_idx]]); break;
        }
        if (abs(cor_r - 2*R) < EPS) {
            sample = k;
            break;
        }
    }

    switch(sample) {
        case 0: break;
        case 1: x[m[min_idx]] = x[m[min_idx]] - 2*lim; break;
        case 2: x[m[min_idx]] = x[m[min_idx]] + 2*lim; break;
    }

    dVX_dVY(x, y, vx, vy, m[min_idx], l[min_idx], R, dt);

    switch(sample) {
        case 0: break;
        case 1: x[m[min_idx]] = x[m[min_idx]] + 2*lim; break;
        case 2: x[m[min_idx]] = x[m[min_idx]] - 2*lim; break;
    }
}

/*공간에서 충돌한후 계의 모든 립자들의 속도계산 */
__global__ void dist_impulses_pwa(double *x, double *y, double *vx, double *vy, int *min_i, int *m, int *l, double lim, double R, double dt) {
    double cor_r;
    int min_idx;
    int sample = 0;

    min_idx = min_i[0];

    for (int k = 0; k < 5; k++) {
        switch(k) {
            case 0: cor_r = search_r(x[m[min_idx]], y[m[min_idx]], x[l[min_idx]], y[l[min_idx]]); break;
            case 1: cor_r = search_r(x[m[min_idx]] - 2*lim, y[m[min_idx]], x[l[min_idx]], y[l[min_idx]]); break;
            case 2: cor_r = search_r(x[m[min_idx]] + 2*lim, y[m[min_idx]], x[l[min_idx]], y[l[min_idx]]); break;
            case 3: cor_r = search_r(x[m[min_idx]], y[m[min_idx]] + 2*lim, x[l[min_idx]], y[l[min_idx]]); break;
            case 4: cor_r = search_r(x[m[min_idx]], y[m[min_idx]] - 2*lim, x[l[min_idx]], y[l[min_idx]]); break;
        }
        if (abs(cor_r - 2*R) < EPS) {
            sample = k;
            break;
        }
    }

    switch(sample) {
        case 0: break;
        case 1: x[m[min_idx]] = x[m[min_idx]] - 2*lim; break;
        case 2: x[m[min_idx]] = x[m[min_idx]] + 2*lim; break;
        case 3: y[m[min_idx]] = y[m[min_idx]] + 2*lim; break;
        case 4: y[m[min_idx]] = y[m[min_idx]] - 2*lim; break;
    }

    dVX_dVY(x, y, vx, vy, m[min_idx], l[min_idx], R, dt);

    switch(sample) {
        case 0: break;
        case 1: x[m[min_idx]] = x[m[min_idx]] + 2*lim; break;
        case 2: x[m[min_idx]] = x[m[min_idx]] - 2*lim; break;
        case 3: y[m[min_idx]] = y[m[min_idx]] - 2*lim; break;
        case 4: y[m[min_idx]] = y[m[min_idx]] + 2*lim; break;
    }
}

/*벽체와 충돌한후 계의 모든 립자들의 속도계산 */
__global__ void dist_impulses_hwh(double *x, double *y, double *vx, double *vy, int *min_i) {
    int min_idx = min_i[0]; 
    vy[min_idx] = -vy[min_idx];
}

__host__ void init_threads (int N, int threads, int *m, int *l) {
    int k, step;
    k = 0;
    step = N - 1;
    for (int i = 0; i < threads; i++) {
        if (step == 0) {
            k++;
            step = N - 1 - k;
        }
        m[i] = k;
        l[i] = N - step;
        step--;
    }
}

__host__ bool readFile(double *array, string filename, int N) {
    setlocale(LC_ALL, "rus");
    ifstream file (filename);

    if(!file) {
        return 0;
    } else {
        for (int i = 0; i < N; i++) {
            file >> array[i];
        }
        return 1;
    }
}

int main() {
    printf("\n//------------------------------//\n\n");
    int iter=0;
	int j=0;
    int preset = 0;
    int N = 4096;
    double R = 1;
    double vmod0 = 100;
    int boost = 1000;
    int cells = 4096;
    double cell_size = 2.4;
    int seed = 900;

	int gasu=0;
   
    if (2*R>cell_size) {
        printf("Частица не влезает в ячейку\n");
        return 0;
    }

    srand(seed);
    string file_x("x.txt");
    string file_y("y.txt");
    string file_vx("vx.txt");
    string file_vy("vy.txt");

    Timer t;

    switch(preset) {
        case 0:
            printf("\nДве горизонтальных твердых стенки--두개의 수평인 굳은 벽이 있다"); 
            printf("\nДве вертикальных периодических стенки---두개의 수직이며 주기적인 굳은 벽이 있다");
            printf("\nДва противоположных потока---두개의 반대흐름이 있다\n\n");
            break;
        case 1:
            printf("\n4 периодических стенки--4개의 주기적인 벽들이 있다");
            printf("\nСлучайная ориентация векторов скорости--속도벡토르값들은 우연발생\n");
            break;
    }

    iter=5000000;
    printf(" количество итераций кода iter(정적기억기할당능력을 고려하여 최대코드반복회수는)=%d 로 설정해놓았음. ", iter);

    printf("\n");

    int *coll = (int*)malloc(iter * sizeof(int));
    bool flag1;
    printf("Указать одно значение столкновений для всех итераций\n모든 반복에서 하나의 충돌값지정 - 0\n");
    printf("Указать разные количества столкновений за итерации\n모든 반복에서 여러개의 충돌값지정  - 1\n");
    flag1 = 0;
    scanf("%i", &flag1);
    if (flag1 == 0) {
        printf("Укажите одно количество столкновений для всех итераций\n(모든 반복에 대해 동일한 충돌수를 지정하시오): ");
        scanf("%i", &coll[0]);
        for (int i = 1; i < iter; i++) {
            coll[i] = coll[0];
        }
    } else {
        printf("Введите количества соударений на каждую итерацию кода colli\n( 각 반복에 대한 충돌 수를 입력합니다): \n");
        for (int i = 0; i < iter; i++) {
            printf("iter %i/%i:", i+1, iter);
            scanf("%i", &coll[i]);
            printf("\n");
        }
    }

    double *x, *y, *vx, *vy, Mv;
	
    double sumE0, sumE1;
    double system_time,system_time1,system_time2;
	system_time1=0;
	system_time2=0;
    double *out;

    int *m;
    int *l;
    double border;
    int threads;
    double tmin, tmin2;

    double *dev_x, *dev_y, *dev_vx, *dev_vy;
    double *dev_tmin0, *dev_tmin02, *dev_out, *dev_out2;
    int *dev_min_index, *dev_min_index2, *dev_m, *dev_l;

    threads = (N * N - N) / 2;

    double dSIZE = threads * sizeof(double);
    double dSIZE_N = N * sizeof(double);
    double dSIZE_threads = threads * sizeof(double);
    int iSIZE = threads * sizeof(int);

   cudaMalloc((void**)&dev_x, N * sizeof(double));
   cudaMalloc((void**)&dev_y, N * sizeof(double));
   cudaMalloc((void**)&dev_vx, N * sizeof(double));
   cudaMalloc((void**)&dev_vy, N * sizeof(double));

    cudaMalloc((void **) &dev_m, iSIZE);
    cudaMalloc((void **) &dev_l, iSIZE);
    cudaMalloc((void **) &dev_tmin0, dSIZE);
    cudaMalloc((void **) &dev_tmin02, dSIZE_N);
    cudaMalloc((void **) &dev_out, sizeof(double));
    cudaMalloc((void **) &dev_out2, sizeof(double));
    cudaMalloc((void **) &dev_min_index, iSIZE);
    cudaMalloc((void **) &dev_min_index2, iSIZE);

   	x = (double *)malloc(N * sizeof(double));
    y = (double *)malloc(N * sizeof(double));
	vx = (double *)malloc(N * sizeof(double));
	vy = (double *)malloc(N * sizeof(double));
    m = (int *) malloc(iSIZE);
    l = (int *) malloc(iSIZE);
    out = (double *) malloc(dSIZE);

    init_threads(N, threads, m, l); 

    int a = sqrt(cells);
    border = a*cell_size/2.;
    sumE0 = 0;
    sumE1 = 0;
    int np = 0;

	bool flag=0;
    printf("%i частиц-립자들\n\n", N);
    printf("Новый запуск? (yes-0 / no-1)? \n 새로 시작하겠습니까?(예-0 아니 1): = ");
    scanf("%i", &flag);
	if(flag==0)
	{
		printf("%d이므로 모의를 새로 시작한다\n",flag);
	}
	else
	{
		printf("%d이므로 전번모의에서 얻은 자료를 리용하여 모의를 계속한다.\n",flag);
	}
    if (flag==0) 
	{
             for (int i = 0; i < a; i++) {
                for(int k = 0; k < a; k++) {
					x[np] = - a*cell_size/2 + cell_size/2 + i*cell_size;
                    y[np] = - a*cell_size/2 + cell_size/2 + k*cell_size;
                    np++;
                }
            }

            for (int i = 0; i < N; i++) {
                vx[i] = ((vmod0 * (double) rand() / RAND_MAX) * 2) - vmod0;
                if (rand() % 2 == 0) { 
                    vy[i] = sqrt(vmod0 * vmod0 - vx[i] * vx[i]);
                } else { 
                    vy[i] = -(sqrt(vmod0 * vmod0 - vx[i] * vx[i]));
                }
            }

            if (preset == 0) {
                for (int i = 0; i < N; i++) {
                    if (y[i] > 0) {
                        vx[i] = vx[i] + boost;
                    } else {
                        vx[i] = vx[i] - boost;
                    }
                }
            }
    }
    else
	{
			printf("%d\n이므로 전번모의에서 얻은 자료를 리용하여 모의를 계속한다.\n",flag);
			FILE* f_time1;
				f_time1 = fopen("time.txt", "r+");
				fscanf(f_time1, "%lf %d", &system_time1, &j);
			fclose(f_time1);
			
			printf("%.20f시간부터 계속 %d로부터 되풀이수증가\n",system_time1,j);
			
            if (readFile(x, file_x, N) == 1)        {printf("Координаты x считаны\t+\n");}
            else                                    {printf("Ошибка:\tКоординаты x не считаны!!!\n"); return 0;}
            if (readFile(y, file_y, N) == 1)        {printf("Координаты y считаны\t+\n");}
            else                                    {printf("Ошибка:\tКоординаты y не считаны!!!\n"); return 0;}
            if (readFile(vx, file_vx, N) == 1)      {printf("Скорости vx считаны\t+\n");}
            else                                    {printf("Ошибка:\tСкорости x не считаны!!!\n"); return 0;}
            if (readFile(vy, file_vy, N) == 1)      {printf("Скорости vy считаны\t+\n");}
            else                                    {printf("Ошибка:\tСкорости x не считаны!!!\n\n"); return 0;}
	}
		
    for (int i = 0; i < N; i++) {
        sumE0 += vx[i]*vx[i] + vy[i]*vy[i];
    }
    sumE0 = sumE0/2;

    printf("Начальная энергия-초기에네르기: %.16f\n", sumE0);

    if (flag == 0) {
        FILE* f0;
        f0 = fopen("info.csv", "w");
        fprintf(f0,"N,R,vmod0,boost,cell_size,cells,seed,threads,blocks,blockSize\n");
        fprintf(f0,"%i,%f,%f,%i,%f,%i,%i,%i,%i,%i\n",N,R,vmod0,boost,cell_size,cells,seed,threads,blocks,blockSize);
        fclose(f0);

        FILE* f1;
        f1 = fopen("log.csv", "w");
        fprintf(f1,"x,y,vx,vy,V,coll,evol_time,prog_time\n");
        for (int i = 0; i < N; i++) {
            fprintf(f1,"%.16f,%.16f,%.16f,%.16f,%.16f", x[i], y[i], vx[i], vy[i], search_V(vx[i], vy[i]));
            if (i == 0) {
                fprintf(f1,",0,0,0");
            } fprintf(f1,"\n");
        } fclose(f1);
    }

    t.reset();
    CUDA_CHECK_ERROR(cudaMemcpy(dev_m, m, iSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_l, l, iSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_x, x, dSIZE_N, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_y, y, dSIZE_N, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_vx, vx, dSIZE_N, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_vy, vy, dSIZE_N, cudaMemcpyHostToDevice));	
			
     printf("반복회수/총되풀이수 готово, прошло 실행시간 min, 다음번충돌까지 시간\n");              
    
    FILE *f3, *f4, *f5;
	FILE* f_historam;
    f_historam = fopen("f_historam of.csv", "a");
    f4 = fopen("log_pzy_f(t).csv", "a");
    f5= fopen("log_pzy_df(t).csv", "a");
	
	fprintf(f4, "Начальная энергия-초기에네르기: %.16f\n", sumE0);
	
	int histogram[10000], histogram_freeflight[100000];
	for(int kkk=0;kkk<100000;kkk++)
	{
		histogram_freeflight[kkk]=0;
	}
	fprintf(f_historam, "Начальная энергия-초기에네르기: %.16f\n", sumE0);
    
    switch(preset) {
        case 0: 
		{
			while(system_time1<1 && j < iter)
			{
				j++;
				for(int i=0;i<10000;i++)
				{
					histogram[i]=0;
				}
      			system_time = 0; 
		
				for (int i = 0; i < coll[j]; i++) {

					out[0] = DOUBLE_MAX;
					int gridSize_threads = (threads + blockSize - 1) / blockSize;
					kernel_pwv <<<gridSize_threads, blockSize>>>(dev_x, dev_y, dev_vx, dev_vy, dev_m, dev_l, dev_tmin0, border, R);
					CUDA_CHECK_ERROR(cudaMemcpy(dev_out, out, sizeof(double), cudaMemcpyHostToDevice));
					cuda_minimum <<<blocks, blockSize>>>(dev_tmin0, threads, dev_out);
					CUDA_CHECK_ERROR(cudaMemcpy(out, dev_out, sizeof(double), cudaMemcpyDeviceToHost));
					tmin = out[0];

					out[0] = DOUBLE_MAX;
					kernel_hwh <<<N/blockSize, blockSize>>> (R, dev_y, dev_vy, dev_tmin02, border);
					CUDA_CHECK_ERROR(cudaMemcpy(dev_out2, out, sizeof(double), cudaMemcpyHostToDevice));
					cuda_minimum <<<blocks, blockSize>>>(dev_tmin02, N, dev_out2);
					CUDA_CHECK_ERROR(cudaMemcpy(out, dev_out2, sizeof(double), cudaMemcpyDeviceToHost));
					tmin2 = out[0]; 

					// 충돌 처리 전 꿀롱 힘 적용
					applyCoulombForces <<<N/blockSize, blockSize>>> (dev_x, dev_y, dev_vx, dev_vy, tmin, N);

					if (tmin < tmin2) {
						system_time += tmin; 
						search_min_index <<<blocks, blockSize>>> (dev_tmin0, tmin, dev_min_index);
						dXdY_limit <<<N/blockSize, blockSize>>>(dev_x, dev_y, dev_vx, dev_vy, tmin, border);
						dist_impulses_pwv <<<1,1>>> (dev_x, dev_y, dev_vx, dev_vy, dev_min_index, dev_m, dev_l, border, R, tmin);
					} else {
						system_time += tmin2; 
						search_min_index <<<N/blockSize, blockSize>>> (dev_tmin02, tmin2, dev_min_index2);
						dXdY_limit <<<N/blockSize, blockSize>>>(dev_x, dev_y, dev_vx, dev_vy, tmin2, border);
						dist_impulses_hwh <<<1,1>>> (dev_x, dev_y, dev_vx, dev_vy, dev_min_index2);
					}
				}

				CUDA_CHECK_ERROR(cudaMemcpy(x, dev_x, dSIZE_N, cudaMemcpyDeviceToHost));
    			CUDA_CHECK_ERROR(cudaMemcpy(y, dev_y, dSIZE_N, cudaMemcpyDeviceToHost));
   				CUDA_CHECK_ERROR(cudaMemcpy(vx, dev_vx, dSIZE_N, cudaMemcpyDeviceToHost));
   				CUDA_CHECK_ERROR(cudaMemcpy(vy, dev_vy, dSIZE_N, cudaMemcpyDeviceToHost));
				
				sumE1 = 0;
				for (int i = 0; i < N; i++) { 
					sumE1 += vx[i]*vx[i] + vy[i]*vy[i];
				}
				sumE1 = sumE1/2.;
			
				if (abs(sumE0/sumE1 - 1.) > 0.1) { 
					printf("\n!!! Ошибка в законе сохранения энергии !!!\n");
					break;
				} else {
					
					FILE* f_time;
					f_time = fopen("time.txt", "w");
					fprintf(f_time,"%.20f %d",system_time1,j);
					fclose(f_time);
					
					FILE* f21;
					f21 = fopen("x.txt", "w");
					for(int i = 0; i < N; i++) {
						fprintf(f21, "%.16f\n", x[i]);
					} 
					fclose(f21);

					FILE* f22;
					f22 = fopen("y.txt", "w");
					for(int i = 0; i < N; i++) {
						fprintf(f22, "%.16f\n", y[i]);
					} fclose(f22);

					FILE* f23;
					f23 = fopen("vx.txt", "w");
					for(int i = 0; i < N; i++) {
						fprintf(f23, "%.16f\n", vx[i]);
					} 
					fclose(f23);

					FILE* f24;
					f24 = fopen("vy.txt", "w");
					for(int i = 0; i < N; i++) {
						fprintf(f24, "%.16f\n", vy[i]);
					} 
					fclose(f24);

					f3 = fopen("log.csv", "a");
					
				   double max_v = 0.0;
					Mv=0;
				   for (int i = 0; i < N; i++) {
                        fprintf(f3,"%.16f,%.16f,%.16f,%.16f,%.16f", x[i], y[i], vx[i], vy[i], search_V(vx[i], vy[i]));
						
						if(max_v < search_V(vx[i], vy[i])){
							max_v = search_V(vx[i], vy[i]);
						}

                        if (i == 0) {
                            fprintf(f3,",%i,%.20f,%.6f", coll[j], system_time, t.elapsed()/60.);
                        }
                        fprintf(f3,"\n");
						Mv=Mv+search_V(vx[i], vy[i])*search_V(vx[i], vy[i]);
                    } 
					Mv=Mv/(double)N;
					
					fclose(f3);
	
					int jj=0;
					for(double vv=0; vv <= max_v; vv+=10.0){
						histogram[jj]=0;
						for(int k=0;k<N;k++)
						{
							if((search_V(vx[k], vy[k]) >= vv) && (search_V(vx[k], vy[k]) < vv+10.0))
							{
								histogram[jj]++;
							}						
						}
						jj++;
					}

					/*자유비행확률밀도함수얻기*/
					int jjj=0;
					for(double tt=0; tt <= 0.00001; tt+=0.0000000001){
						if((system_time >= tt) && (system_time < tt+0.0000000001))
							{
								histogram_freeflight[jjj]++;
							}
						jjj++;								
					}
					
					if(j%100000==0)
					{
						printf("Настояшая энергия-ена현재에네르기: %.16f\n Настояшое время현재시간  %.16f\n", sumE1, system_time1);
						fprintf(f_historam, "Настояшая энергия-ена현재에네르기: %.16f\n Настояшое время현재시간  %.16f\n", sumE1,system_time1);
						printf("%.6f\n", max_v);
						fprintf(f_historam,"%.6f\n", max_v);
						
						for(int k=0;k<jj;k++)
						{
							fprintf(f_historam,"%i, %.6f\n",k*10, double(histogram[k])/double(N));
							printf("%i, %i\n",k*10, histogram[k]);
							printf("%i, %.6f\n",k*10, double(histogram[k])/double(N));
						}
						printf("-------------------------------\n");
						fprintf(f_historam,"%.6f\n",sqrt(Mv));
						printf("-------------------------------\n");
						printf("자유비행시간밀도함수출력\n");

						fprintf(f_historam, "자유비행시간밀도함수출력\n");
						for(int k=0;k<jjj;k++)
						{
							printf("%.20f, %.6f\n",double(k*0.0000000001), double(histogram_freeflight[k])/double(100000));
							fprintf(f_historam,"%.20f, %.6f\n",double(k*0.0000000001), double(histogram_freeflight[k])/double(100000));
						} 
						printf("-------------------------------\n");
						printf("-------------------------------\n");
						for(int kkk=0;kkk<100000;kkk++)
						{
							histogram_freeflight[kkk]=0;
						}
						
					}
					fprintf(f4,"%i, %.5f , %.20f\n", j+1, t.elapsed()/60.,system_time);

					system_time1+=  system_time; 
					system_time2+= system_time; 
					gasu++;
					if (system_time2>0.001){
						printf("%i/%i готово, прошло %.5f min\n", j+1, iter, t.elapsed()/60.);
						fprintf(f5,"%i, %.20f\n", gasu,system_time1);
						printf("%i, %.20f\n", gasu,system_time1);
						gasu=0;
						system_time2=0;
					}
	
				}
			}
			break;			
		}
        case 1: 
		{
			while(system_time1<1 && j < iter)
			{
				j++;
		 	               
		        system_time = 0; 
				for(int i=0;i<10000;i++)
				{
					histogram[i]=0;
				}
				
                for (int i = 0; i < coll[j]; i++) {

                    out[0] = DOUBLE_MAX;
                    kernel_pwa <<<threads/blockSize, blockSize>>>(dev_x, dev_y, dev_vx, dev_vy, dev_m, dev_l, dev_tmin0, border, R);
                    CUDA_CHECK_ERROR(cudaMemcpy(dev_out, out, sizeof(double), cudaMemcpyHostToDevice));
                    cuda_minimum <<<blocks, blockSize>>>(dev_tmin0, threads, dev_out);
                    CUDA_CHECK_ERROR(cudaMemcpy(out, dev_out, sizeof(double), cudaMemcpyDeviceToHost));

                    tmin = out[0];
                    system_time += tmin; 
                    search_min_index <<<blocks, blockSize>>> (dev_tmin0, tmin, dev_min_index);
                    dXdY_limit <<<N/blockSize, blockSize>>>(dev_x, dev_y, dev_vx, dev_vy, tmin, border);
                    // 수정: dt 인자 추가
                    dist_impulses_pwa <<<1,1>>> (dev_x, dev_y, dev_vx, dev_vy, dev_min_index, dev_m, dev_l, border, R, tmin);
                }
				CUDA_CHECK_ERROR(cudaMemcpy(x, dev_x, dSIZE_N, cudaMemcpyDeviceToHost));
    			CUDA_CHECK_ERROR(cudaMemcpy(y, dev_y, dSIZE_N, cudaMemcpyDeviceToHost));
   				CUDA_CHECK_ERROR(cudaMemcpy(vx, dev_vx, dSIZE_N, cudaMemcpyDeviceToHost));
   				CUDA_CHECK_ERROR(cudaMemcpy(vy, dev_vy, dSIZE_N, cudaMemcpyDeviceToHost));
				
                sumE1 = 0;
                for (int i = 0; i < N; i++) { 
                    sumE1 += vx[i]*vx[i] + vy[i]*vy[i];
                }
                sumE1 = sumE1/2.;
				if (abs(sumE0/sumE1 - 1.) > 0.1) { 
                    printf("\n!!! Ошибка в законе сохранения энергии !!!\n");
                    break;
                } else {

                   FILE* f_time;
					f_time = fopen("time.txt", "w");
					fprintf(f_time,"%.20f %d",system_time1,j);
					fclose(f_time);
					
					FILE* f21;
                    f21 = fopen("x.txt", "w");
					for(int i = 0; i < N; i++) {
                        fprintf(f21, "%.16f\n", x[i]);
                    } fclose(f21);

                    FILE* f22;
                    f22 = fopen("y.txt", "w");
                    for(int i = 0; i < N; i++) {
                        fprintf(f22, "%.16f\n", y[i]);
                    } fclose(f22);

                    FILE* f23;
                    f23 = fopen("vx.txt", "w");
                    for(int i = 0; i < N; i++) {
                        fprintf(f23, "%.16f\n", vx[i]);
                    } fclose(f23);

                    FILE* f24;
                    f24 = fopen("vy.txt", "w");
                    for(int i = 0; i < N; i++) {
                        fprintf(f24, "%.16f\n", vy[i]);
                    } fclose(f24);

                    f3 = fopen("log.csv", "a");
                  
   				   double max_v=0.0;
				   
				   Mv=0;
				   for (int i = 0; i < N; i++) {
                        fprintf(f3,"%.16f,%.16f,%.16f,%.16f,%.16f", x[i], y[i], vx[i], vy[i], search_V(vx[i], vy[i]));
						
						if(max_v < search_V(vx[i], vy[i])){
							max_v = search_V(vx[i], vy[i]);
						}

                        if (i == 0) {
                            fprintf(f3,",%i,%.20f,%.6f", coll[j], system_time, t.elapsed()/60.);
                        }
                        fprintf(f3,"\n");
						Mv=Mv+search_V(vx[i], vy[i])*search_V(vx[i], vy[i]);
                    } 
					Mv=Mv/(double)N;
					
					
					int jj=0;
					for(double vv=0; vv<=max_v; vv+=10.0){
						histogram[jj]=0;
						for(int k=0;k<N;k++)
						{
							if((search_V(vx[k], vy[k]) >= vv) && (search_V(vx[k], vy[k]) < vv+10.0))
							{
								histogram[jj]++;
							}							
						}
						jj++;
					}

					/*자유비행확률밀도함수얻기*/
					int jjj=0;
					for(double tt=0; tt <= 0.00001; tt+=0.0000000001){
						if((system_time >= tt) && (system_time < tt+0.0000000001))
							{
								histogram_freeflight[jjj]++;
							}
						jjj++;								
					}
					
					if(j%100000==0)
					{
						printf("Настояшая энергия-ена현재에네르기: %.16f\n Настояшое время현재시간  %.16f\n", sumE1, system_time1);
						fprintf(f_historam, "Настояшая энергия-ена현재에네르기: %.16f\n Настояшое время현재시간  %.16f\n", sumE1,system_time1);
						printf("%.6f\n", max_v);
						fprintf(f_historam,"%.6f\n", max_v);
						
						for(int k=0;k<jj;k++)
						{
							fprintf(f_historam,"%i, %.6f\n",k*10, double(histogram[k])/double(N));
							printf("%i, %i\n",k*10, histogram[k]);
							printf("%i, %.6f\n",k*10, double(histogram[k])/double(N));
						}
						printf("-------------------------------\n");
						fprintf(f_historam,"%.6f\n",sqrt(Mv));
						printf("-------------------------------\n");
						printf("자유비행시간밀도함수출력\n");

						fprintf(f_historam, "자유비행시간밀도함수출력\n");
						for(int k=0;k<jjj;k++)
						{
							printf("%.20f, %.6f\n",double(k*0.0000000001), double(histogram_freeflight[k])/double(100000));
							fprintf(f_historam,"%.20f, %.6f\n",double(k*0.0000000001), double(histogram_freeflight[k])/double(100000));
						} 
						printf("-------------------------------\n");
						printf("-------------------------------\n");
						for(int kkk=0;kkk<100000;kkk++)
						{
							histogram_freeflight[kkk]=0;
						}
						
					}
					
					fprintf(f4,"%i, %.5f , %.20f\n", j+1, t.elapsed()/60.,system_time);

					system_time1+=  system_time; 
					system_time2+= system_time; 
					gasu++;
					if (system_time2>0.001){
						printf("%i/%i готово, прошло %.5f min\n", j+1, iter, t.elapsed()/60.);
						fprintf(f5,"%i, %.20f\n", gasu,system_time1);
						printf("%i, %.20f\n", gasu,system_time1);
						gasu=0;
						system_time2=0;
					}
                }
            } 
			break;
		}
    }
    printf("\nКонечная энергия최종에네르기: %.16f\n", sumE1);
    fprintf(f4,"\nКонечная энергия최종에네르기: %.16f\n", sumE1);
    printf("Время исполнения программы 프로그람실행시간(минут): %.5f\n", t.elapsed()/60.);
    fprintf(f4,"Время исполнения программы 프로그람실행시간(минут): %.5f\n", t.elapsed()/60.);
    printf("Время эволюции системы체계진화시간: %.20f\n", system_time1);
    fprintf(f4,"Время эволюции системы체계진화시간: %.20f\n", system_time1);
    fclose(f4);
    fclose(f5);
    fclose(f_historam);
	
    free(x);
    free(y);
    free(vx);
    free(vy);
    free(m);
    free(l);
    free(coll);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_vx);
    cudaFree(dev_vy);
    cudaFree(dev_m);
    cudaFree(dev_l);

    return 0;
}