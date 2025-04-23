#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
//TODO: more than two-dimensions

//n-body or no?
#define GRAVITY (9.81f) //seemingly of Earth - at least, partly

#define MASSCART 1.0f
#define MASSPOLE 0.1f
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5f // half pole length
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0f
#define TAU 0.02f // timestep duration

#define X_THRESHOLD 2.4f
#define THETA_THRESHOLD_RADIANS (12 * 2 * M_PI / 360)
#define MAX_STEPS 200
#define WIDTH 600
#define HEIGHT 200
#define SCALE 100

typedef struct Log Log;
struct Log {
    float perf;
    float episode_length;
    float x_threshold_termination;
    float pole_angle_termination;
    float max_steps_termination;
    float n;
    float score;
};

typedef struct Cartpole Cartpole;
struct Cartpole {
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Log log;
    float x;
    float x_dot;
    float theta;
    float theta_dot;
    int tick;
    int continuous;
    float episode_return;
};

void add_log(Cartpole* env) {
    if (env->episode_return > 0) {
        env->log.perf = env->episode_return / MAX_STEPS;
    } else {
        env->log.perf = 0.0f;
    }
    env->log.episode_length += env->tick;
    env->log.score += env->tick;
    env->log.x_threshold_termination += (env->x < -X_THRESHOLD || env->x > X_THRESHOLD);
    env->log.pole_angle_termination += (env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS);
    env->log.max_steps_termination += (env->tick >= MAX_STEPS);
    env->log.n += 1;
}

void init(Cartpole* env) {
    env->tick = 0;
    memset(&env->log, 0, sizeof(Log));
}

void allocate(Cartpole* env) {
    init(env);
    env->observations = (float*)calloc(4, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(Cartpole* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void compute_observations(Cartpole* env) {
    env->observations[0] = env->x;
    env->observations[1] = env->x_dot;
    env->observations[2] = env->theta;
    env->observations[3] = env->theta_dot;
}

void c_reset(Cartpole* env) {
    env->episode_return = 0.0f;
    env->x = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->x_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->tick = 0;

    compute_observations(env);
}

void c_step(Cartpole* env) {
    // float force = 0.0;
    // if (env->continuous) {
    //     force = env->actions[0] * FORCE_MAG;
    // } else {
    //     force = (env->actions[0] > 0.5f) ? FORCE_MAG : -FORCE_MAG; 
    // }

    float a = env->actions[0];

    /* ===== runtime sanity check –– delete after debugging ===== */
    if (!isfinite(a) || a < -1.0001f || a > 1.0001f) {
        fprintf(stderr,
                "[ACTION] tick=%d  raw=%.6f\n",
                env->tick, a);
        fflush(stderr);
    }
    /* ========================================================== */

    if (!isfinite(a))             a = 0.0f;
    a = fminf(fmaxf(a, -1.0f), 1.0f);
    env->actions[0] = a;

    float force = env->continuous ? a * FORCE_MAG
                                  : (a > 0.5f ? FORCE_MAG : -FORCE_MAG);

    float costheta = cosf(env->theta);
    float sintheta = sinf(env->theta);

    float temp = (force + POLEMASS_LENGTH * env->theta_dot * env->theta_dot * sintheta) / TOTAL_MASS;
    float thetaacc = (GRAVITY * sintheta - costheta * temp) / 
                     (LENGTH * (4.0f / 3.0f - MASSPOLE * costheta * costheta / TOTAL_MASS));
    float xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

    env->x += TAU * env->x_dot;
    env->x_dot += TAU * xacc;
    env->theta += TAU * env->theta_dot;
    env->theta_dot += TAU * thetaacc;

    env->tick += 1;

    bool terminated = env->x < -X_THRESHOLD || env->x > X_THRESHOLD ||
                env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS;
    bool truncated = env->tick >= MAX_STEPS;
    bool done = terminated || truncated;

    env->rewards[0] = done ? 0.0f : 1.0f;
    env->episode_return += env->rewards[0];
    env->terminals[0] = terminated ? 1 : 0;

    if (done) {
        add_log(env);
        c_reset(env);
    }

    compute_observations(env);
}
