#ifdef UEFI
unsigned int x = 0;
unsigned int y = 0;
unsigned int text_fg_color = 0xFFFFFFFF;
unsigned int text_bg_color = 0xFF061717;
#include "uefi_compat.h"
Bitmap_Font* font1;
Bitmap_Font* font2;
char text1[255];
char text2[255];
#else
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "raylib.h"
#endif

#define X_THRESHOLD 2.4f
#define THETA_THRESHOLD_RADIANS (12 * 2 * M_PI / 360)
#define MAX_STEPS 200
#define WIDTH 600
#define HEIGHT 200
#define SCALE 100

#ifdef UEFI
bool console_signal = false;
#endif

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

typedef struct Client Client;
struct Client {
};

typedef struct Cartpole Cartpole;
struct Cartpole {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* truncations;
    Log log;
    int num_agents;
    Client* client;
    float x;
    float x_prev;
    float x_dot;
    float theta;
    float theta_prev;
    float theta_dot;
    int tick;
    float cart_mass;
    float pole_mass;
    float pole_length;
    float gravity;
    float force_mag;
    float tau;
    int continuous;
    float episode_return;
    unsigned int rng;
    int origin_x;
    int origin_y;
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
#ifdef UEFI
    console_signal = false;
#endif
}

void allocate(Cartpole* env) {
    init(env);
    env->observations = (float*)calloc(4, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void free_allocated(Cartpole* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void c_close(Cartpole* env) {
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

Client* make_client(Cartpole* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
#ifndef UEFI
    InitWindow(WIDTH, HEIGHT, "file");
    SetTargetFPS(60);
#endif
    return client;
}

void close_client(Client* client) {
#ifndef UEFI
    CloseWindow();
#else
    console_signal = true;
#endif
    free(client);
}

void c_render(Cartpole* env) {
#ifndef UEFI
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);
    if (IsKeyPressed(KEY_TAB))
        ToggleFullscreen();
#endif
    if (env->client == NULL) {
        env->client = make_client(env);
    }

#ifndef UEFI
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
#endif
    DrawLine(env->origin_x, env->origin_y + (HEIGHT / 1.5), env->origin_x + WIDTH, env->origin_y + (HEIGHT / 1.5), PUFF_CYAN);
    float cart_x = env->origin_x + (WIDTH / 2 + env->x * SCALE);
    float cart_x_prev = env->origin_x + (WIDTH / 2 + env->x_prev * SCALE);
    float cart_y = env->origin_y + (HEIGHT / 1.6);
    DrawRectangle((int)(cart_x_prev - 20), (int)(cart_y - 10), 40, 20, PUFF_BACKGROUND);
    DrawRectangle((int)(cart_x - 20), (int)(cart_y - 10), 40, 20, PUFF_CYAN);
    float pole_length = 2.0f * 0.5f * SCALE;
    float pole_x2 = cart_x + sinf(env->theta) * pole_length;
    float pole_x2_prev = cart_x_prev + sinf(env->theta_prev) * pole_length;
    float pole_y2 = cart_y - cosf(env->theta) * pole_length;
    float pole_y2_prev = cart_y - cosf(env->theta_prev) * pole_length;
    DrawLineEx((Vector2){cart_x_prev, cart_y}, (Vector2){pole_x2_prev, pole_y2_prev}, 5, PUFF_BACKGROUND);
    DrawLineEx((Vector2){cart_x, cart_y}, (Vector2){pole_x2, pole_y2}, 5, PUFF_RED);
#ifndef UEFI
    DrawText(TextFormat("Steps: %i", env->tick), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Cart Position: %.2f", env->x), 10, 40, 20, PUFF_WHITE);
    DrawText(TextFormat("Pole Angle: %.2f", env->theta * 180.0f / M_PI), 10, 70, 20, PUFF_WHITE);
    EndDrawing();
#else
    //sprintf(text1,"s:%i x,th:%.2f,%.2f      \n", env->tick, env->x, env->theta * 180.0f / M_PI);
    x = env->origin_x / 2;
    y = env->origin_y / 2;
    print_string(text1, font1);
#endif
}

void compute_observations(Cartpole* env) {
    env->observations[0] = env->x;
    env->observations[1] = env->x_dot;
    env->observations[2] = env->theta;
    env->observations[3] = env->theta_dot;
}

void c_reset(Cartpole* env) {
    env->episode_return = 0.0f;
    env->x = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->x_dot = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta_dot = ((float)rand_r(&env->rng) / (float)RAND_MAX) * 0.08f - 0.04f;
    env->tick = 0;
    
    compute_observations(env);
}

void c_step(Cartpole* env) {  
    float a = env->actions[0];
    if (!isfinite(a)) {
        a = 0.0f;
    }
    a = fminf(fmaxf(a, -1.0f), 1.0f);
    env->actions[0] = a;

    env->x_prev = env->x;
    env->theta_prev = env->theta;

    float force = env->continuous ? a * env->force_mag
        : (a > 0.5f ? env->force_mag: -env->force_mag);

    float costheta = cosf(env->theta);
    float sintheta = sinf(env->theta);

    float total_mass = env->cart_mass + env->pole_mass;
    float polemass_length = total_mass + env->pole_mass;
    float temp = (force + polemass_length * env->theta_dot * env->theta_dot * sintheta) / total_mass;
    float thetaacc = (env->gravity * sintheta - costheta * temp) / 
                     (env->pole_length * (4.0f / 3.0f - total_mass * costheta * costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    env->x += env->tau * env->x_dot;
    env->x_dot += env->tau * xacc;
    env->theta += env->tau * env->theta_dot;
    env->theta_dot += env->tau * thetaacc;

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
