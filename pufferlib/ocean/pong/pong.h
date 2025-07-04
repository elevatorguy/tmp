#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "raylib.h"

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct Pong Pong;
struct Pong {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    float paddle_yl;
    float paddle_yr;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    unsigned int score_l;
    unsigned int score_r;
    float width;
    float height;
    float paddle_width;
    float paddle_height;
    float ball_width;
    float ball_height;
    float paddle_speed;
    float ball_initial_speed_x;
    float ball_initial_speed_y;
    float ball_max_speed_y;
    float ball_speed_y_increment;
    unsigned int max_score;
    float min_paddle_y;
    float max_paddle_y;
    float paddle_dir;
    int tick;
    int n_bounces;
    int win;
    int frameskip;
    int continuous;
};

void init(Pong* env) {
    // logging
    env->tick = 0;
    env->n_bounces = 0;
    env->win = 0;

    // precompute
    env->min_paddle_y = -env->paddle_height / 2;
    env->max_paddle_y = env->height - env->paddle_height/2;
    
    env->paddle_dir = 0;
}

void allocate(Pong* env) {
    init(env);
    env->observations = (float*)calloc(8, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(Pong* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void c_close(Pong* env) {
}

void add_log(Pong* env) {
    float score = (float)env->score_r - (float)env->score_l;
    env->log.episode_length += env->tick;
    env->log.episode_return += score;
    env->log.score += score;
    env->log.perf += (float)(env->score_r) / ((float)env->score_l + (float)env->score_r);
    env->log.n += 1;
}

void compute_observations(Pong* env) {
    env->observations[0] = (env->paddle_yl - env->min_paddle_y) / (env->max_paddle_y - env->min_paddle_y);
    env->observations[1] = (env->paddle_yr - env->min_paddle_y) / (env->max_paddle_y - env->min_paddle_y);
    env->observations[2] = env->ball_x / env->width;
    env->observations[3] = env->ball_y / env->height;
    env->observations[4] = (env->ball_vx + env->ball_initial_speed_x) / (2 * env->ball_initial_speed_x);
    env->observations[5] = (env->ball_vy + env->ball_max_speed_y) / (2 * env->ball_max_speed_y);
    env->observations[6] = env->score_l / env->max_score;
    env->observations[7] = env->score_r / env->max_score;
}

void reset_round(Pong* env) {
    env->paddle_yl = env->height / 2 - env->paddle_height / 2;
    env->paddle_yr = env->height / 2 - env->paddle_height / 2;
    env->ball_x = env->width / 5;
    env->ball_y = env->height / 2 - env->ball_height / 2;
    env->ball_vx = env->ball_initial_speed_x;
    env->ball_vy = (rand() % 2 - 1) * env->ball_initial_speed_y;
    env->tick = 0;
    env->n_bounces = 0;
}

void c_reset(Pong* env) {
    reset_round(env);
    env->score_l = 0;
    env->score_r = 0;
    compute_observations(env);
}

void c_step(Pong* env) {
    env->tick += 1;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    // move ego paddle
    if (env->continuous) {
        env->paddle_dir = env->actions[0];
    } else {
        float act = env->actions[0];
        env->paddle_dir = 0;
        if (act == 0.0) { // still
            env->paddle_dir = 0;
        } else if (act == 1.0) { // up
            env->paddle_dir = 1;
        } else if (act == 2.0) { // down
            env->paddle_dir = -1;
        }
    }

    for (int i = 0; i < env->frameskip; i++) {
        env->paddle_yr += env->paddle_speed * env->paddle_dir;
        
        // move opponent paddle
        float opp_paddle_delta = env->ball_y - (env->paddle_yl + env->paddle_height / 2);
        opp_paddle_delta = fminf(fmaxf(opp_paddle_delta, -env->paddle_speed), env->paddle_speed);
        env->paddle_yl += opp_paddle_delta;

        // clip paddles
        env->paddle_yr = fminf(fmaxf(
            env->paddle_yr, env->min_paddle_y), env->max_paddle_y);
        env->paddle_yl = fminf(fmaxf(
            env->paddle_yl, env->min_paddle_y), env->max_paddle_y);

        // move ball
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;

        // handle collision with top & bottom walls
        if (env->ball_y < 0 || env->ball_y + env->ball_height > env->height) {
            env->ball_vy = -env->ball_vy;
        }

        // handle collision on left
        if (env->ball_x < 0) {
            if (env->ball_y + env->ball_height > env->paddle_yl && \
                env->ball_y < env->paddle_yl + env->paddle_height) {
                // collision with paddle
                env->ball_vx = -env->ball_vx;
                env->n_bounces += 1;
            } else {
                // collision with wall: WIN
                env->win = 1;
                env->score_r += 1;
                env->rewards[0] = 1; // agent wins
                if (env->score_r == env->max_score) {
                    env->terminals[0] = 1;
                    add_log(env);
                    c_reset(env);
                    return;
                } else {
                    reset_round(env);
                    return;
                }
            }
        }

        // handle collision on right (TODO duplicated code)
        if (env->ball_x + env->ball_width > env->width) {
            if (env->ball_y + env->ball_height > env->paddle_yr && \
                env->ball_y < env->paddle_yr + env->paddle_height) {
                // collision with paddle
                env->ball_vx = -env->ball_vx;
                env->n_bounces += 1;
		env->rewards[0] = 0.1; // agent bounced the ball
                // ball speed change
                env->ball_vy += env->ball_speed_y_increment * env->paddle_dir;
                env->ball_vy = fminf(fmaxf(env->ball_vy, -env->ball_max_speed_y), env->ball_max_speed_y);
                if (fabsf(env->ball_vy) < 0.01) { // we dont want a horizontal ball
                    env->ball_vy = env->ball_speed_y_increment;
                }
            } else {
                // collision with wall: LOSE
                env->win = 0;
                env->score_l += 1;
                env->rewards[0] = -1.0;
                if (env->score_l == env->max_score) {
                    env->terminals[0] = 1;
                    add_log(env);
                    c_reset(env);
                    return;
                } else {
                    reset_round(env);
                    return;
                }
            }

            // clip ball
            env->ball_x = fminf(fmaxf(env->ball_x, 0), env->width - env->ball_width);
            env->ball_y = fminf(fmaxf(env->ball_y, 0), env->height - env->ball_height);
        }
        compute_observations(env);
    }
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    float paddle_width;
    float paddle_height;
    float ball_width;
    float ball_height;
    float x_pad;
    Color paddle_left_color;
    Color paddle_right_color;
    Color ball_color;
    Texture2D ball;
};

Client* make_client(Pong* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->paddle_width = env->paddle_width;
    client->paddle_height = env->paddle_height;
    client->ball_width = env->ball_width;
    client->ball_height = env->ball_height;
    client->x_pad = 3*client->paddle_width;
    client->paddle_left_color = (Color){255, 0, 0, 255};
    client->paddle_right_color = (Color){0, 255, 255, 255};
    client->ball_color = (Color){255, 255, 255, 255};

    InitWindow(env->width + 2*client->x_pad, env->height, "PufferLib Pong");
    SetTargetFPS(60 / env->frameskip);

    client->ball = LoadTexture("resources/shared/puffers_128.png");
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Pong* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Draw left paddle
    DrawRectangle(
        client->x_pad - client->paddle_width,
        client->height - env->paddle_yl - client->paddle_height,
        client->paddle_width,
        client->paddle_height,
        client->paddle_left_color
    );

    // Draw right paddle
    DrawRectangle(
        client->width + client->x_pad,
        client->height - env->paddle_yr - client->paddle_height,
        client->paddle_width,
        client->paddle_height,
        client->paddle_right_color
    );

    // Draw ball
    DrawTexturePro(
        client->ball,
        (Rectangle){
            (env->ball_vx > 0) ? 0 : 128,
            0, 128, 128,
        },
        (Rectangle){
            client->x_pad + env->ball_x,
            client->height - env->ball_y - client->ball_height,
            client->ball_width,
            client->ball_height
        },
        (Vector2){0, 0},
        0,
        WHITE
    );

    //DrawFPS(10, 10);

    // Draw scores
    DrawText(
        TextFormat("%i", env->score_l),
        client->width / 2 + client->x_pad - 50 - MeasureText(TextFormat("%i", env->score_l), 30) / 2,
        10, 30, (Color){0, 187, 187, 255}
    );
    DrawText(
        TextFormat("%i", env->score_r),
        client->width / 2 + client->x_pad + 50 - MeasureText(TextFormat("%i", env->score_r), 30) / 2,
        10, 30, (Color){0, 187, 187, 255}
    );

    EndDrawing();
}
