#ifdef UEFI
unsigned int x = 0;
unsigned int y = 0;
unsigned int text_fg_color = 0xFFFFFFFF;
unsigned int text_bg_color = 0xFF061717;
#include "uefi_compat.h"
Bitmap_Font* font1;
Bitmap_Font* font2;
typedef struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} Color;
char text1[255];
char text2[255];
#else
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif

#define NOOP 0
#define LEFT 1
#define RIGHT 2
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE 1.0f/60.0f

#define BRICK_INDEX_NO_COLLISION -4
#define BRICK_INDEX_SIDEWALL_COLLISION -3
#define BRICK_INDEX_BACKWALL_COLLISION -2
#define BRICK_INDEX_PADDLE_COLLISION -1

bool console_signal = false;

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client {
    float width;
    float height;
    float paddle_width;
    float paddle_height;
    float ball_width;
    float ball_height;
#ifndef UEFI
    Texture2D ball;
#endif
} Client;

typedef struct Breakout {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int score;
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float paddle_x_prev;
    float paddle_y_prev;
    float ball_x_prev;
    float ball_y_prev;
    float ball_vx;
    float ball_vy;
    float* brick_x;
    float* brick_y;
    float* brick_states;
    int balls_fired;
    float initial_paddle_width;
    float paddle_width;
    float paddle_height;
    float paddle_speed;
    float ball_speed;
    float initial_ball_speed;
    float max_ball_speed;
    int hits;
    int width;
    int height;
    int num_bricks;
    int brick_rows;
    int brick_cols;
    int ball_width;
    int ball_height;
    int brick_width;
    int brick_height;
    int num_balls;
    int max_score;
    int half_max_score;
    int score_at_last_ball;
    int tick;
    int frameskip;
    unsigned char hit_brick;
    int continuous;
    unsigned int rng;
    bool barrier;
    unsigned char barrier_timer;
    int origin_x;
    int origin_y;
} Breakout;

typedef struct CollisionInfo CollisionInfo;
struct CollisionInfo {
    float t;
    float overlap;
    float x;
    float y;
    float vx;
    float vy;
    int brick_index;
};

void generate_brick_positions(Breakout*);
void init(Breakout* env);
void allocate(Breakout* env);
void c_close(Breakout* env);
void free_allocated(Breakout* env);
void add_log(Breakout* env);
void compute_observations(Breakout* env);
static inline bool calc_vline_collision(float xw, float yw, float hw, float x, float y, float vx, float vy, float h, CollisionInfo* col);
static inline bool calc_hline_collision(float xw, float yw, float ww, float x, float y, float vx, float vy, float w, CollisionInfo* col);
static inline void calc_brick_collision(Breakout* env, int idx, CollisionInfo* collision_info);
static inline int column_index(Breakout* env, float x);
static inline int row_index(Breakout* env, float y);
void calc_all_brick_collisions(Breakout* env, CollisionInfo* collision_info);
bool calc_paddle_ball_collisions(Breakout* env, CollisionInfo* collision_info);
void calc_all_wall_collisions(Breakout* env, CollisionInfo* collision_info);
void check_wall_bounds(Breakout* env);
void destroy_brick(Breakout* env, int brick_idx);
bool handle_collisions(Breakout* env);
void reset_round(Breakout* env);
void c_reset(Breakout* env);
void step_frame(Breakout* env, float action);
void c_step(Breakout* env);
Client* make_client(Breakout* env);
void close_client(Client* client);
#ifdef UEFI
extern void DrawRectangle(int x, int y, int w, int h, Color color);
#endif
void c_render(Breakout* env);

void generate_brick_positions(Breakout* env) {
    env->half_max_score=0;
    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int idx = row * env->brick_cols + col;
            env->brick_x[idx] = col*env->brick_width;
            env->brick_y[idx] = row*env->brick_height + Y_OFFSET;
            env->half_max_score += 7 - 3 * (idx / env->brick_cols / 2);
        }
    }
    env->max_score=2*env->half_max_score;
}

void init(Breakout* env) {
    env->tick = 0;
    env->num_bricks = env->brick_rows * env->brick_cols;
    assert(env->num_bricks > 0);

    env->brick_x = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_y = (float*)calloc(env->num_bricks, sizeof(float));
    env->brick_states = (float*)calloc(env->num_bricks, sizeof(float));
    env->num_balls = -1;
    generate_brick_positions(env);
    env->barrier = false;
    env->barrier_timer = 10;
    console_signal = false;
}

void allocate(Breakout* env) {
    init(env);
    env->observations = (float*)calloc(11 + env->num_bricks, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
}

void c_close(Breakout* env) {
    free(env->brick_x);
    free(env->brick_y);
    free(env->brick_states);
}

void free_allocated(Breakout* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(Breakout* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)env->max_score;
    env->log.n += 1;
}

void compute_observations(Breakout* env) {
    if (!env->observations || !env->brick_states || env->num_bricks <= 0) return;
    
    env->observations[0] = env->paddle_x / env->width;
    env->observations[1] = env->paddle_y / env->height;
    env->observations[2] = env->ball_x / env->width;
    env->observations[3] = env->ball_y / env->height;
    env->observations[4] = env->ball_vx / 512.0f;
    env->observations[5] = env->ball_vy / 512.0f;
    env->observations[6] = env->balls_fired / 5.0f;
    env->observations[7] = env->score / 864.0f;
    env->observations[8] = env->num_balls / 5.0f;
    env->observations[9] = env->paddle_width / (2.0f * HALF_PADDLE_WIDTH);
    if (env->observations + 10 + env->num_bricks <= env->observations + 11 + env->num_bricks) {
        memcpy(env->observations + 10, env->brick_states, sizeof(float) * env->num_bricks);
    }
}

// Collision of a stationary vertical line segment (xw,yw) to (xw,yw+hw)
// with a moving line segment (x+vx*t,y+vy*t) to (x+vx*t,y+vy*t+h).
static inline bool calc_vline_collision(float xw, float yw, float hw, float x,
        float y, float vx, float vy, float h, CollisionInfo* col) {
    float t_new = (xw - x) / vx;
    float topmost = fminf(yw + hw, y + h + vy * t_new);
    float botmost = fmaxf(yw, y + vy * t_new);
    float overlap_new = topmost - botmost;

    // Collision finds the smallest time of collision with the greatest overlap
    // between the ball and the wall.
    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f  &&
        (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = xw;
        col->y = y + vy * t_new;
        col->vx = -vx;
        col->vy = vy;
        return true;
    }
    return false;
}
static inline bool calc_hline_collision(float xw, float yw, float ww,
        float x, float y, float vx, float vy, float w, CollisionInfo* col) {
    float t_new = (yw - y) / vy;
    float rightmost = fminf(xw + ww, x + w + vx * t_new);
    float leftmost = fmaxf(xw, x + vx * t_new);
    float overlap_new = rightmost - leftmost;

    // Collision finds the smallest time of collision with the greatest overlap between the ball and the wall.
    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f &&
        (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = x + vx * t_new;
        col->y = yw;
        col->vx = vx;
        col->vy = -vy;
        return true;
    }
    return false;
}
static inline void calc_brick_collision(Breakout* env, int idx,
        CollisionInfo* collision_info) {
    bool collision = false;
    // Brick left wall collides with ball right side
    if (env->ball_vx > 0) {
        if (calc_vline_collision(env->brick_x[idx], env->brick_y[idx], env->brick_height,
                env->ball_x + env->ball_width, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height, collision_info)) {
            collision = true;
            collision_info->x -= env->ball_width;
        }
    }

    // Brick right wall collides with ball left side
    if (env->ball_vx < 0) {
        if (calc_vline_collision(env->brick_x[idx] + env->brick_width, env->brick_y[idx], env->brick_height,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height, collision_info)) {
            collision = true;
        }
    }

    // Brick top wall collides with ball bottom side
    if (env->ball_vy > 0) {
        if (calc_hline_collision(env->brick_x[idx], env->brick_y[idx], env->brick_width,
                env->ball_x, env->ball_y + env->ball_height, env->ball_vx, env->ball_vy, env->ball_width, collision_info)) {
            collision = true;
            collision_info->y -= env->ball_height;
        }
    }

    // Brick bottom wall collides with ball top side
    if (env->ball_vy < 0) {
        if (calc_hline_collision(env->brick_x[idx], env->brick_y[idx] + env->brick_height, env->brick_width,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_width, collision_info)) {
            collision = true;
        }
    }
    if (collision) {
        collision_info->brick_index = idx;
    }
}
static inline int column_index(Breakout* env, float x) {
    return (int)(x / env->brick_width);
}
static inline int row_index(Breakout* env, float y) {
    return (int)((y - Y_OFFSET) / env->brick_height);
}

void calc_all_brick_collisions(Breakout* env, CollisionInfo* collision_info) {
    float ball_x = env->ball_x;
    float ball_x_dst = ball_x + env->ball_vx;
    float ball_y = env->ball_y;
    float ball_y_dst = ball_y + env->ball_vy;
    float ball_width = env->ball_width;
    float ball_height = env->ball_height;

    int row_from = row_index(env, ball_y < ball_y_dst ? ball_y : ball_y_dst);
    if (row_from < 0) {
        row_from = 0;
    }

    if (row_from > env->brick_rows) {
        return;
    }

    int column_from = column_index(env, ball_x < ball_x_dst ? ball_x : ball_x_dst);
    if (column_from < 0) {
        column_from = 0;
    }

    float ball_x_end = ball_x + ball_width;
    float ball_x_dst_end = ball_x_dst + ball_width;
    int column_to = column_index(env, ball_x_dst_end > ball_x_end ? ball_x_dst_end : ball_x_end);
    if (column_to >= env->brick_cols) {
        column_to = env->brick_cols - 1;
    }

    float ball_y_end = ball_y + ball_height;
    float ball_y_dst_end = ball_y_dst + ball_height;
    int row_to = row_index(env, ball_y_dst_end > ball_y_end ? ball_y_dst_end : ball_y_end);
    if (row_to >= env->brick_rows) {
        row_to = env->brick_rows - 1;
    }

    for (int row = row_from; row <= row_to; row++) {
        for (int column = column_from; column <= column_to; column++) {
            int brick_index = row * env->brick_cols + column;
            if (env->brick_states[brick_index] == 0.0f)
                calc_brick_collision(env, brick_index, collision_info);
        }
    }
}

bool calc_paddle_ball_collisions(Breakout* env, CollisionInfo* collision_info) {
    float base_angle = M_PI / 4.0f;

    // Check if ball is above the paddle
    if (env->ball_y + env->ball_height + env->ball_vy < env->paddle_y) {
        return false;
    }

    // Check for collision
    // If we've found another collision (eg the ball hits the wall before the paddle)
    // this correctly skips the paddle collision.
    if (!calc_hline_collision(env->paddle_x, env->paddle_y, env->paddle_width,
          env->ball_x, env->ball_y + env->ball_height, env->ball_vx, env->ball_vy, env->ball_width,
          collision_info) || collision_info->t > 1.0f) {
        return false;
    }

    collision_info->y -= env->ball_height;
    collision_info->brick_index = BRICK_INDEX_PADDLE_COLLISION;

    env->hit_brick = false;
    float relative_intersection = (
        (env->ball_x + env->ball_width / 2) - env->paddle_x) / env->paddle_width;
    float angle = -base_angle + relative_intersection * 2 * base_angle;
    env->ball_vx = sinf(angle) * env->ball_speed * TICK_RATE;
    env->ball_vy = -cosf(angle) * env->ball_speed * TICK_RATE;
    env->hits += 1;
    if (env->hits % 4 == 0 && env->ball_speed < env->max_ball_speed) {
        env->ball_speed += 64;
    }
    if (env->score == env->half_max_score) {
        for (int i = 0; i < env->num_bricks; i++) {
            env->brick_states[i] = 0.0;
        }
    }
    return true;
}

void calc_all_wall_collisions(Breakout* env, CollisionInfo* collision_info) {
    if (env->ball_vx < 0) {
        if (calc_vline_collision(0, 0, env->height,
                env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height,
                collision_info)) {
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (env->ball_vx > 0) {
        if (calc_vline_collision(env->width, 0, env->height,
                 env->ball_x + env->ball_width, env->ball_y, env->ball_vx, env->ball_vy, env->ball_height,
                 collision_info)) {
            collision_info->x -= env->ball_width;
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (env->ball_vy < 0) {
        if (calc_hline_collision(0, 0, env->width,
                 env->ball_x, env->ball_y, env->ball_vx, env->ball_vy, env->ball_width,
                 collision_info)) {
            collision_info->brick_index = BRICK_INDEX_BACKWALL_COLLISION;
        }
    }
}

// With rare floating point conditions, the ball could escape the bounds.
// Let's handle that explicitly.
void check_wall_bounds(Breakout* env) {
    float offset = env->max_ball_speed * 1.1f * TICK_RATE;
    if (env->ball_x < 0) {
        env->ball_x += offset;
    }
    if (env->ball_x > env->width) {
        env->ball_x -= offset;
    }
    if (env->ball_y < 0) {
        env->ball_y += offset;
    }
}

void destroy_brick(Breakout* env, int brick_idx) {
    float gained_points = 7 - 3 * ((brick_idx / env->brick_cols) / 2);

    env->score += gained_points;
    env->brick_states[brick_idx] = 1.0;

    env->rewards[0] += gained_points;

    if (brick_idx / env->brick_cols < 3) {
        env->ball_speed = env->max_ball_speed;
    }
}

bool handle_collisions(Breakout* env) {
    CollisionInfo collision_info = {
        .t = 2.0f,
        .overlap = -1.0f,
        .x = 0.0f,
        .y = 0.0f,
        .vx = 0.0f,
        .vy = 0.0f,
        .brick_index = BRICK_INDEX_NO_COLLISION,
    };

    check_wall_bounds(env);

    calc_all_brick_collisions(env, &collision_info);
    calc_all_wall_collisions(env, &collision_info);
    calc_paddle_ball_collisions(env, &collision_info);
    if (collision_info.brick_index != BRICK_INDEX_PADDLE_COLLISION
            && collision_info.t <= 1.0f) {
        env->ball_x = collision_info.x;
        env->ball_y = collision_info.y;
        env->ball_vx = collision_info.vx;
        env->ball_vy = collision_info.vy;
        if (collision_info.brick_index >= 0) {
            destroy_brick(env, collision_info.brick_index);
        }
        if (collision_info.brick_index == BRICK_INDEX_BACKWALL_COLLISION) {
            env->paddle_width = HALF_PADDLE_WIDTH;
        }
    }
    return collision_info.brick_index != BRICK_INDEX_NO_COLLISION;
}

void reset_round(Breakout* env) {
    env->balls_fired = 0;
    env->hit_brick = false;
    env->hits = 0;
    env->ball_speed = env->initial_ball_speed;
    env->paddle_width = env->initial_paddle_width;

    env->paddle_x = env->width / 2.0 - env->paddle_width / 2;
    env->paddle_y = env->height - env->paddle_height - 10;

    env->ball_x = env->paddle_x + (env->paddle_width / 2 - env->ball_width / 2);
    env->ball_y = env->height / 2 - 30;

    env->ball_vx = 0.0;
    env->ball_vy = 0.0;
    if(env->barrier_timer > 0) {
        env->barrier_timer--;
    }
    else {
        //env->barrier = !(env->barrier);
        env->barrier_timer = 10;
    }
}

void c_reset(Breakout* env) {
    env->score = 0;
    env->num_balls = 5;
    env->score_at_last_ball = 0;
    for (int i = 0; i < env->num_bricks; i++) {
        env->brick_states[i] = 0.0;
    }
    reset_round(env);
    env->tick = 0;
    compute_observations(env);
}

void step_frame(Breakout* env, float action) {
    float act = 0.0;
    if (env->balls_fired == 0) {
        env->balls_fired = 1;
        float direction = M_PI / 3.25f;

        env->ball_vy = cosf(direction) * env->ball_speed * TICK_RATE;
        env->ball_vx = sinf(direction) * env->ball_speed * TICK_RATE;
        /*if(rand_r(&env->rng) % 2 == 0) {
            env->ball_vx = -env->ball_vx;
        }*/
        env->ball_vx = -env->ball_vx;
    }
     else if (action == LEFT) {
        act = -1.0;
    } else if (action == RIGHT) {
        act = 1.0;
    }
    if (env->continuous){
        act = action;
    }
    env->paddle_x_prev = env->paddle_x;
    env->paddle_y_prev = env->paddle_y;
    env->paddle_x += act * env->paddle_speed * TICK_RATE;
    if (env->paddle_x <= 0){
        env->paddle_x = fmaxf(0, env->paddle_x);
    } else {
        env->paddle_x = fminf(env->width - env->paddle_width, env->paddle_x);
    }

    env->ball_x_prev = env->ball_x;
    env->ball_y_prev = env->ball_y;

    //Handle collisions.
    //Regular timestepping is done only if there are no collisions.
    if(!handle_collisions(env)){
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;
    }

    if(env->barrier == true) {
        int line_y = (int)env->paddle_y - 50;
        if (env->ball_y > 0 && env->ball_y <= line_y && env->ball_y + env->ball_height > line_y) {
            env->ball_vy = -env->ball_vy;
            //what if env->ball_vy becomes zero?
        }
        /*if(env->tick > 3000) {
            env->barrier = false;
        }*/
    }

    if (env->ball_y >= env->paddle_y + env->paddle_height) {
        env->num_balls -= 1;
        env->score_at_last_ball = env->score;
        reset_round(env);
    }
    if (env->num_balls < 0 || env->score == env->max_score) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
}

void c_step(Breakout* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action = env->actions[0];
    
    // Bounds check action before using
    if (action != LEFT && action != NOOP && action != RIGHT) {
        action = NOOP;
#ifdef UEFI
        env->barrier = true;
    }
    else {
        env->barrier = false;
#endif
    }

    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
    }

    compute_observations(env);
}

//Color BRICK_COLORS[6] = {RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE};

Client* make_client(Breakout* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->paddle_width = env->paddle_width;
    client->paddle_height = env->paddle_height;
    client->ball_width = env->ball_width;
    client->ball_height = env->ball_height;

#ifndef UEFI
    InitWindow(env->width, env->height, "file");
    SetTargetFPS(60 / env->frameskip);

    client->ball = LoadTexture("resources/shared/puffers_128.png");
#endif
    return client;
}

void close_client(Client* client) {
#ifndef UEFI
    CloseWindow();
#endif
    console_signal = true;
    free(client);
}

void c_render(Breakout* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client* client = env->client;

#ifndef UEFI
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){6,24,24,255});

    DrawRectangle(env->paddle_x, env->paddle_y,
        env->paddle_width, env->paddle_height, (Color){0, 255, 255, 255});

    DrawTexturePro(
        client->ball,
        (Rectangle){
            (env->ball_vx > 0) ? 0 : 128,
            0, 128, 128,
        },
        (Rectangle){
            env->ball_x,
            env->ball_y,
            env->ball_width,
            env->ball_height
        },
        (Vector2){0, 0},
        0,
        WHITE
    );
#else
    // Clear and draw paddle
    DrawRectangle(env->origin_x + (int)env->paddle_x_prev, env->origin_y + (int)env->paddle_y_prev, env->paddle_width, env->paddle_height, (Color){6,24,24,255});
    DrawRectangle(env->origin_x + (int)env->paddle_x, env->origin_y + (int)env->paddle_y, env->paddle_width, env->paddle_height, (Color){0,255,255,255});
    // Clear and draw ball
    DrawRectangle(env->origin_x + (int)env->ball_x_prev, env->origin_y + (int)env->ball_y_prev, env->ball_width, env->ball_height, (Color){6,24,24,255});
    DrawRectangle(env->origin_x + (int)env->ball_x, env->origin_y + (int)env->ball_y, env->ball_width, env->ball_height, (Color){0xFF,0xFF,0xFF,0xFF});
#endif

    // Dotted line 50px above paddle (yellow = passthrough, orange = bounce)
    int line_y = (int)env->paddle_y - 50;
    for (int lx = 0; lx < (int)env->width; lx += 10) {
        if(env->barrier) {
            DrawRectangle(env->origin_x + lx, env->origin_y + line_y, 5, 2, (Color){0xFF,0xFF,0x00,0xFF});
        }
        else {
            DrawRectangle(env->origin_x + lx, env->origin_y + line_y, 5, 2, (Color){0x77,0x77,0x00,0xFF});
        }
    }

    // Brick colors by row: RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE
    static const Color BRICK_COLORS[6] = {
        (Color){0xCC,0x22,0x22,0xFF},
        (Color){0xFF,0x88,0x00,0xFF},
        (Color){0xFF,0xFF,0x00,0xFF},
        (Color){0x00,0xFF,0x00,0xFF},
        (Color){0x00,0xCC,0xFF,0xFF},
        (Color){0x44,0x44,0xFF,0xFF}
    };
    for(int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int brick_idx = row * env->brick_cols + col;
            int x = env->brick_x[brick_idx];
            int y = env->brick_y[brick_idx];
            if (env->brick_states[brick_idx] == 1) {
                DrawRectangle(env->origin_x + x, env->origin_y + y, env->brick_width, env->brick_height, (Color){6,24,24,255});
                continue;
            }
            Color brick_color = BRICK_COLORS[row];
            DrawRectangle(env->origin_x + x, env->origin_y + y, env->brick_width, env->brick_height, brick_color);
        }
    }
#ifndef UEFI
    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Balls: %i", env->num_balls), client->width - 80, 10, 20, WHITE);
    EndDrawing();
#else
    int destroyed = 0;
    for (int i = 0; i < env->num_bricks; i++) {
        if (env->brick_states[i] == 1) destroyed++;
    }

    //sprintf(text2, "a0:%d bx:%d by:%d px:%d py:%d\n", (int)env->actions[0], (int)(env->observations[2]*100), (int)(env->observations[3]*100), (int)(env->observations[0]*100), (int)(env->observations[1]*100));
    //y = env->height;
    //x = 0;
    //print_string(text2, font1); //secondary
    sprintf(text1,"sc:%u ff:%d dst:%d t:%d xy:%d %d       \n",
        env->score, env->balls_fired, destroyed,
        (int)env->tick, (int)env->ball_x, (int)env->ball_y);
    x = env->origin_x / 2;
    y = env->origin_y / 2;
    print_string(text1, font1);
    //print_string("testing font2", font2);
#endif
}
