#ifndef UEFI
#include <time.h>
#endif
#include "breakout.h"
#include "puffernet.h"

#ifdef UEFI

#define arch_header <arch/ARCH/ARCH.h>
#include arch_header

#include <stdnoreturn.h>

// UEFI framebuffer - global variables (initialized in kmain)
uint32_t* fb;
uint32_t xres;
uint32_t yres;

void DrawRectangle(int x, int y, int w, int h, Color color) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            unsigned int px = x + col;
            unsigned int py = y + row;
            if (py >= 0 && py < yres && px >= 0 && px < xres) {
                fb[py*xres + px] = (color.a << 24) | (color.r << 16) | (color.g << 8) | color.b;
            }
        }
    }
}

void forward_net(PufferNet* net, float* observations, float* actions, bool use_rnd) {
    linear(net->encoder, observations);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    if (net->is_continuous) {
        _gaussian_mean(net->decoder->output, actions, net->num_agents, net->num_actions);
    } else {
        if(use_rnd) {
            softmax_multidiscrete(net->multidiscrete, net->decoder->output, actions);
        }
        else {
            argmax_multidiscrete(net->multidiscrete, net->decoder->output, actions);
        }
    }
}

__attribute__((section(".kernel"), aligned(0x1000)))
noreturn void EFIAPI kmain(Kernel_Parms *kargs) {
    UINT64 fb_base = (UINT64)kargs->gop_mode.FrameBufferBase;
    UINTN pitch = kargs->gop_mode.Info->PixelsPerScanLine;
    UINTN w = kargs->gop_mode.Info->HorizontalResolution;

    font1 = &kargs->fonts[0];
    font2 = &kargs->fonts[1];

    if (fb_base != 0 && w > 0) {
        volatile UINT32 *test_fb = (volatile UINT32*)fb_base;
        UINTN h = kargs->gop_mode.Info->VerticalResolution;
        fb = (uint32_t*)fb_base;
        xres = w;
        yres = h;
        bool duration = true;
        while(duration) {
            extern unsigned char resources_breakout_breakout_weights_bin[];
            extern unsigned int resources_breakout_breakout_weights_bin_len;
            Weights* weights1 = load_weightsEmbedded(resources_breakout_breakout_weights_bin, resources_breakout_breakout_weights_bin_len);
            Weights* weights2 = load_weightsEmbedded(resources_breakout_breakout_weights_bin, resources_breakout_breakout_weights_bin_len);
            Weights* weights3 = load_weightsEmbedded(resources_breakout_breakout_weights_bin, resources_breakout_breakout_weights_bin_len);
            Weights* weights4 = load_weightsEmbedded(resources_breakout_breakout_weights_bin, resources_breakout_breakout_weights_bin_len);
            Weights* weights5 = load_weightsEmbedded(resources_breakout_breakout_weights_bin, resources_breakout_breakout_weights_bin_len);
            Weights* weights6 = load_weightsEmbedded(resources_breakout_breakout_weights_bin, resources_breakout_breakout_weights_bin_len);

            int logit_sizes[1] = {3};
            PufferNet* net = make_puffernet(weights1, 1, 118, 64, 2, logit_sizes, 1);
            PufferNet* net2 = make_puffernet(weights2, 1, 118, 64, 2, logit_sizes, 1);
            PufferNet* net3 = make_puffernet(weights3, 1, 118, 64, 2, logit_sizes, 1);
            PufferNet* net4 = make_puffernet(weights4, 1, 118, 64, 2, logit_sizes, 1);
            PufferNet* net5 = make_puffernet(weights5, 1, 118, 64, 2, logit_sizes, 1);
            PufferNet* net6 = make_puffernet(weights6, 1, 118, 64, 2, logit_sizes, 1);

            Breakout env1 = {
                .frameskip = 1,
                .width = 576, .height = 330,
                .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
                .ball_width = 32, .ball_height = 32,
                .brick_width = 32, .brick_height = 12,
                .brick_rows = 6, .brick_cols = 18,
                .initial_ball_speed = 256, .max_ball_speed = 448,
                .paddle_speed = 620, .continuous = 0,
                .origin_x = 0, .origin_y = 0,
            };
            Breakout env2 = {
                .frameskip = 1,
                .width = 576, .height = 330,
                .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
                .ball_width = 32, .ball_height = 32,
                .brick_width = 32, .brick_height = 12,
                .brick_rows = 6, .brick_cols = 18,
                .initial_ball_speed = 256, .max_ball_speed = 448,
                .paddle_speed = 620, .continuous = 0,
                .origin_x = 606, .origin_y = 0,
            };
            Breakout env3 = {
                .frameskip = 1,
                .width = 576, .height = 330,
                .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
                .ball_width = 32, .ball_height = 32,
                .brick_width = 32, .brick_height = 12,
                .brick_rows = 6, .brick_cols = 18,
                .initial_ball_speed = 256, .max_ball_speed = 448,
                .paddle_speed = 620, .continuous = 0,
                .origin_x = 1212, .origin_y = 0,
            };
            Breakout env4 = {
                .frameskip = 1,
                .width = 576, .height = 330,
                .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
                .ball_width = 32, .ball_height = 32,
                .brick_width = 32, .brick_height = 12,
                .brick_rows = 6, .brick_cols = 18,
                .initial_ball_speed = 256, .max_ball_speed = 448,
                .paddle_speed = 620, .continuous = 0,
                .origin_x = 0, .origin_y = 350,
            };
            Breakout env5 = {
                .frameskip = 1,
                .width = 576, .height = 330,
                .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
                .ball_width = 32, .ball_height = 32,
                .brick_width = 32, .brick_height = 12,
                .brick_rows = 6, .brick_cols = 18,
                .initial_ball_speed = 256, .max_ball_speed = 448,
                .paddle_speed = 620, .continuous = 0,
                .origin_x = 606, .origin_y = 350,
            };
            Breakout env6 = {
                .frameskip = 1,
                .width = 576, .height = 330,
                .initial_paddle_width = 62, .paddle_width = 62, .paddle_height = 8,
                .ball_width = 32, .ball_height = 32,
                .brick_width = 32, .brick_height = 12,
                .brick_rows = 6, .brick_cols = 18,
                .initial_ball_speed = 256, .max_ball_speed = 448,
                .paddle_speed = 620, .continuous = 0,
                .origin_x = 1212, .origin_y = 350,
            };

            allocate(&env1);
            allocate(&env2);
            allocate(&env3);
            allocate(&env4);
            allocate(&env5);
            allocate(&env6);

            env1.client = make_client(&env1);
            env2.client = make_client(&env2);
            env3.client = make_client(&env3);
            env4.client = make_client(&env4);
            env5.client = make_client(&env5);
            env6.client = make_client(&env6);

            c_reset(&env1);
            c_reset(&env2);
            c_reset(&env3);
            c_reset(&env4);
            c_reset(&env5);
            c_reset(&env6);

            for (int y = 0; y < yres; y++)
                for (int x = 0; x < xres; x++)
                    fb[y*xres + x] = 0xFF061717;

            x = 606 / 2;
            y = 700 / 2;
            print_string("Figure 1: three softmax above three argmax", font1);

            int frame = 0;
            while (!console_signal) {
                if (frame % 4 == 0) {
                    forward_net(net, env1.observations, env1.actions, true);
                    forward_net(net2, env2.observations, env2.actions, true);
                    forward_net(net3, env3.observations, env3.actions, true);
                    forward_net(net4, env4.observations, env4.actions, false);
                    forward_net(net5, env5.observations, env5.actions, false);
                    forward_net(net6, env6.observations, env6.actions, false);
                }

                frame++;
                c_step(&env1);
                c_step(&env2);
                c_step(&env3);
                c_step(&env4);
                c_step(&env5);
                c_step(&env6);
                c_render(&env1);
                c_render(&env2);
                c_render(&env3);
                c_render(&env4);
                c_render(&env5);
                c_render(&env6);
                if(frame > (int)10000) {
                    break;
                }
            }
            free_puffernet(net);
            free_puffernet(net2);
            free_puffernet(net3);
            free_puffernet(net4);
            free_puffernet(net5);
            free_puffernet(net6);
            free(weights1);
            free(weights2);
            free(weights3);
            free(weights4);
            free(weights5);
            free(weights6);
            free_allocated(&env1);
            free_allocated(&env2);
            free_allocated(&env3);
            free_allocated(&env4);
            free_allocated(&env5);
            free_allocated(&env6);
            close_client(env1.client);
            close_client(env2.client);
            close_client(env3.client);
            close_client(env4.client);
            close_client(env5.client);
            close_client(env6.client);
        }
    }
    for (y = 0; y < yres; y++)
    for (x = 0; x < xres; x++)
        fb[y*xres + x] = 0x00000000;
}
#endif //ifdef UEFI

void demo() {
    Weights* weights = load_weights("resources/breakout/breakout_weights.bin");
    int logit_sizes[1] = {3};
    PufferNet* net = make_puffernet(weights, 1, 118, 64, 2, logit_sizes, 1);

    Breakout env = {
        .frameskip = 1,
        .width = 576,
        .height = 330,
        .initial_paddle_width = 62,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
        .initial_ball_speed = 256,
        .max_ball_speed = 448,
        .paddle_speed = 620,
        .continuous = 0,
    };
    allocate(&env);

    env.client = make_client(&env);

    c_reset(&env);
    int frame = 0;
#ifndef UEFI
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        // User can take control of the paddle
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_SPACE)) {
                env.barrier = true;
            }
            else {
                env.barrier = false;
            }
            if(env.continuous) {
                float move = GetMouseWheelMove();
                float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
                env.actions[0] = clamped_wheel;
            } else {
                env.actions[0] = 0.0;
                if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 1;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 2;
            }
        } else if (frame % 4 == 0) {
#else
    while (!console_signal) {
        if (frame % 4 == 0) {
#endif
            forward_puffernet(net, env.observations, env.actions);
        }

        frame = (frame + 1) % 4;
        c_step(&env);
        c_render(&env);
    }
    free_puffernet(net);
    free(weights);
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}
