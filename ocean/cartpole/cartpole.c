// local compile/eval implemented for discrete actions only
// eval with python demo.py --mode eval --env puffer_cartpole --eval-mode-path <path to model>
#ifndef UEFI
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#endif
#include "cartpole.h"
#include "puffernet.h"
#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 2
#define CONTINUOUS 0

#ifdef UEFI
#define arch_header <arch/ARCH/ARCH.h>
#include arch_header

#include <stdnoreturn.h>

// UEFI framebuffer - global variables (initialized in kmain)
uint32_t* fb;
uint32_t xres;
uint32_t yres;

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

#endif

const char* WEIGHTS_PATH = "resources/cartpole/cartpole_weights.bin";

#ifndef UEFI
float movement(float action, int userControlMode) {
    if (userControlMode) {
        return (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) ? 1.0f : -1.0f;
    } else {
        return (action > 0.5f) ? 1.0f : -1.0f;
    }
}
#else
__attribute__((section(".kernel"), aligned(0x1000)))
noreturn void EFIAPI kmain(Kernel_Parms *kargs) {
    UINT64 fb_base = (UINT64)kargs->gop_mode.FrameBufferBase;
    UINTN w = kargs->gop_mode.Info->HorizontalResolution;

    font1 = &kargs->fonts[0];
    font2 = &kargs->fonts[1];

    if (fb_base != 0 && w > 0) {
        UINTN h = kargs->gop_mode.Info->VerticalResolution;
        fb = (uint32_t*)fb_base;
        xres = w;
        yres = h;
        bool duration = true;
        while(duration) {
            extern unsigned char resources_cartpole_cartpole_weights_bin[];
            extern unsigned int resources_cartpole_cartpole_weights_bin_len;
            Weights* weights1 = load_weightsEmbedded(resources_cartpole_cartpole_weights_bin, resources_cartpole_cartpole_weights_bin_len);
            Weights* weights2 = load_weightsEmbedded(resources_cartpole_cartpole_weights_bin, resources_cartpole_cartpole_weights_bin_len);
            Weights* weights3 = load_weightsEmbedded(resources_cartpole_cartpole_weights_bin, resources_cartpole_cartpole_weights_bin_len);
            Weights* weights4 = load_weightsEmbedded(resources_cartpole_cartpole_weights_bin, resources_cartpole_cartpole_weights_bin_len);
            Weights* weights5 = load_weightsEmbedded(resources_cartpole_cartpole_weights_bin, resources_cartpole_cartpole_weights_bin_len);
            Weights* weights6 = load_weightsEmbedded(resources_cartpole_cartpole_weights_bin, resources_cartpole_cartpole_weights_bin_len);

            int logit_sizes[1] = {ACTIONS_SIZE};
            PufferNet* net = make_puffernet(weights1, 1, OBSERVATIONS_SIZE, 64, 2, logit_sizes, 1);
            PufferNet* net2 = make_puffernet(weights2, 1, OBSERVATIONS_SIZE, 64, 2, logit_sizes, 1);
            PufferNet* net3 = make_puffernet(weights3, 1, OBSERVATIONS_SIZE, 64, 2, logit_sizes, 1);
            PufferNet* net4 = make_puffernet(weights4, 1, OBSERVATIONS_SIZE, 64, 2, logit_sizes, 1);
            PufferNet* net5 = make_puffernet(weights5, 1, OBSERVATIONS_SIZE, 64, 2, logit_sizes, 1);
            PufferNet* net6 = make_puffernet(weights6, 1, OBSERVATIONS_SIZE, 64, 2, logit_sizes, 1);

            Cartpole env1 = {
                .continuous = CONTINUOUS,
                .cart_mass = 1.0f,
                .pole_mass = 0.1f,
                .pole_length = 0.5f,
                .gravity = 9.8f,
                .force_mag = 10.0f,
                .tau = 0.02f,
                .origin_x = 0, .origin_y = 0,
            };
            Cartpole env2 = {
                .continuous = CONTINUOUS,
                .cart_mass = 1.0f,
                .pole_mass = 0.1f,
                .pole_length = 0.5f,
                .gravity = 9.8f,
                .force_mag = 10.0f,
                .tau = 0.02f,
                .origin_x = WIDTH + 30, .origin_y = 0,
            };
            Cartpole env3 = {
                .continuous = CONTINUOUS,
                .cart_mass = 1.0f,
                .pole_mass = 0.1f,
                .pole_length = 0.5f,
                .gravity = 9.8f,
                .force_mag = 10.0f,
                .tau = 0.02f,
                .origin_x = 2*(WIDTH + 30), .origin_y = 0,
            };
            Cartpole env4 = {
                .continuous = CONTINUOUS,
                .cart_mass = 1.0f,
                .pole_mass = 0.1f,
                .pole_length = 0.5f,
                .gravity = 9.8f,
                .force_mag = 10.0f,
                .tau = 0.02f,
                .origin_x = 0, .origin_y = HEIGHT + 20,
            };
            Cartpole env5 = {
                .continuous = CONTINUOUS,
                .cart_mass = 1.0f,
                .pole_mass = 0.1f,
                .pole_length = 0.5f,
                .gravity = 9.8f,
                .force_mag = 10.0f,
                .tau = 0.02f,
                .origin_x = WIDTH + 30, .origin_y = HEIGHT + 20,
            };
            Cartpole env6 = {
                .continuous = CONTINUOUS,
                .cart_mass = 1.0f,
                .pole_mass = 0.1f,
                .pole_length = 0.5f,
                .gravity = 9.8f,
                .force_mag = 10.0f,
                .tau = 0.02f,
                .origin_x = 2*(WIDTH + 30), .origin_y = HEIGHT + 20,
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
#endif

#ifndef UEFI
void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH);
    
    int logit_sizes[1] = {ACTIONS_SIZE};
    PufferNet* net = make_puffernet(weights, 1, OBSERVATIONS_SIZE, 32, 2, logit_sizes, 1);
    
    Cartpole env = {
        .continuous = CONTINUOUS,
        .cart_mass = 1.0f,
        .pole_mass = 0.1f,
        .pole_length = 0.5f,
        .gravity = 9.8f,
        .force_mag = 10.0f,
        .tau = 0.02f,
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        int userControlMode = IsKeyDown(KEY_LEFT_SHIFT);

        if (!userControlMode) {
            forward_puffernet(net, env.observations, env.actions);
            env.actions[0] = movement(env.actions[0], 0);
        } else {
            env.actions[0] = movement(env.actions[0], userControlMode);
        }   

        c_step(&env);
        c_render(&env);

        if (env.terminals[0] > 0.5f) {
            c_reset(&env);
        }
    }

    free_puffernet(net);
    free(weights);
    free_allocated(&env);
}
#endif
int main() {
#ifndef UEFI
    srand(time(NULL));
    demo();
#endif
    return 0;
}
