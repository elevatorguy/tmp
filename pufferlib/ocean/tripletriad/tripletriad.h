#include <stdlib.h>
#include <math.h>
#include "raylib.h"
#include <stdio.h>

#define SELECT_CARD_1 0
#define SELECT_CARD_2 1
#define SELECT_CARD_3 2
#define SELECT_CARD_4 3
#define SELECT_CARD_5 4
#define PLACE_CARD_1 5
#define PLACE_CARD_2 6
#define PLACE_CARD_3 7
#define PLACE_CARD_4 8
#define PLACE_CARD_5 9
#define PLACE_CARD_6 10
#define PLACE_CARD_7 11
#define PLACE_CARD_8 12
#define PLACE_CARD_9 13
#define TICK_RATE 1.0f/60.0f
#define MAX_EPISODE_LENGTH 30

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// how to start game compile - LD_LIBRARY_PATH=raylib-5.0_linux_amd64/lib ./tripletriadgame 

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct CTripleTriad CTripleTriad;
struct CTripleTriad {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
    int card_width;
    int card_height;
    float* board_x;
    float* board_y;
    int** board_states;
    int width;
    int height;
    int game_over;
    int num_cards;
    int*** cards_in_hand;
    int* card_selected;
    int** card_locations;
    int* action_masks;
    int*** board_card_values;
    int* score;
    int tick;
    float perf;
    float episode_return;
    float episode_length;
    Client* client;
};

void add_log(CTripleTriad* env) {
    env->log.perf += env->perf;
    env->log.score += env->score[0];
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->episode_length;
    env->log.n += 1;
}

void generate_board_positions(CTripleTriad* env) {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int idx = row * 3 + col;
            env->board_x[idx] = col* env->card_width;
            env->board_y[idx] = row*env->card_height;
        }
    }
}

void generate_cards_in_hand(CTripleTriad* env) {
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            for(int k=0; k< 4; k++) {
                env->cards_in_hand[i][j][k] = (rand() % 7) + 1;
            }
        }
    }
}

void generate_card_locations(CTripleTriad* env) {
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            env->card_locations[i][j] = 0;
        }
    }
}

void generate_card_selections(CTripleTriad* env) {
    for(int i=0; i< 2; i++) {
        env->card_selected[i] = -1;
    }
}

void generate_board_states(CTripleTriad* env) {
    for(int i=0; i< 3; i++) {
        for(int j=0; j< 3; j++) {
            env->board_states[i][j] = 0;
        }
    }
}

void generate_board_card_values(CTripleTriad* env) {
    for(int i=0; i< 3; i++) {
        for(int j=0; j< 3; j++) {
            for(int k=0; k< 4; k++) {
                env->board_card_values[i][j][k] = 0;
            }
        }
    }
}

void generate_scores(CTripleTriad* env) {
    for(int i=0; i< 2; i++) {
        env->score[i] = 5;
    }
}

void init_ctripletriad(CTripleTriad* env) {
    // Allocate memory for board_x, board_y, and board_states
    env->board_x = (float*)calloc(9, sizeof(float));
    env->board_y = (float*)calloc(9, sizeof(float));
    env->cards_in_hand = (int***)calloc(2, sizeof(int**));
    env->card_selected = (int*)calloc(2, sizeof(int));
    env->card_locations = (int**)calloc(2, sizeof(int*));
    env->action_masks = (int*)calloc(15, sizeof(int));
    env->board_states = (int**)calloc(3, sizeof(int*));
    env->board_card_values = (int***)calloc(3, sizeof(int**));
    env->score = (int*)calloc(2, sizeof(int));
    for(int i=0; i< 2; i++) {
        env->cards_in_hand[i] = (int**)calloc(5, sizeof(int*));
        for(int j=0; j< 5; j++) {
            env->cards_in_hand[i][j] = (int*)calloc(4, sizeof(int));
        }
    }
    for(int i=0; i< 3; i++) {
        env->board_states[i] = (int*)calloc(3, sizeof(int));
    }
    for(int i=0; i< 2; i++) {
        env->card_locations[i] = (int*)calloc(5, sizeof(int));
    }
    for(int i=0; i< 3; i++) {
        env->board_card_values[i] = (int**)calloc(3, sizeof(int*));
        for(int j=0; j< 3; j++) {
            env->board_card_values[i][j] = (int*)calloc(4, sizeof(int));
        }
    }
    generate_board_positions(env);
    generate_cards_in_hand(env);
    generate_card_locations(env);
    generate_card_selections(env);
    generate_board_states(env);
    generate_board_card_values(env);
    generate_scores(env);
}

void allocate_ctripletriad(CTripleTriad* env) {
    env->actions = (int*)calloc(1, sizeof(int));
    env->observations = (float*)calloc(env->width*env->height, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    init_ctripletriad(env);
}

void c_close(CTripleTriad* env) {
    free(env->board_x);
    free(env->board_y);
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            free(env->cards_in_hand[i][j]);
        }
        free(env->cards_in_hand[i]);
        free(env->card_locations[i]);
    }
    free(env->cards_in_hand);
    free(env->card_locations);
    free(env->card_selected);
    free(env->action_masks);
    for(int i=0; i< 3; i++) {
        free(env->board_states[i]);
    }
    free(env->board_states);
    for(int i=0; i< 3; i++) {
        for(int j=0; j< 3; j++) {
            free(env->board_card_values[i][j]);
        }
        free(env->board_card_values[i]);
    }
    free(env->board_card_values);
    free(env->score);
}

void free_allocated_ctripletriad(CTripleTriad* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void compute_observations(CTripleTriad* env) {
    int idx=0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            env->observations[idx] = env->board_states[i][j];
            idx++;
        }
    }
    for (int i = 0; i < 15; i++) {
        env->observations[idx] = env->action_masks[i];
        idx++;
    }

    for (int i = 0; i < 2; i++) {
        env->observations[idx] = env->card_selected[i];
        idx++;
    }
    for (int i = 0; i < 2; i++) {
        env->observations[idx] = env->score[i];
        idx++;
    }
    for (int i=0;i<3;i++) {
        for (int j=0;j<3;j++) {
            for (int k=0;k<4;k++) {
                env->observations[idx] = env->board_card_values[i][j][k];
                idx++;
            }
        }
    }
    for (int i=0;i<2;i++){
        for (int j=0;j<5;j++) {
            for (int k=0;k<4;k++) {
                env->observations[idx] = env->cards_in_hand[i][j][k];
                idx++;
            }
        }
    }
    for (int i=0;i<2;i++) {
        for (int j=0;j<5;j++) {
            env->observations[idx] = env->card_locations[i][j];
            idx++;
        }
    }
}

void c_reset(CTripleTriad* env) {
    env->game_over = 0;
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            for(int k=0; k< 4; k++) {
                env->cards_in_hand[i][j][k] = (rand() % 7) + 1;
            }
        }
    }
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            env->card_locations[i][j] = 0;
        }
    }
    for(int i=0; i< 2; i++) {
        env->card_selected[i] = -1;
    }
    for(int i=0; i< 3; i++) {
        for(int j=0; j< 3; j++) {
            env->board_states[i][j] = 0;
        }
    }
    for (int i = 0; i < 15; i++) {
        env->action_masks[i] = 0;
    }
    for (int i=0; i< 3; i++) {
        for (int j=0; j< 3; j++) {
            for (int k=0; k< 4; k++) {
                env->board_card_values[i][j][k] = 0;
            }
        }
    }
    for(int i=0; i< 2; i++) {
        env->score[i] = 5;
    }
    env->terminals[0] = 0;
    compute_observations(env);
    env->tick = 0;
    env->episode_length = 0;
    env->episode_return = 0;
}

void select_card(CTripleTriad* env, int card_selected, int player) {
    int player_idx = (player == 1) ? 0 : 1;
    env->card_selected[player_idx] = card_selected-1;
}

void place_card(CTripleTriad* env, int card_placement, int player) {
    // Determine the player index (0 for player 1, 1 for player 2)
    int player_idx = (player == 1) ? 0 : 1;
    // Update the card's location on the board
    env->card_locations[player_idx][env->card_selected[player_idx]] = card_placement;
    // Update the board state to reflect the player who placed the card
    env->board_states[(card_placement-1)/3][(card_placement-1)%3] = player;
    // Copy the card values from the player's hand to the board
    for (int i = 0; i < 4; i++) {
        env->board_card_values[(card_placement-1)/3][(card_placement-1)%3][i] = env->cards_in_hand[player_idx][env->card_selected[player_idx]][i];
    }
}

void update_action_masks(CTripleTriad* env) {
    // First, reset all action masks to 0 (available)
    for (int i = 0; i < 15; i++) {
        env->action_masks[i] = 0;
    }

    // Update masks for card placement
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            if (env->card_locations[i][j] != 0) {
                int action_idx = env->card_locations[i][j] + 4;
                if (action_idx >= 5 && action_idx < 14) {
                    env->action_masks[action_idx] = 1;  // Mark as unavailable
                }
            }
        }
    }
}

void check_win_condition(CTripleTriad* env, int player) {
    int count = 0;
    for (int i=0; i< 3; i++) {
        for (int j=0; j< 3; j++) {
            if (env->board_states[i][j] !=0) {
                count++;
            } 
        }
    }
    if (count == 9) {
        // add a draw condition and winner value is 0
        if (env->score[0] == env->score[1]) {
            env->terminals[0] = 1;
            env->rewards[0] = 0.0;
            env->game_over = 1;
        } else {
            int winner = env->score[0] > env->score[1] ? 1 : -1;
            env->terminals[0] = 1;
            env->rewards[0] = winner; // 1 for player win, -1 for opponent win
            env->episode_return += winner;
            env->game_over = 1;
        }
    }
    return;
}

int get_bot_card_placement(CTripleTriad* env) {
    int valid_placements[9];  // Maximum 9 possible placements
    int num_valid_placements = 0;

    // Find valid placements
    for (int i = 5; i < 14; i++) {
        if (env->action_masks[i] == 0) {
            valid_placements[num_valid_placements++] = i - 4;
            if (num_valid_placements == 9) break;  // Safety check
        }
    }
    
    // Randomly select a valid placement
    if (num_valid_placements > 0) {
        return valid_placements[rand() % num_valid_placements];
    }

    // If no valid placements, return 0 (this should not happen in a normal game)
    return 0;
}

int get_bot_card_selection(CTripleTriad* env) {
    int valid_selections[5];  // Maximum 5 possible selections
    int num_valid_selections = 0;

    // Find valid selections
    for (int i = 0; i < 5; i++) {
        if (env->card_locations[1][i] == 0) {  // Check if the card has not been placed
            valid_selections[num_valid_selections++] = i + 1;
        }
    }

    // Randomly select a valid card
    if (num_valid_selections > 0) {
        return valid_selections[rand() % num_valid_selections];
    }

    // If no valid selections, return 0 (this should not happen in a normal game)
    return 0;
}

bool check_legal_placement(CTripleTriad* env, int card_placement, int player) {
    int row = (card_placement - 1) / 3;
    int col = (card_placement - 1) % 3;
    if (env->board_states[row][col] != 0) {
        return 0;
    } else {
        return 1;
    }
}

void check_card_conversions(CTripleTriad* env, int card_placement, int player) {
    // Given that card locations last 4 values of the most inner array are organized, NSEW. 
    // Check the cards in those directions and what their values are 
    // If the card in the direction is greater than the card in the current location, then convert the game state of the lower value card to the player with the higher value card.
    int card_idx = card_placement - 1;
    int row = card_idx / 3;
    int col = card_idx % 3;
    int enemy_player = (player == 1) ? -1 : 1;
    int player_idx = (player == 1) ? 0 : 1;
    int enemy_player_idx = (player == 1) ? 1 : 0;
    int values[4] = {
        env->board_card_values[row][col][0],  // North
        env->board_card_values[row][col][1],  // South
        env->board_card_values[row][col][2],  // East
        env->board_card_values[row][col][3]   // West
    };

    int adjacent_indices[4][2] = {
        {row - 1, col},  // North
        {row + 1, col},  // South
        {row, col + 1},  // East
        {row, col - 1}   // West
    };

    int adjacent_value_indices[4] = {1, 0, 3, 2};  // South, North, West, East

    for (int i = 0; i < 4; i++) {
        int adj_row = adjacent_indices[i][0];
        int adj_col = adjacent_indices[i][1];

        // Check if the adjacent cell is within the board
        if (adj_row >= 0 && adj_row < 3 && adj_col >= 0 && adj_col < 3) {
            int adjacent_value = env->board_card_values[adj_row][adj_col][adjacent_value_indices[i]];
            if (adjacent_value < values[i] && adjacent_value != 0 && env->board_states[adj_row][adj_col] == enemy_player) {
                env->board_states[adj_row][adj_col] = player;
                env->score[player_idx]++;
                env->score[enemy_player_idx]--;
            }
        }
    }
}

void c_step(CTripleTriad* env) {
    env->episode_length += 1;
    env->rewards[0] = 0.0;
    int action = env->actions[0];

    if (env->episode_length >= MAX_EPISODE_LENGTH) {
        env->game_over = 1;
        env->episode_return -= 1.0;
        env->rewards[0] -= 1.0;
    }

    // reset the game if game over
    if (env->game_over == 1) {
        env->perf = (env->score[0] > env->score[1]) ? 1.0 : 0.0;
        add_log(env);
        c_reset(env);
        return;
    }
    // select a card if the card is in the range of 1-5 and the card is not placed
    if (action >= SELECT_CARD_1 && action <= SELECT_CARD_5 ) {
        // Prevent model from just swapping between selected cards to avoid playing
        env->episode_return -= 0.1;
        env->rewards[0] -= 0.1;

        int card_selected = action + 1;
        
        if(env->card_locations[0][card_selected-1] == 0) {
            select_card(env, card_selected, 1);
        }
    }
    // place a card if the card is in the range of 1-9 and the card is selected
    else if (action >= PLACE_CARD_1 && action <= PLACE_CARD_9  ) {
        int card_placement = action - 4;
        bool card_placed = false;
        if(env->card_selected[0] >= 0) {
            if(check_legal_placement(env, card_placement, 1)) {
                place_card(env,card_placement, 1);
                check_card_conversions(env, card_placement, 1);
                check_win_condition(env, 1);
                update_action_masks(env);
                env->card_selected[0] = -1;
                card_placed = true;
            } else {
                env->episode_return -= 0.1;
                env->rewards[0] -= 0.1;
            }
        } else {
            env->episode_return -= 0.1;
            env->rewards[0] -= 0.1;
        }

        // opponent turn 
        if (env->terminals[0] == 0 && card_placed == true ) {
            int bot_card_selected = get_bot_card_selection(env);
            if(bot_card_selected > 0) {
                select_card(env,bot_card_selected, -1);
                int bot_card_placement = get_bot_card_placement(env);
                place_card(env,bot_card_placement, -1);
                check_card_conversions(env, bot_card_placement, -1);
                check_win_condition(env, -1);
                update_action_masks(env);
                env->card_selected[1] = -1;
            }
            
        }
    }
    if (env->terminals[0] == 1) {
        env->game_over=1;
    }
    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
};

Client* make_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;

    InitWindow(width, height, "PufferLib Ray TripleTriad");
    SetTargetFPS(60);

    return client;
}

void c_render(CTripleTriad* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = make_client(env->width, env->height);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // create 3x3 board for triple triad
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int board_idx = row * 3 + col;
            Color piece_color=PURPLE;
            if (env->board_states[row][col] == 0.0) {
                piece_color = PUFF_BACKGROUND;
            } else if (env->board_states[row][col] == 1.0) {
                piece_color = PUFF_CYAN;
            } else if (env->board_states[row][col] == -1.0) {
                piece_color = PUFF_RED;
            }
            int x = env->board_x[board_idx];
            int y = env->board_y[board_idx];
            DrawRectangle(x+196+10 , y+10 , env->card_width, env->card_height, piece_color);
            DrawRectangleLines(x+196+10 , y+10 , env->card_width, env->card_height, PUFF_WHITE);
        }
    }
    for(int i=0; i< 2; i++) {
        for(int j=0; j< 5; j++) {
            // starting locations for cards in hand
            int card_x = (i == 0) ? 10 : (env->width - env->card_width - 10);
            int card_y = 10 + env->card_height/2*j;

            // locations if card is placed
            if (env->card_locations[i][j] != 0) {
                card_x = env->board_x[env->card_locations[i][j]-1] + 196 + 10;
                card_y = env->board_y[env->card_locations[i][j]-1] + 10;
            }
            // Draw card background
            // adjusts card color based on board state 
            Color card_color = (i != 0) ? PUFF_RED : PUFF_CYAN;
            // check if index is in bounds first    
            if (env->card_locations[i][j] != 0) {
                if (env->board_states[(env->card_locations[i][j]-1)/3][(env->card_locations[i][j]-1)%3] == -1) {
                    card_color = PUFF_RED;
                } else if (env->board_states[(env->card_locations[i][j]-1)/3][(env->card_locations[i][j]-1)%3] == 1) {
                    card_color = PUFF_CYAN;
                } else {
                    card_color = (i != 0) ? PUFF_CYAN : PUFF_RED;
                }
            }
            DrawRectangle(card_x, card_y, env->card_width, env->card_height, card_color);
            // change background if card is selected, highlight it
            Rectangle rect = (Rectangle){card_x, card_y, env->card_width, env->card_height};
            if (env->card_selected[i] == j) {
                DrawRectangleLinesEx(rect, 3, PUFF_RED);
            } else {
                DrawRectangleLinesEx(rect, 2, PUFF_WHITE);
            }
        
            for(int k=0; k< 4; k++) {
                int x_offset, y_offset;
                switch(k) {
                    case 0: // North (top)
                        x_offset = card_x + 25;
                        y_offset = card_y + 5;
                        break;
                    case 1: // South (bottom)
                        x_offset = card_x + 25;
                        y_offset = card_y + 45;
                        break;
                    case 2: // East (right)
                        x_offset = card_x + 45;
                        y_offset = card_y + 25;
                        break;
                    case 3: // West (left)
                        x_offset = card_x + 5;
                        y_offset = card_y + 25;
                        break;
                }

                Color text_color = PUFF_WHITE;
                DrawText(TextFormat("%d", env->cards_in_hand[i][j][k]), x_offset, y_offset, 20, text_color);
            }
            // add a little text on the top right that says Card 1, Card 2, Card 3, Card 4, Card 5
            DrawText(TextFormat("Card %d", j+1), card_x + env->card_width -50, card_y + 5, 10, PUFF_WHITE);
        }
        if (i == 0) {
            DrawText(TextFormat("%d", env->score[i]), env->card_width *0.4, env->height - 100, 100, PUFF_WHITE);
        } else {
            DrawText(TextFormat("%d", env->score[i]), env->width - env->card_width *.6, env->height - 100, 100, PUFF_WHITE);
        }
    }
    EndDrawing();

    //PlaySound(client->sound);
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
