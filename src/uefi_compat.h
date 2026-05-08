// uefi_compat.h - Minimal UEFI compatibility stubs
#ifndef UEFI_COMPAT_H
#define UEFI_COMPAT_H

#include <file.h>
#include <lib.h>

typedef struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} Color;

// Memory functions - arena allocator for UEFI
static char _arena_buf[8*1024*1024];  // 8MB arena
static size_t _arena_offset = 0;
static unsigned int _rand_state = 12345;

void* malloc(size_t size);

void* calloc(size_t nmemb, size_t size) {
    return malloc(nmemb * size);
}

void* malloc(size_t size) {
    size = (size + 7) & ~7;
    if (_arena_offset + size > sizeof(_arena_buf)) return NULL;
    void* ptr = _arena_buf + _arena_offset;
    _arena_offset += size;
    return ptr;
}

void free(void* ptr) {}

// Simple LCG random - returns 0 to RAND_MAX
int rand(void) {
    _rand_state = (_rand_state * 1103515245 + 12345) & 0x7fffffff;
    return _rand_state;
}

void srand(unsigned int seed) {
    _rand_state = seed;
}

// Math stubs - Taylor series approximations for UEFI (no libm)
#define M_PI 3.14159265358979323846

float _sinf(float x) {
    // Normalize to [-PI, PI]
    while (x > M_PI) x -= 2*M_PI;
    while (x < -M_PI) x += 2*M_PI;
    // Taylor series: x - x^3/6 + x^5/120 - ...
    float x2 = x*x;
    return x * (1.0f - x2/6.0f * (1.0f - x2/20.0f * (1.0f - x2/42.0f)));
}

float _cosf(float x) {
    while (x > M_PI) x -= 2*M_PI;
    while (x < -M_PI) x += 2*M_PI;
    float x2 = x*x;
    return 1.0f - x2/2.0f * (1.0f - x2/12.0f * (1.0f - x2/30.0f));
}

float _expf(float x) {
    // Taylor: 1 + x + x^2/2 + x^3/6 + ...
    float sum = 1.0f;
    float term = 1.0f;
    for (int i = 1; i < 10; i++) {
        term *= x / i;
        sum += term;
    }
    return sum;
}

float _logf(float x) {
    // Taylor around x=1: (x-1) - (x-1)^2/2 + (x-1)^3/3 - ...
    if (x <= 0) return -100;
    float y = x - 1.0f;
    float sum = 0.0f;
    float term = y;
    for (int i = 1; i < 10; i++) {
        sum += term / i;
        term *= -y;
    }
    return sum;
}

float _sqrtf(float x) {
    // Newton's method
    float guess = x / 2.0f;
    for (int i = 0; i < 10; i++) {
        guess = (guess + x / guess) / 2.0f;
    }
    return guess;
}

float _tanhf(float x) {
    // tanh(x) = (e^x - e^-x)/(e^x + e^-x)
    float ex = _expf(x);
    float emx = _expf(-x);
    return (ex - emx) / (ex + emx);
}

#undef sinf
#undef cosf
#undef expf
#undef logf
#undef sqrtf
#undef tanhf
#define sinf(x) _sinf(x)
#define cosf(x) _cosf(x)
#define expf(x) _expf(x)
#define logf(x) _logf(x)
#define sqrtf(x) _sqrtf(x)
#define tanhf(x) _tanhf(x)
#define tanh(x) _tanhf(x)
#define log(x) _logf(x)
#define sqrt(x) _sqrtf(x)
#define fminf(x, y) ((x) < (y) ? (x) : (y))
#define fmaxf(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))
#define abs(x) ((x) < 0 ? -(x) : (x))
#define isfinite(x) (1)

typedef struct Vector2 { float x; float y; } Vector2;

inline int rand_r(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

// Random
int rand(void);
void srand(unsigned int seed);
#define RAND_MAX 2147483647

// Assert stub
#define assert(expr) ((void)0)

// File I/O stubs
typedef struct { int _; } FILE;
#define stdin ((FILE*)0)
#define stdout ((FILE*)1)
#define stderr ((FILE*)2)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
FILE* fopen(const char* path, const char* mode);
int fclose(FILE* stream);
long ftell(FILE* stream);
size_t fread(void* ptr, size_t size, size_t nmemb, FILE* stream);
int fseek(FILE* stream, long offset, int whence);
void rewind(FILE* stream);
void perror(const char* s);
void free(void* ptr);

// Print to framebuffer - uses global fb, xres, yres from breakout.c
extern uint32_t* fb;
extern uint32_t xres;
extern uint32_t yres;

// Font globals - set by bootloader
extern uint32_t _font_width;
extern uint32_t _font_height;

void line_feed(Bitmap_Font *font) {
    if ((y + font->height) < (yres - font->height)) y += font->height;
    else {
        uint32_t char_line_px    = xres * font->height;
        uint32_t char_line_bytes = char_line_px * 4;
        uint32_t char_lines      = yres / font->height;

        memcpy(fb, fb + char_line_px, char_line_bytes * (char_lines-1));

        uint32_t px = ((yres / font->height) - 1) * font->height * xres;
        for (uint32_t i = 0; i < char_line_px; i++)
            fb[px++] = text_bg_color;
    }
}

void print_string(char *string, Bitmap_Font *font) {
    uint32_t glyph_size = ((font->width + 7) / 8) * font->height;
    uint32_t glyph_width_bytes = (font->width + 7) / 8;
    for (char c = *string++; c != '\0'; c = *string++) {
        if (c == '\r') { x = 0; continue; }
        if (c == '\n') { line_feed(font); continue; }

        uint8_t *glyph = &font->glyphs[c * glyph_size];

        for (uint32_t i = 0; i < font->height; i++) {
            uint64_t mask = 1 << (font->width-1);
            uint64_t bytes = font->left_col_first ?
                             ((uint64_t)glyph[0] << 56) |
                             ((uint64_t)glyph[1] << 48) |
                             ((uint64_t)glyph[2] << 40) |
                             ((uint64_t)glyph[3] << 32) |
                             ((uint64_t)glyph[4] << 24) |
                             ((uint64_t)glyph[5] << 16) |
                             ((uint64_t)glyph[6] <<  8) |
                             ((uint64_t)glyph[7] <<  0)
                             : *(uint64_t *)glyph;
            for (uint32_t px = 0; px < font->width; px++) {
                //uint32_t x_1 = x/2;
                //uint32_t y_1 = y/2;
                fb[(y*2)*xres + (x*2)] = bytes & mask ? text_fg_color : text_bg_color;
                fb[(y*2)*xres + (x*2)+1] = bytes & mask ? text_fg_color : text_bg_color;
                fb[((y*2)+1)*xres + (x*2)] = bytes & mask ? text_fg_color : text_bg_color;
                fb[((y*2)+1)*xres + (x*2)+1] = bytes & mask ? text_fg_color : text_bg_color;
                mask >>= 1;
                x++;
            }
            y++;
            x -= font->width;
            glyph += glyph_width_bytes;
        }

        y -= font->height;
        if (x + font->width < (xres/2) - font->width) x += font->width;
        else {
            x = 0;
            line_feed(font);
        }
    }
}

void DrawPixel(int x, int y, Color color) {
    if (x >= 0 && (unsigned int)x < xres && y >= 0 && (unsigned int)y < yres)
        fb[y*xres + x] = (color.a << 24) | (color.r << 16) | (color.g << 8) | color.b;
}

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

void DrawLine(int x1, int y1, int x2, int y2, Color color) {
    int dx = x2 - x1, dy = y2 - y1;
    int steps = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);
    if (steps == 0) { DrawPixel(x1, y1, color); return; }
    for (int i = 0; i <= steps; i++) {
        int x = x1 + dx * i / steps;
        int y = y1 + dy * i / steps;
        DrawPixel(x, y, color);
    }
}

void DrawLineEx(Vector2 v1, Vector2 v2, int thickness, Color color) {
    int x1 = v1.x;
    int y1 = v1.y;
    int x2 = v2.x;
    int y2 = v2.y;
    int dx = x2 - x1, dy = y2 - y1;
    int steps = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);
    if (steps == 0) {
        for (int ty = -thickness/2; ty <= thickness/2; ty++)
            for (int tx = -thickness/2; tx <= thickness/2; tx++)
                DrawPixel(x1+tx, y1+ty, color);
        return;
    }
    for (int i = 0; i <= steps; i++) {
        int x = x1 + dx * i / steps;
        int y = y1 + dy * i / steps;
        for (int ty = -thickness/2; ty <= thickness/2; ty++)
            for (int tx = -thickness/2; tx <= thickness/2; tx++)
                DrawPixel(x+tx, y+ty, color);
    }
}

#endif // UEFI_COMPAT_H
