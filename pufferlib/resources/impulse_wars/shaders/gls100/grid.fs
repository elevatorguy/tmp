#version 100

precision highp float;

// Input vertex attributes (from vertex shader)
varying vec3 fragPosition;
varying vec4 fragColor;

// Input uniform values
uniform vec2 pos[4];
uniform vec4 color[4];

const float falloff = 6.0;
const float epsilon = 0.1;

void main()
{
    vec3 lightAccum = vec3(0.0);

    // Texel color fetching from texture sampler
    for (int i = 0; i < 4; i++) {
        vec2 playerPos = pos[i];
        vec4 playerColor = color[i];
        float dist = distance(playerPos, fragPosition.xz);
        if (dist == 0.0) {
            continue;
        }

        float intensity = falloff / (dist * dist);
        lightAccum.r += intensity * (playerColor.r / 255.0);
        lightAccum.g += intensity * (playerColor.g / 255.0);
        lightAccum.b += intensity * (playerColor.b / 255.0);
    }

    if (length(lightAccum) < epsilon) {
        discard;
    }

    gl_FragColor = vec4(lightAccum, 1.0);
}
