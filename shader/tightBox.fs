#version 330 core
out vec4 FragColor;
in vec3 Pos;
uniform vec3 color;

void main()
{

    vec3 colored = normalize(Pos);
    FragColor = vec4(colored,1.0);
}
