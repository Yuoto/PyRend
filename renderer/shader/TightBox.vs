#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 intrinsic;
uniform mat4 extrinsic;
out vec3 Pos;

void main()
{
	gl_Position = intrinsic * extrinsic * model * vec4(aPos, 1.0);
	Pos = aPos;
}


