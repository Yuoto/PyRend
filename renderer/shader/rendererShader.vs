

#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aColor;
layout (location = 3) in vec2 aTexCoords;

out vec3 Normal;
out vec4 Color;
out vec3 FragPos;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 intrinsic;
uniform mat4 extrinsic;


void main(){

	gl_Position = intrinsic*extrinsic*model*vec4(aPos,1.0);
	Normal = mat3(transpose(inverse(extrinsic*model)))*aNormal;
	Color = aColor;
	FragPos = vec3(extrinsic*model*vec4(aPos,1.0));
	TexCoords = aTexCoords;
}

