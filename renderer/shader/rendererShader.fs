#version 330 core


in vec2 TexCoords;
in vec3 Normal;
in vec4 Color;
in vec3 FragPos;

out vec4 FragColor;

struct Material{
  sampler2D diffuse;
  sampler2D ddn;  
  sampler2D specular;
  float shininess;
};

struct Light{
  vec3 position;

  vec3 ambient;
  vec3 diffuse;
  vec3 specular;

  float constant;
  float linear;
  float quadratic;

  bool enableDirectional;
  bool enableAttenuation;
};

uniform vec3 viewPos;
uniform Material material;
uniform Light light;
uniform bool hasTexture;

void main()
{    
	float alhpa = clamp(Color.w,0.0,1.0);
	vec3 color = clamp(Color.xyz,0.0,1.0);
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	//Considering Texture
	if (hasTexture){
		ambient = light.ambient* texture(material.diffuse,TexCoords).rgb*color*alhpa;
		diffuse = light.diffuse*texture(material.diffuse,TexCoords).rgb*color*alhpa;
		specular = light.specular*texture(material.specular,TexCoords).rgb*color*alhpa;
	} else {
		ambient = light.ambient*color*alhpa;
		diffuse = light.diffuse*color*alhpa;
		specular = light.specular*color*alhpa;
	}

	//considering direction(update diffuse and specular coefficient)
	if (light.enableDirectional){
		//diffuse
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(light.position-FragPos);
		float diff = max(dot(norm,lightDir),0.0);
		diffuse += diff*diffuse;
		
		//specular
		vec3 viewDir = normalize(viewPos-FragPos);
		vec3 reflectDir = reflect(-lightDir,norm);
		float spec = pow(max(dot(viewDir,reflectDir),0.0),material.shininess);
		specular += spec*specular;
	}
	
	//considering atennuation
	if (light.enableAttenuation){
		//attenuation
		float distance = length(light.position - FragPos);
		float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

		ambient *= attenuation;
		diffuse *= attenuation;
		specular *= attenuation;
	}

	FragColor= vec4(ambient+diffuse+specular,1.0);

}
