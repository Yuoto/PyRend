#version 330 core


in vec2 TexCoords;
in vec3 Normal;
in vec4 Color;
in vec3 FragPos;

out vec4 FragColor;

struct Material{
  sampler2D map_Kd;  //diffuse map
  sampler2D map_Ka;  //ambient map
  sampler2D map_Ks;  //specular map
  
  vec3 Kd;  //diffuse coefficient
  vec3 Ka;  //ambient coefficient
  vec3 Ks;  //specular coefficient
  float Ns;  //shininess
};

struct Light{
  vec3 position;

  vec3 color;
  float strength;

  float constant;
  float linear;
  float quadratic;

  bool enableDirectional;
  bool enableAttenuation;
};

uniform vec3 viewPos;
uniform Material material;
uniform Light light;
uniform bool hasNormal, hasColor, hasTexture;


void main()
{    
	float alhpa = clamp(Color.w,0.0,1.0);
	vec3 color = clamp(Color.xyz,0.0,1.0);
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	//Considering Texture
	
	if (hasTexture){
		ambient = light.strength*light.color*texture(material.map_Ka,TexCoords).rgb*alhpa;
		diffuse = light.strength*light.color*texture(material.map_Kd,TexCoords).rgb*alhpa;
		specular = light.strength*light.color*texture(material.map_Ks,TexCoords).rgb*alhpa;

	} else if(hasColor){
		ambient = light.strength*light.color*material.Ka*color*alhpa;
		diffuse = light.strength*light.color*material.Kd*color*alhpa;
		specular = light.strength*light.color*material.Ks*color*alhpa;
	}else{
			ambient = light.strength*light.color*material.Ka*alhpa;
		diffuse = light.strength*light.color*material.Kd*alhpa;
		specular = light.strength*light.color*material.Ks*alhpa;
	}

	//considering direction(update diffuse and specular coefficient)
	if (light.enableDirectional && hasNormal){
		//diffuse
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(light.position-FragPos);
		float diff = max(dot(norm,lightDir),0.0);
		diffuse += diff*diffuse;
		
		//specular
		vec3 viewDir = normalize(viewPos-FragPos);
		vec3 reflectDir = reflect(-lightDir,norm);
		float spec = pow(max(dot(viewDir,reflectDir),0.0),material.Ns);
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
