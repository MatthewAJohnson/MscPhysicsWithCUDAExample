#pragma once
#pragma comment(lib, "cudart.lib")
#include <iostream>


#include <minmax.h>
#include "../../nclgl/OGLRenderer.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <SOIL.h>

#include "../../nclgl/Mesh.h"
#include "../../nclgl/Shader.h"
#include "../../nclgl/GameTimer.h"

class VBOWaterResource : public Mesh
{
public:
	VBOWaterResource();
	~VBOWaterResource();
	void initVBO(MeshBuffer type, float* data, int comps, int num, unsigned int mode);
	void initIBO(unsigned int* data, int num, unsigned int mode);
	void draw() const;

	void update(float msec);
private:
	void generateGrid();
	float time; 
	unsigned int restart_index;
	int width, height;
	struct cudaGraphicsResource* cudaVBO;
};