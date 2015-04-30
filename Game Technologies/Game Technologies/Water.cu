//#pragma once
//#pragma comment(lib, "cudart.lib")
//#include <iostream>
//
//
//#include <minmax.h>
//#include "../../nclgl/OGLRenderer.h"
//#include <cuda_gl_interop.h>
//#include <cuda_runtime.h>
//
//#include <SOIL.h>
//
//#include "../../nclgl/Mesh.h"
//#include "../../nclgl/Shader.h"
//#include "../../nclgl/GameTimer.h"
//
//using namespace std;
//
//GLuint waterTexture;
//
//class VBOWaterResource : public Mesh
//{
//public:
//	VBOWaterResource();
//	~VBOWaterResource();
//	void initVBO(MeshBuffer type, float* data, int comps, int num, unsigned int mode);
//	void initIBO(unsigned int* data, int num, unsigned int mode);
//	void draw() const;
//
//	void update();
//private:
//	void generateGrid();
//
//	unsigned int restart_index;
//	int width, height;
//	struct cudaGraphicsResource* cudaVBO;
//};

#include "Water.h"
VBOWaterResource::VBOWaterResource(): width(100), height(100) {

	width = max(2, width);
	height = max(2, height);
	time = 0;
	generateGrid();

	//cudaGLSetGLDevice(0);
	cudaError t = cudaGraphicsGLRegisterBuffer(&cudaVBO, bufferObject[VERTEX_BUFFER]/*vbo[VBO_VERTEX]*/, cudaGraphicsMapFlagsNone); 
	if (/*cudaGraphicsGLRegisterBuffer(&cudaVBO, bufferObject[VERTEX_BUFFER]/*vbo[VBO_VERTEX], cudaGraphicsMapFlagsNone) */t != cudaSuccess)
	{
		 
		printf("Failed with error: %s\n\n\n\n", cudaGetErrorString(t));
	}



}

VBOWaterResource::~VBOWaterResource() {
	if (cudaGraphicsUnregisterResource(cudaVBO) != cudaSuccess)
	{
			printf("Failed\n");
	}
}

void VBOWaterResource::generateGrid() {

	int loop_size = 2*height + 1;

	numVertices = width*height;
	numIndices = (width - 1)*loop_size;

	vertices = new Vector3[numVertices];
	normals = new Vector3[numVertices];
	textureCoords = new Vector2[numVertices];
	indices = new unsigned int [numIndices];

	type = GL_TRIANGLE_STRIP;
	for (int x = 0; x < width; x++) {
		int loops = x*loop_size;
		for (int y = 0; y < height; y++) {
			int offset = y*width + x;

			if (x != width - 1)
				indices[loops + 2*y + 1] = offset;
			if (x != 0)
				indices[loops - loop_size + 2*y] = offset;

			//vertices[3*offset + 0].x = 2*(x*1.0f/(width-1)) - 1;
			//vertices[3*offset + 1] = 0;
			//vertices[3*offset + 2] = 2*(y*1.0f/(height-1)) - 1;

			vertices[offset] = Vector3(2*(x*1.0f/(width-1)) - 1, 0, 2*(y*1.0f/(height-1)) - 1);

			//normals[3*offset + 0] = 0;
			//normals[3*offset + 1] = 1;
			//normals[3*offset + 2] = 0;

			normals[offset] = Vector3(0,1,0);

		//	textureCoords[2*offset + 0] = x*1.0f/(width-1);
			//textureCoords[2*offset + 1] = y*1.0f/(height-1);

			textureCoords[offset] = Vector2(x*1.0f/(width-1),y*1.0f/(height-1) );

		}
		if (x != width - 1)
			indices[loops + loop_size - 1] = width*height;
	}

	restart_index = width*height;

	//	glBindVertexArray(vao[0]);
	///*initVBO(VBO_VERTEX*/bufferObject((), (float*)verts, 3, num_verts, GL_DYNAMIC_DRAW);
	///*initVBO*/bufferObject(VBO_NORMAL, (float*)norms, 3, num_verts, GL_DYNAMIC_DRAW);
	//initVBO(VBO_TEXCOORD, (float*)texcoords, 2, num_verts, GL_DYNAMIC_DRAW);
	//initIBO(indices, num_indices, GL_DYNAMIC_DRAW);
	BufferData();
	glBindVertexArray(0);

//	delete[] vertices;
//	delete[] normals;
//	delete[] textureCoords;
//	delete[] indices;
}

__global__ void vboTestResource_update(float* ptr, int width, int height, float time) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = y*width + x;
	if (x >= width || y >= height) return;

	float period = 10; // smaller number = fewer waves
	float rate = 1.0;  //smaller number = slower waves

	float cx = x*0.5f/width - 0.5f;//affects origin of waves ... probably
	float cy = y*0.5f/height - 0.5f;//affects origin of waves ... probably

	float wave = sin(sqrt(cx*cx + cy*cy)*period - rate*time);

	int sign = wave>0?1:-1;
	wave = sign*sqrt(sign*wave);

	ptr[3*offset + 1] = wave/20; //smaller number, more wavey waves

	period *= 3;
	rate *= -9;

	ptr[3*offset + 1] += (sin(x*period/(width - 1) + rate*time) + sin(y*period/(height - 1) + rate*time))/60;//bigger number,  more wavey waves
}

void VBOWaterResource::update(float msec) {
	 time += msec * 0.0009;//GameTimer().GetTimedMS();

	float* devBuff;
	size_t size;

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((width - 1)/threadsPerBlock.x + 1, (height - 1)/threadsPerBlock.y + 1);

	if (cudaGraphicsMapResources(1, &cudaVBO, 0) != cudaSuccess)
	{
		printf("Failed\n");
	}

	cudaGraphicsResourceGetMappedPointer((void**)&devBuff, &size, cudaVBO);

	vboTestResource_update<<<numBlocks, threadsPerBlock>>>(devBuff, width, height, time);

	if (cudaGraphicsUnmapResources(1, &cudaVBO, 0) != cudaSuccess)
	{
		printf("Failed\n");
	}
}

void VBOWaterResource::initVBO(MeshBuffer type, float* data, int comps, int num, unsigned int mode) {
	glBindVertexArray(arrayObject);
	glGenBuffers(1, &bufferObject[type]);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObject[type]);
	glBufferData(GL_ARRAY_BUFFER, num*comps*sizeof(GLfloat), (GLvoid*)data, mode);
	glVertexAttribPointer((GLuint)type, comps, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray((GLuint)type);
	glBindVertexArray(0);
}

void VBOWaterResource::initIBO(unsigned int* data, int num, unsigned int mode) {
	glBindVertexArray(arrayObject);
	numIndices = num;
	glGenBuffers(1, &bufferObject[INDEX_BUFFER]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject[INDEX_BUFFER]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, num * sizeof(GLuint), (GLvoid*)data, mode);
	glBindVertexArray(0);
}

void VBOWaterResource::draw() const {
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glPrimitiveRestartIndex(restart_index);
	glEnable(GL_PRIMITIVE_RESTART);
	{
	glBindVertexArray(arrayObject);
	glDrawElements(type, numIndices, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
	}
	glDisable(GL_PRIMITIVE_RESTART);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}