#pragma once
#include "../../nclgl/SceneNode.h"
#include "../../nclgl/OBJMesh.h"
#include  "../../nclgl/Mesh.h"
#include "../../nclgl/HeightMap.h"
#include "../../nclgl/ParticleEmitter.h"
#include "BranchNode.h"
#include <stdlib.h>
#include <time.h>
class Tree : public SceneNode
{
public:
	Tree(Mesh* cylin, Mesh* leaf,ParticleEmitter* p,Shader* s, float x, float y, float z);
	virtual ~Tree();
	virtual void Update(float msec);
	void AddNode(Mesh* cylin,Mesh* leaf, ParticleEmitter* p,Shader* s, float x, float y, float z, SceneNode* parent, float length, int count);
	void DrawTree();
		
protected:
	
	SceneNode* trunk;
	float radius;
	float height;
};