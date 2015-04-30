#pragma once
#include "../../nclgl/SceneNode.h"

class BranchNode : public SceneNode
{
public:
	BranchNode(Mesh* m, int depth, float x, float y, float z);
	
	
	virtual void Update(float msec); 
	virtual void Draw(const OGLRenderer &r);
	
	float currentScale;
	
	
	float thisX;
	float thisZ;
	float thisY;
	
};