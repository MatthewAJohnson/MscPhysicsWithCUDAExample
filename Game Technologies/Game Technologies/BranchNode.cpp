#include "BranchNode.h"

BranchNode::BranchNode(Mesh* m,  int depth, float x, float y,float z) : SceneNode(m)
{
	
	SetModelScale(Vector3(0,0,0));
	thisX = x;
	thisZ = z;
	thisY = y;
	currentScale = (depth*0.5)+(((rand() % 200) / 200)-0.5);
}


void BranchNode::Update(float msec)
{
	float actualScale;
	if(currentScale < thisY)
	{
		currentScale += msec *0.001;
	}
	else
	{
		currentScale = thisY;
	}
	actualScale = max(currentScale, 0);
	SetModelScale(Vector3(thisX,actualScale,thisZ));
	SceneNode::Update(msec);

}

void BranchNode::Draw(const OGLRenderer &r)
{
	if(currentScale > 0)
	{
		SceneNode::Draw(r);
	}
}

