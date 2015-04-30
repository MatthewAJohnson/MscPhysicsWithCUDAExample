#include "Tree.h"

//Mesh* Tree::cylinder = NULL;

Tree::Tree(Mesh* cylin, Mesh* leaf,ParticleEmitter* p,Shader* s, float x, float y, float z)
{
	srand(time(NULL));

	float length = 150;
	float scale = 0.75;
	//tree = Mesh::GenerateCircle(100);
	trunk = new SceneNode(cylin, Vector4(0,1,0,1));
	
	trunk->SetMesh(cylin);
	trunk->SetModelScale(Vector3(0.3,scale,0.3));
	trunk->SetTransform(Matrix4::Translation(Vector3((RAW_HEIGHT*HEIGHTMAP_X /2.0f), 125, (RAW_WIDTH*HEIGHTMAP_X / 2.0f))));
	trunk->SetTransform(trunk->GetTransform()  * Matrix4::Rotation(0.0f, Vector3(1,0,0)));// * Matrix4::Scale(Vector3(0.1, scale, 0.1)));
	//trunk->GetMesh()->SetTexture(SOIL_load_OGL_texture(TEXTUREDIR"water.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS));
	
	AddChild(trunk);
	
	AddNode(cylin, leaf,p,s, 0.2,scale,0.2, trunk,length,9);
	//AddNode(cylin, leaf,p,s,  0.2,scale,0.2, trunk, length,9);
		
	Update(0);
}

Tree::~Tree()
{
	delete trunk;
}

void Tree::AddNode(Mesh* cylin, Mesh* leaf, ParticleEmitter* p,Shader* s, float x, float y, float z, SceneNode* parent, float length, int count)
{
		float degrees = rand() % 90 - 45;
	
		BranchNode* branch = new BranchNode(cylin, count-11, x, y, z);
		branch->SetMesh(cylin);
		branch->SetModelScale(Vector3(x, y, z));
		//branch->SetTransform(Matrix4::Translation(Vector3(0,length,0)) * Matrix4::Scale(Vector3(0.75,1.0,0.75)));
		if((int)count%2 == 0)
		{
			if(degrees >=0)
			{
				branch->SetTransform(Matrix4::Translation(Vector3(0,length*y,0)) * Matrix4::Rotation(degrees, Vector3(1,0,0)) );//* Matrix4::Scale(Vector3(0.75,1.0,0.75)));
				//branch->SetTransform(Matrix4::Rotation(45, Vector3(1,0,0)) * branch->GetTransform()  ) ;
			}
			else
			{
				branch->SetTransform(Matrix4::Translation(Vector3(0,length*y,0)) * Matrix4::Rotation(degrees, Vector3(1,0,0)) );
			}
		}
		else
		{
			if(degrees >=0)
			{
				branch->SetTransform(Matrix4::Translation(Vector3(0,length*y,0)) * Matrix4::Rotation(degrees, Vector3(0,0,1)) );
			}
			else
			{
				branch->SetTransform(Matrix4::Translation(Vector3(0,length*y,0)) * Matrix4::Rotation(degrees, Vector3(0,0,1)) );
				//branch->SetTransform(Matrix4::Rotation(45, Vector3(0,0,1)) * branch->GetTransform() );
			}
			
		}

		parent->AddChild(branch);
		
		--count;
		//leaf node
		if(count < 0)
		{
			float leafFlowerParticleChance = rand() % 100 + 1;
			if(leafFlowerParticleChance > 15)
			{
				BranchNode* leafNode = new BranchNode(leaf,count-11,12,15,12 );
				leafNode->SetMesh(leaf);
				branch->AddChild(leafNode);
				leafNode->SetTransform(Matrix4::Translation(Vector3(0,length*y,0)) * Matrix4::Rotation(180, Vector3(1,0,0)) );
			}
			else if(leafFlowerParticleChance <= 15)
			{
				BranchNode* particleNode = new BranchNode(p, count-11,10,10,10);
			//	particleNode->SetFlag(true);
				branch->AddChild(particleNode);
				
				particleNode->SetTransform(Matrix4::Translation(Vector3(0,length*y,0)));
				particleNode->SetTransform(Matrix4::Translation(Vector3(0,length*y,0)) * Matrix4::Rotation(180, Vector3(1,0,0)) );
			}
		}
		//recursively add more branches
		if(count >=0)
		{
		AddNode(cylin, leaf, p, s, x*0.75,y*0.8,z*0.75, branch,length,count);
		}
		if(count >=0)
		{
		AddNode(cylin, leaf, p, s, x*0.75,y,z*0.75, branch,length,count);
		}
		
	
}

void Tree::Update(float msec)
{
	//BranchNode::Update(msec);
	SceneNode::Update(msec);

}

void Tree::DrawTree()
{

}