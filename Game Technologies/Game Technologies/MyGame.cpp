#include "MyGame.h"
//nothing is real, everything is possible
//#define false true
/*
Creates a really simple scene for our game - A cube robot standing on
a floor. As the module progresses you'll see how to get the robot moving
around in a physically accurate manner, and how to stop it falling
through the floor as gravity is added to the scene. 

You can completely change all of this if you want, it's your game!

*/
MyGame::MyGame()	{
	gameCamera = new Camera(-30.0f,0.0f,Vector3(0,450,850));
		mass = 0.001f;
		velo = -1000;
	Renderer::GetRenderer().SetCamera(gameCamera);

	CubeRobot::CreateCube();
	
	/*
	We're going to manage the meshes we need in our game in the game class!

	You can do this with textures, too if you want - but you might want to make
	some sort of 'get / load texture' part of the Renderer or OGLRenderer, so as
	to encapsulate the API-specific parts of texture loading behind a class so
	we don't care whether the renderer is OpenGL / Direct3D / using SOIL or 
	something else...
	*/
	cube	= new OBJMesh(MESHDIR"cube.obj");
	quad	= Mesh::GenerateQuad();
	sphere	= new OBJMesh(MESHDIR"ico.obj");

	/*
	A more 'robust' system would check the entities vector for duplicates so as
	to not cause problems...why not give it a go?
	*/
	//allEntities.push_back(BuildRobotEntity());

	allEntities.push_back(BuildQuadEntity(1000.0f));


	//allEntities.push_back(BuildTestSphereEntity(100.0f, Vector3(-400, 300, 0), Vector3(100.0f, 0, 0)));
	//allEntities.push_back(BuildTestSphereEntity(100.0f, Vector3(400, 300, 0), Vector3(-100.0f, 0, 0)));

	for(int i = 0; i < 10; ++i)
	{
		for (int x = 0; x < 10; ++x)
		{
			for(int z = 0; z < 10; ++z)
			{
				allEntities.push_back(BuildTestSphereEntity(10.0f, Vector3(400+(x*20), 200+(i*20), 300+(z*20)), Vector3(0, 0, 0)));
			}
		}
	}
	growingSphere = BuildTestSphereEntity(20.0f,Vector3(0,200,0),Vector3(0,0,0));
	growingSphere->GetRenderNode().SetModelScale(Vector3(20.0f,20.0f,20.0f));
	growingSphere->GetPhysicsNode().SetCollisionVolumeType( new CollisionSphere(20.0f));
	growingSphere->GetPhysicsNode().DisableLifeSpan();
	growingSphere->GetPhysicsNode().DisableGravity();
	growingSphere->GetPhysicsNode().SetInverseMass(0.0f);
	growingSphere->GetPhysicsNode().ImoveableObject();
	allEntities.push_back(growingSphere);
	//GameEntity* testSphere = BuildSphereEntity(15.0f);
	//testSphere->GetPhysicsNode().ImoveableObject();
	
	//allEntities.push_back(testSphere);

}

MyGame::~MyGame(void)	{
	/*
	We're done with our assets now, so we can delete them
	*/
	delete cube;
	delete quad;
	delete sphere;

	CubeRobot::DeleteCube();

	//GameClass destructor will destroy your entities for you...
}

/*
Here's the base 'skeleton' of your game update loop! You will presumably
want your games to have some sort of internal logic to them, and this
logic will be added to this function.
*/
void MyGame::UpdateGame(float msec)
{
	if(gameCamera) 
	{
		gameCamera->UpdateCamera(msec);
	
	
		if(Window::GetKeyboard()->KeyDown(KEYBOARD_3))
		{
			mass += 0.02;
		}
		if(Window::GetKeyboard()->KeyDown(KEYBOARD_1))
		{
			velo -= 500;
		}
		if(Window::GetKeyboard()->KeyDown(KEYBOARD_0))
		{
			mass -= 0.02f;
		}
		if(Window::GetKeyboard()->KeyDown(KEYBOARD_2))
		{
			velo += 100;
		}
		if(Window::GetMouse()->ButtonDown(MOUSE_LEFT) && !Window::GetMouse()->ButtonHeld(MOUSE_LEFT))
		{
			GameEntity* mouseCube = BuildSphereEntity(10.0f);
			mouseCube->GetPhysicsNode().SetPosition(gameCamera->GetPosition());
			
		//	Matrix4 temp = mouseCube->GetPhysicsNode().BuildTransform();	
		
			mouseCube->GetPhysicsNode().SetAngularVelocity(Vector3(0.0f,0.0f,0.0f));
		//	mouseCube->GetRenderNode().SetTransform(temp);

			mouseCube->GetPhysicsNode().SetInverseMass(mass);
			mouseCube->GetPhysicsNode().SetLinearVelocity(gameCamera->BuildViewMatrix().CameraFacing()*velo);
			//mouseCube->GetPhysicsNode().SetForce(gameCamera->BuildViewMatrix().CameraFacing()*-300000);
			
		
			allEntities.push_back(mouseCube);
		}
	}
	
	float value = 1.001f;
	if(Window::GetKeyboard()->KeyDown(KEYBOARD_UP))
	{
		value += 0.05f;
	}
	if(Window::GetKeyboard()->KeyDown(KEYBOARD_DOWN))
	{
		value -= 0.05f;
	}

	//sorry, not sorry
	float sphere = growingSphere->GetPhysicsNode().GetCollisionVolume()->GetSize();// = growingSphere->GetPhysicsNode().GetCollisionVolume()->GetType();
	sphere = sphere * value;
	float temp = sphere;
	Vector3 scale = growingSphere->GetRenderNode().GetModelScale() * value;
	growingSphere->GetRenderNode().SetModelScale(scale);
	growingSphere->GetPhysicsNode().SetCollisionVolumeType(new CollisionSphere(temp));

	

	for(vector<GameEntity*>::iterator i = allEntities.begin(); i != allEntities.end(); i)
	{
		(*i)->Update(msec);
		if((*i)->GetPhysicsNode().HasLifeSpan() && (*i)->GetPhysicsNode().GetCurrentLifeSpan() <= 0/* && !(*i)->GetPhysicsNode().GetRestedState() */)
		{
			Vector3 tempPos = (*i)->GetPhysicsNode().GetPosition();

			(*i)->DisconnectFromSystems();
			
			i = allEntities.erase(i);
			
			GameEntity* x = BuildSphereEntity(100.0f);
			x->GetPhysicsNode().SetPosition(tempPos);
			x->GetPhysicsNode().SetForce(Vector3(0,10,0));
			x->GetRenderNode().SetColour(Vector4(1,0.5,1.5,0.5));
			x->GetPhysicsNode().DisableLifeSpan();
			//x->ConnectToSystems();
			//tempList.push_back(x);
			allEntities.push_back(x);
			
		}
		else
		{
			++i;
		}


	
	}
//	for(vector<GameEntity*>::iterator y = tempList.begin(); y !=tempList.end(); ++y)
//	{
		//GameEntity* x = (*y);
		//allEntities.push_back((*y));
		//tempList.erase(y);

//	}
	//for(vector<GameEntity*>::iterator y = tempList.begin(); y !=tempList.end(); ++y)
	//{
	//	(*y)->DisconnectFromSystems();
	//	tempList.erase(y);
	//}
	/*
	Here's how we can use OGLRenderer's inbuilt debug-drawing functions! 
	I meant to talk about these in the graphics module - Oops!

	We can draw squares, lines, crosses and circles, of varying size and
	colour - in either perspective or orthographic mode.

	Orthographic debug drawing uses a 'virtual canvas' of 720 * 480 - 
	that is 0,0 is the top left, and 720,480 is the bottom right. A function
	inside OGLRenderer is provided to convert clip space coordinates into
	this canvas space coordinates. How you determine clip space is up to you -
	maybe your renderer has getters for the view and projection matrix?

	Or maybe your Camera class could be extended to contain a projection matrix?
	Then your game would be able to determine clip space coordinates for its
	active Camera without having to involve the Renderer at all?

	Perspective debug drawing relies on the view and projection matrices inside
	the renderer being correct at the point where 'SwapBuffers' is called. As
	long as these are valid, your perspective drawing will appear in the world.

	This gets a bit more tricky with advanced rendering techniques like deferred
	rendering, as there's no guarantee of the state of the depth buffer, or that
	the perspective matrix isn't orthographic. Therefore, you might want to draw
	your debug lines before the inbuilt position before SwapBuffers - there are
	two OGLRenderer functions DrawDebugPerspective and DrawDebugOrtho that can
	be called at the appropriate place in the pipeline. Both take in a viewProj
	matrix as an optional parameter.

	Debug rendering uses its own debug shader, and so should be unaffected by
	and shader changes made 'outside' of debug drawing

	*/
	//Lets draw a box around the cube robot!
	//Renderer::GetRenderer().DrawDebugBox(DEBUGDRAW_PERSPECTIVE, Vector3(0,51,0), Vector3(100,100,100), Vector3(1,0,0));

	////We'll assume he's aiming at something...so let's draw a line from the cube robot to the target
	////The 1 on the y axis is simply to prevent z-fighting!
	//Renderer::GetRenderer().DrawDebugLine(DEBUGDRAW_PERSPECTIVE, Vector3(0,1,0),Vector3(200,1,200), Vector3(0,0,1), Vector3(1,0,0));

	////Maybe he's looking for treasure? X marks the spot!
	//Renderer::GetRenderer().DrawDebugCross(DEBUGDRAW_PERSPECTIVE, Vector3(200,1,200),Vector3(50,50,50), Vector3(0,0,0));

	////CubeRobot is looking at his treasure map upside down!, the treasure's really here...
	//Renderer::GetRenderer().DrawDebugCircle(DEBUGDRAW_PERSPECTIVE, Vector3(-200,1,-200),50.0f, Vector3(0,1,0));
}

/*
Makes an entity that looks like a CubeRobot! You'll probably want to modify
this so that you can have different sized robots, with different masses and
so on!
*/
GameEntity* MyGame::BuildRobotEntity() {
	GameEntity*g = new GameEntity(new CubeRobot(), new PhysicsNode());
	g->ConnectToSystems();
	g->GetPhysicsNode().SetInverseMass(1.0f/20.0f);
	g->GetPhysicsNode().SetForce(Vector3(0,GRAVITY,0));
	g->GetPhysicsNode().SetCollisionVolumeType( new CollisionSphere(100.0f));
	g->GetPhysicsNode().DisableGravity();
	return g;
}

/*
Makes a cube. Every game has a crate in it somewhere!
*/
GameEntity* MyGame::BuildCubeEntity(float size) {
	GameEntity*g = new GameEntity(new SceneNode(cube), new PhysicsNode());
	g->ConnectToSystems();
	SceneNode &test = g->GetRenderNode();
	g->GetPhysicsNode().SetCollisionVolumeType( new CollisionSphere(size));

	test.SetModelScale(Vector3(size,size,size));
	test.SetBoundingRadius(size);

	return g;
}
/*
Makes a sphere.
*/
GameEntity* MyGame::BuildSphereEntity(float radius) 
{
	SceneNode* s = new SceneNode(sphere);

	s->SetModelScale(Vector3(radius,radius,radius));
	s->SetBoundingRadius(radius);

	s->SetColour(Vector4((rand() % 100) / 100.f, (rand() % 100) / 100.f, (rand() % 100) / 100.f, 1.f) );

	GameEntity*g = new GameEntity(s, new PhysicsNode());
	g->ConnectToSystems();
	g->GetPhysicsNode().SetInverseMass(1.0f);
	g->GetPhysicsNode().SolidSphereInertia(radius);
	g->GetPhysicsNode().SetCollisionVolumeType( new CollisionSphere(radius));
	return g;
}

/*
Makes a flat quad, initially oriented such that we can use it as a simple
floor. 
*/
GameEntity* MyGame::BuildQuadEntity(float size) //vec3 angle, 
{
	SceneNode* s = new SceneNode(quad);

	s->SetModelScale(Vector3(size,size,size));
	//Oh if only we had a set texture function...we could make our brick floor again WINK WINK
	s->SetBoundingRadius(size);

	PhysicsNode*p = new PhysicsNode(Quaternion::AxisAngleToQuaterion(Vector3(1,0,0), 90.0f), Vector3());

	GameEntity*g = new GameEntity(s, p);
	g->GetPhysicsNode().SetCollisionVolumeType(new CollisionPlane(Plane(Vector3(0,1,0), 0.f))/*Vector3::Dot(Vector3(0,1,0), Vector3(0,0,0) )))*/);
	p->SetInverseMass(0.0f);
	p->SetPosition(Vector3(0,-100,0));
	p->ImoveableObject();
	p->DisableGravity();
	p->DisableLifeSpan();
	//g->GetRenderNode().SetBoundingRadius(5);
	g->ConnectToSystems();
	return g;
}

GameEntity* MyGame::BuildTestSphereEntity(float radius, Vector3 pos, Vector3 velo)
{
	SceneNode* s = new SceneNode(sphere);

	s->SetModelScale(Vector3(radius,radius,radius));
	s->SetBoundingRadius(radius);
	s->SetColour(Vector4(0,0,1,1));
	PhysicsNode*p = new PhysicsNode();
	p->SetPosition(pos);
	p->SetLinearVelocity(velo);
	p->SetAngularVelocity(Vector3(0.0f,0.0f,0.0f));
	//p->SetLinearVelocity(Vector3(0.0f,0.1f,0.0f));
	p->SetForce(Vector3(0.0f,0.5f,0.0f));
	p->SetInverseMass(0.1f);
	p->SolidSphereInertia(radius);
	p->SetCollisionVolumeType(new CollisionSphere(radius));
	GameEntity*g = new GameEntity(s, p);
	g->ConnectToSystems();
	return g;
}