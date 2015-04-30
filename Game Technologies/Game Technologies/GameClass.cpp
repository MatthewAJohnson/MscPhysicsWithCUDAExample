#include "GameClass.h"

GameClass* GameClass::instance = NULL;

GameClass::GameClass()
{
	physicsPerSecond = 0.0f;
	renderCounter	 = 0.0f;
	physicsCounter	 = 0.0f;
	fpsTimer	     = 0.0f;
	instance		 = this;
}

GameClass::~GameClass(void)	
{
	for(vector<GameEntity*>::iterator i = allEntities.begin(); i != allEntities.end(); ++i) 
	{
		delete (*i);
	}
	delete gameCamera;
}

void GameClass::UpdateCore(float msec) 
{
	renderCounter	-= msec;
	//physicsCounter	+= msec;

	if(renderCounter <= 0.0f)
	{	//Update our rendering logic
		Renderer::GetRenderer().UpdateScene(1000.0f / (float)RENDER_HZ);
		Renderer::GetRenderer().RenderScene();
		renderCounter += (1000.0f / (float)RENDER_HZ);
	}

	fpsTimer += msec;
	frameCounter++;
	if(fpsTimer >= 1000)
	{
		fps = (float)frameCounter / (fpsTimer * 0.001f);
		frameCounter = 0;
		fpsTimer = 0;
		Renderer::GetRenderer().physicsFrameRate = physicsPerSecond;
		Renderer::GetRenderer().framesPerSecond = fps;
		physicsPerSecond = 0;
	}
	Renderer::GetRenderer().totalCollisions = PhysicsSystem::GetPhysicsSystem().collisions;
}


void GameClass::PhysicsHandler(volatile bool* gameComplete)
{
	GameTimer physicsTimer;

	while (!*gameComplete)
	{
		
		physicsCounter	+=  physicsTimer.GetTimedMS();	
		
		//physicsPerSecond = delta;
		while(physicsCounter >= 0.0f)
		{
			
			physicsCounter -= PHYSICS_TIMESTEP;
			PhysicsSystem::GetPhysicsSystem().Update(PHYSICS_TIMESTEP);
			++physicsPerSecond;
		}
	}
}