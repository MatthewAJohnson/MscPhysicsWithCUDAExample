#include "PhysicsSystem.h"
#include <assert.h>
PhysicsSystem* PhysicsSystem::instance = 0;

PhysicsSystem::PhysicsSystem(void)	
{
	collisions = 0;
}

PhysicsSystem::~PhysicsSystem(void)	
{

}

void	PhysicsSystem::Update(float msec) 
{
	listLock.lock();
	BroadPhaseCollisions();
	NarrowPhaseCollisions();

	for(vector<PhysicsNode*>::iterator i = allNodes.begin(); i != allNodes.end(); ++i) 
	{
		(*i)->Update(msec);
	}
	listLock.unlock();
}

void	PhysicsSystem::BroadPhaseCollisions()
{
	
}

void	PhysicsSystem::NarrowPhaseCollisions() 
{
	CollisionData cData;
	
	for(int i = 0; i < allNodes.size() -1; ++i)
	{
		for(int j = i + 1; j < allNodes.size(); ++j)
		{
			bool collision = false;
			CollisionVolumeType typeA = allNodes[i]->GetCollisionVolume()->GetType();
			CollisionVolumeType typeB = allNodes[j]->GetCollisionVolume()->GetType();

			if (typeA == COLLISION_SPHERE && typeB == COLLISION_SPHERE)
			{
				collision = CollisionHelper::SphereSphereCollision(*allNodes[i], *allNodes[j], &cData);		
				 				
				if(collision)
				{
					allNodes[i]->WakeUp();
					allNodes[j]->WakeUp();
					
					CollisionHelper::AddCollisionImpulse(*allNodes[i], *allNodes[j], cData);
					++collisions;
				
					 if (allNodes[i]->HasLifeSpan())
					{
						allNodes[i]->DecreaseLife();
					}
					 if (allNodes[j]->HasLifeSpan())
					{
						allNodes[j]->DecreaseLife();
					}

				}
			}
			
			//if(typeA == COLLISION_SPHERE && typeB == COLLISION_PLANE)
			//{
			//	collision = CollisionHelper::SpherePlaneCollision(*allNodes[i], *allNodes[j], &cData);				
			//}

			if(typeA == COLLISION_PLANE && typeB == COLLISION_SPHERE)
			{
				if(allNodes[j]->GetRestedState() == true)
				{
				
					collision = CollisionHelper::SpherePlaneCollision(*allNodes[j], *allNodes[i], &cData);		
				
					if(collision)
					{
						//allNodes[i]->WakeUp();
						//allNodes[j]->WakeUp();
 						CollisionHelper::AddCollisionImpulse(*allNodes[j], *allNodes[i], cData);
						++collisions;
					
						if(allNodes[i]->HasLifeSpan())
						{
							allNodes[i]->DecreaseLife();
						}
						 if (allNodes[j]->HasLifeSpan())
						{
							allNodes[j]->DecreaseLife();
						}
					}
				}
			}

		/*	if(collision)
			{
				CollisionHelper::AddCollisionImpulse(*allNodes[i], *allNodes[j], cData);
			}*/
		}
	}
}

void	PhysicsSystem::AddNode(PhysicsNode* n) 
{
	listLock.lock();
	allNodes.push_back(n);
	listLock.unlock();
}

void	PhysicsSystem::RemoveNode(PhysicsNode* n) 
{
	listLock.lock();
	for(vector<PhysicsNode*>::iterator i = allNodes.begin(); i != allNodes.end(); ++i) 
	{
		if((*i) == n)
		{
			allNodes.erase(i);
			listLock.unlock();
			return;
		}
	}
	listLock.unlock();
}



bool PhysicsSystem::PointInConvexPolygon(const Vector3 testPosition, Vector3 * convexShapePoints, int numPointsL) const
{
	return false;
}

