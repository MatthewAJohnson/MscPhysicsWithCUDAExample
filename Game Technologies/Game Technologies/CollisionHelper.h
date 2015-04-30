#pragma once
#include "PhysicsNode.h"
#include "PhysicsHelper.h"
class CollisionHelper
{
public:
	static bool SphereSphereCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data = NULL);

	static bool SpherePlaneCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data = NULL);

	static void AddCollisionImpulse(PhysicsNode& p0, PhysicsNode& p1, CollisionData& data );

//	static volatile void InitialiseCollisionCounter(){collisionCounter = 0;}

//	static volatile int collisionCounter;

};

inline float LengthSq(Vector3 v)
{
	return Vector3::Dot(v,v);
}

