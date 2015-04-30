#pragma once
#include "../../nclgl/Plane.h"
/*
Rich: 
There are a few ways of integrating the collision volumes
discussed in this module into your little physics engine.
You could keep pointers to all types inside a physics node,
and check to see which ones are NULL. Or you could make all
collision volumes inherit from a base class, so we only need
a single pointer inside each Physics Node. We can then either
use a process of dynamic casting to determine which exact type
of volume the pointer is, or just cheat and use an enum member
variable (I do this in my own work, you can do whichever you
feel comfortable with!).
*/
enum CollisionVolumeType
{
	COLLISION_SPHERE,
	COLLISION_AABB,
	COLLISION_PLANE
};


class CollisionVolume
{
public:
	virtual CollisionVolumeType GetType() { return type;}
	virtual float GetSize(){return temp;}

protected:
	CollisionVolumeType type;
	float temp;
};

class CollisionPlane : public CollisionVolume
{
public:
	
	//CollisionPlane(Vector3 normal, float distance): distance(distance), normal(normal) {type = COLLISION_PLANE;}

	CollisionVolumeType GetType() const {
		return COLLISION_PLANE;
	}

	float GetSize(){return distance;}

	//Vector3 GetNormal() const {
	//	return normal;
	//}

	//float GetDistance() const {
	//	return distance;
	//}

	//type = COLLISION_PLANE;
//	float distance;
//	Vector3 normal;
	CollisionPlane(const Plane p)
	{
		normal = p.GetNormal();
		distance = p.GetDistance();
		type = COLLISION_PLANE;
	}
	Vector3 normal;
	float distance;
};

class CollisionSphere : public CollisionVolume
{
public:
	CollisionSphere( float r)
	{
	//	m_pos = p;
		m_radius = r;
		type = COLLISION_SPHERE;
	}
	float GetSize(){return m_radius;}
	//Vector3 m_pos;
	float m_radius;
};

class CollisionAABB : public CollisionVolume
{
public:
	CollisionAABB(const Vector3& newHalfDims)
	{
		m_halfdims = newHalfDims;
		type = COLLISION_AABB;
		//m_pos = p;
	//	m_halfdims.x = m_pos.x * 0.5f; 
	//	m_halfdims.y = m_pos.y * 0.5f;
	//	m_halfdims.z = m_pos.z * 0.5f;

	}
	//Vector3 m_pos;
	Vector3 m_halfdims;
};

class CollisionData 
{
public:
	Vector3 m_point;
	Vector3 m_normal;
	float m_penetration;
};